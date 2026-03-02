"""
FinAlpha API — FastAPI backend
Endpoints:
  GET /analyze?ticker=AAPL  → risk metrics + FinBERT sentiment
  GET /health               → health check for Render
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import requests as http_requests
import os
import uvicorn
import logging
import time
from functools import lru_cache
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finalpha")

# ── Simple TTL Cache ─────────────────────────────────────────────────────────
_cache: dict = {}
_cache_lock = Lock()
CACHE_TTL = 15 * 60  # 15 นาที

def cache_get(key: str):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry["ts"] < CACHE_TTL:
            logger.info(f"[CACHE HIT] {key}")
            return entry["data"]
    return None

def cache_set(key: str, data):
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}

# ── Retry helper สำหรับ yfinance (rate limit / network error) ─────────────────
def yf_ticker_history(ticker: str, period: str = "1y", retries: int = 3):
    """ดึง history พร้อม retry backoff เมื่อโดน rate limit"""
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period=period)
            if not hist.empty:
                return hist
            # hist ว่างแต่ไม่ error — อาจโดน block เงียบๆ
            raise ValueError(f"Empty data returned for {ticker}")
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "too many requests" in msg or "rate limit" in msg or "429" in msg
            if attempt < retries - 1 and is_rate_limit:
                wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                logger.warning(f"[{ticker}] Rate limited, retry {attempt+1}/{retries} in {wait}s")
                time.sleep(wait)
            else:
                raise
    raise ValueError(f"Failed to fetch data for {ticker} after {retries} retries")

def yf_spy_returns(retries: int = 3):
    """ดึง SPY returns พร้อม retry backoff"""
    cached = cache_get("spy_returns")
    if cached is not None:
        return cached
    for attempt in range(retries):
        try:
            spy = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
            spy.columns = spy.columns.get_level_values(0)
            ret = spy["Close"].pct_change().dropna()
            cache_set("spy_returns", ret)
            return ret
        except Exception as e:
            msg = str(e).lower()
            if attempt < retries - 1 and ("too many requests" in msg or "429" in msg):
                wait = 2 ** (attempt + 1)
                logger.warning(f"[SPY] Rate limited, retry {attempt+1}/{retries} in {wait}s")
                time.sleep(wait)
            else:
                return None  # Beta fallback = 1.0

app = FastAPI(title="FinAlpha API", version="1.0.0")

# Allow all origins (ปรับเป็น domain จริงตอน production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Hugging Face Inference API ────────────────────────────────────────────────
# ไม่โหลด model ลงเครื่อง — เรียก HF API แทน ประหยัด RAM ไปได้ ~400MB
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")  # ใส่ใน Render Environment Variables
print("Using Hugging Face Inference API for FinBERT.")


# ── Helper: FinBERT sentiment via HF Inference API ───────────────────────────
def get_sentiment(texts: list[str]) -> dict:
    """เรียก Hugging Face Inference API — ไม่โหลด model ลงเครื่อง"""
    if not texts:
        return {"label": "neutral", "score": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    all_scores = {"positive": [], "negative": [], "neutral": []}
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    for text in texts[:10]:  # จำกัด 10 ข่าว เพื่อไม่ให้ช้าเกิน
        try:
            resp = http_requests.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": text[:512]},
                timeout=15
            )
            if resp.status_code != 200:
                continue
            results = resp.json()
            # HF คืน [[{label, score}, ...]]
            if isinstance(results, list) and isinstance(results[0], list):
                results = results[0]
            for item in results:
                lbl = item["label"].lower()
                if lbl in all_scores:
                    all_scores[lbl].append(float(item["score"]))
        except Exception:
            continue

    if not any(all_scores.values()):
        return {"label": "neutral", "score": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    avg = {k: float(np.mean(v)) if v else 0.0 for k, v in all_scores.items()}
    label = max(avg, key=avg.get)
    score = avg["positive"] - avg["negative"]

    return {
        "label":    label,
        "score":    round(score, 4),
        "positive": round(avg["positive"], 4),
        "negative": round(avg["negative"], 4),
        "neutral":  round(avg["neutral"], 4),
    }


# ── Helper: Risk metrics ──────────────────────────────────────────────────────
def calc_risk_metrics(ticker: str) -> dict:
    # Check cache first
    cached = cache_get(f"risk:{ticker}")
    if cached:
        return cached

    hist = yf_ticker_history(ticker)  # retry built-in
    logger.info(f"[{ticker}] history rows: {len(hist)}")

    if hist.empty or len(hist) < 20:
        raise ValueError(f"Insufficient data for {ticker}")

    returns = hist["Close"].pct_change().dropna()

    # Volatility (annualized)
    vol = float(returns.std() * np.sqrt(252))

    # Beta vs SPY (with cache + retry)
    try:
        spy_ret = yf_spy_returns()
        if spy_ret is not None:
            aligned = returns.align(spy_ret, join="inner")
            cov  = np.cov(aligned[0], aligned[1])
            beta = float(cov[0, 1] / cov[1, 1])
        else:
            beta = 1.0
    except Exception:
        beta = 1.0

    # Sharpe (rf = 5%)
    rf     = 0.05 / 252
    excess = returns - rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    # Sortino
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino  = float((returns.mean() * 252 - 0.05) / downside) if downside > 0 else 0.0

    # VaR 95%
    var_95 = float(np.percentile(returns, 5))

    # RSI
    delta = hist["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    rsi   = float(100 - 100 / (1 + rs.iloc[-1]))

    # Risk score 0–100
    risk_score = min(100, int(
        vol * 60 +
        max(0, beta - 1) * 15 +
        max(0, 3.5 - sharpe) * 5 +
        max(0, rsi - 70) * 0.5
    ))

    if risk_score < 30:
        risk_label = "Low Risk"
    elif risk_score < 60:
        risk_label = "Medium Risk"
    else:
        risk_label = "High Risk"

    price = float(hist["Close"].iloc[-1])
    total_return = float((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1))

    result = {
        "ticker":       ticker.upper(),
        "price":        round(price, 2),
        "total_return": round(total_return * 100, 2),
        "vol":          f"{vol*100:.1f}%",
        "beta":         round(beta, 2),
        "sharpe":       round(sharpe, 2),
        "sortino":      round(sortino, 2),
        "var":          f"{var_95*100:.2f}%",
        "rsi":          round(rsi, 1),
        "risk_score":   risk_score,
        "risk_label":   risk_label,
    }
    cache_set(f"risk:{ticker}", result)
    return result


# ── Helper: Sentiment → recommendation text ───────────────────────────────────
def sentiment_to_rec(sentiment: dict, risk: dict) -> str:
    label = sentiment["label"]
    score = sentiment["score"]
    rsi   = risk["rsi"]
    risk_label = risk["risk_label"]

    if label == "positive" and score > 0.3 and rsi < 70:
        return f"Buy — Positive news sentiment ({score:+.2f}) with healthy momentum"
    elif label == "positive" and rsi >= 70:
        return f"Hold — Positive sentiment but overbought RSI ({rsi}), wait for pullback"
    elif label == "negative" and score < -0.3:
        return f"Sell / Avoid — Negative news sentiment ({score:+.2f}), {risk_label}"
    elif label == "neutral":
        return f"Hold — Neutral sentiment ({score:+.2f}), monitor for clearer signal"
    else:
        return f"Hold — Mixed signals (sentiment: {score:+.2f}, RSI: {rsi})"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/analyze")
def analyze(ticker: str):
    ticker = ticker.strip().upper()

    if not ticker or len(ticker) > 10:
        raise HTTPException(status_code=400, detail="Invalid ticker")

    try:
        # 1. Risk metrics
        risk = calc_risk_metrics(ticker)

        # 2. News sentiment
        stock  = yf.Ticker(ticker)
        news   = stock.news or []
        texts  = []
        for item in news:
            content = item.get("content", {})
            title   = content.get("title", item.get("title", ""))
            summary = content.get("summary", "")
            text    = (title + ". " + summary).strip()
            if text:
                texts.append(text)

        sentiment = get_sentiment(texts)

        # 3. Recommendation
        rec = sentiment_to_rec(sentiment, risk)

        return {
            **risk,
            "sentiment":      sentiment,
            "recommendation": rec,
            "news_count":     len(texts),
        }

    except ValueError as e:
        logger.error(f"[{ticker}] ValueError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[{ticker}] Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


# ── Helper: calculate indicators ──────────────────────────────────────────────
def _ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def _bollinger(series, n=20, k=2):
    mid   = series.rolling(n).mean()
    std   = series.rolling(n).std()
    return mid + k*std, mid, mid - k*std

def _rsi(series, n=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / loss
    return 100 - 100 / (1 + rs)

def _macd(series, fast=12, slow=26, signal=9):
    macd_line   = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    hist        = macd_line - signal_line
    return macd_line, signal_line, hist


@app.get("/chart/{ticker}")
def chart(ticker: str, period: str = "1y"):
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 10:
        raise HTTPException(status_code=400, detail="Invalid ticker")

    cache_key = f"chart:{ticker}:{period}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        hist = yf_ticker_history(ticker, period=period)
        if hist.empty or len(hist) < 5:
            raise ValueError(f"Insufficient data for {ticker}")

        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"] if "Volume" in hist.columns else None

        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        bb_upper, bb_mid, bb_lower = _bollinger(close)
        rsi_vals = _rsi(close)
        macd_line, signal_line, macd_hist = _macd(close)
        vol_sma20 = volume.rolling(20).mean() if volume is not None else None

        rows = []
        for i, (idx, row) in enumerate(hist.iterrows()):
            date_str = idx.strftime("%Y-%m-%d")
            rows.append({
                "date":       date_str,
                "open":       round(float(row["Open"]),  4),
                "high":       round(float(row["High"]),  4),
                "low":        round(float(row["Low"]),   4),
                "close":      round(float(row["Close"]), 4),
                "volume":     int(row["Volume"]) if volume is not None and not np.isnan(row["Volume"]) else 0,
                "ema20":      round(float(ema20.iloc[i]),    4) if not np.isnan(ema20.iloc[i])    else None,
                "ema50":      round(float(ema50.iloc[i]),    4) if not np.isnan(ema50.iloc[i])    else None,
                "bb_upper":   round(float(bb_upper.iloc[i]),4) if not np.isnan(bb_upper.iloc[i]) else None,
                "bb_mid":     round(float(bb_mid.iloc[i]),  4) if not np.isnan(bb_mid.iloc[i])   else None,
                "bb_lower":   round(float(bb_lower.iloc[i]),4) if not np.isnan(bb_lower.iloc[i]) else None,
                "rsi":        round(float(rsi_vals.iloc[i]),2) if not np.isnan(rsi_vals.iloc[i]) else None,
                "macd":       round(float(macd_line.iloc[i]),4)   if not np.isnan(macd_line.iloc[i])   else None,
                "macd_signal":round(float(signal_line.iloc[i]),4) if not np.isnan(signal_line.iloc[i]) else None,
                "macd_hist":  round(float(macd_hist.iloc[i]),4)   if not np.isnan(macd_hist.iloc[i])   else None,
                "vol_sma20":  round(float(vol_sma20.iloc[i]),2)   if vol_sma20 is not None and not np.isnan(vol_sma20.iloc[i]) else None,
            })

        current_price = round(float(close.iloc[-1]), 2)
        prev_close    = round(float(close.iloc[-2]), 2) if len(close) > 1 else current_price
        change_pct    = round((current_price - prev_close) / prev_close * 100, 2)

        result = {
            "ticker":        ticker,
            "period":        period,
            "current_price": current_price,
            "change_pct":    change_pct,
            "high_52w":      round(float(high.max()), 2),
            "low_52w":       round(float(low.min()),  2),
            "avg_volume":    int(volume.mean()) if volume is not None else 0,
            "data":          rows,
        }
        cache_set(cache_key, result)
        return result

    except ValueError as e:
        logger.error(f"[chart:{ticker}] ValueError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[chart:{ticker}] Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chart failed: {str(e)}")