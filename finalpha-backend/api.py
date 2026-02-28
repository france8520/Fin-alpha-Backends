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
import torch
import scipy.special
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache
import uvicorn

app = FastAPI(title="FinAlpha API", version="1.0.0")

# Allow all origins (ปรับเป็น domain จริงตอน production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Load FinBERT once at startup ──────────────────────────────────────────────
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert   = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert.eval()
print("FinBERT ready.")


# ── Helper: FinBERT sentiment ─────────────────────────────────────────────────
def get_sentiment(texts: list[str]) -> dict:
    """รับ list of text → คืน avg pos/neg/neu score และ label"""
    if not texts:
        return {"label": "neutral", "score": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    all_scores = {"positive": [], "negative": [], "neutral": []}

    with torch.no_grad():
        for text in texts[:20]:  # จำกัด 20 ข่าวเพื่อความเร็ว
            inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = finbert(**inputs)
            probs   = scipy.special.softmax(outputs.logits.numpy().squeeze())
            labels  = finbert.config.id2label

            for idx, prob in enumerate(probs):
                all_scores[labels[idx]].append(float(prob))

    avg = {k: float(np.mean(v)) for k, v in all_scores.items()}
    label = max(avg, key=avg.get)
    score = avg["positive"] - avg["negative"]  # -1 ถึง +1

    return {
        "label":    label,
        "score":    round(score, 4),
        "positive": round(avg["positive"], 4),
        "negative": round(avg["negative"], 4),
        "neutral":  round(avg["neutral"], 4),
    }


# ── Helper: Risk metrics ──────────────────────────────────────────────────────
def calc_risk_metrics(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    hist  = stock.history(period="1y")

    if hist.empty or len(hist) < 20:
        raise ValueError(f"Insufficient data for {ticker}")

    returns = hist["Close"].pct_change().dropna()

    # Volatility (annualized)
    vol = float(returns.std() * np.sqrt(252))

    # Beta vs SPY
    try:
        spy     = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
        spy.columns = spy.columns.get_level_values(0)
        spy_ret = spy["Close"].pct_change().dropna()
        aligned = returns.align(spy_ret, join="inner")
        cov     = np.cov(aligned[0], aligned[1])
        beta    = float(cov[0, 1] / cov[1, 1])
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

    return {
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
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

# ── Chart Data Endpoint ───────────────────────────────────────────────────────
@app.get("/chart/{ticker}")
def chart(ticker: str, period: str = "1y"):
    """
    คืน OHLCV + indicators สำหรับ chart
    period: 1mo, 3mo, 6mo, 1y, 2y, 5y
    """
    import math
    VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}
    ticker = ticker.strip().upper()
    if period not in VALID_PERIODS:
        period = "1y"

    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=period)

        if hist.empty or len(hist) < 5:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        hist.index = hist.index.tz_localize(None)

        # EMA
        hist["ema20"] = hist["Close"].ewm(span=20).mean()
        hist["ema50"] = hist["Close"].ewm(span=50).mean()

        # Bollinger Bands
        sma20 = hist["Close"].rolling(20).mean()
        std20 = hist["Close"].rolling(20).std()
        hist["bb_upper"] = sma20 + 2 * std20
        hist["bb_lower"] = sma20 - 2 * std20
        hist["bb_mid"]   = sma20

        # RSI 14
        delta = hist["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        hist["rsi"] = 100 - 100 / (1 + rs)

        # MACD
        ema12 = hist["Close"].ewm(span=12).mean()
        ema26 = hist["Close"].ewm(span=26).mean()
        hist["macd"]        = ema12 - ema26
        hist["macd_signal"] = hist["macd"].ewm(span=9).mean()
        hist["macd_hist"]   = hist["macd"] - hist["macd_signal"]

        # Volume SMA
        hist["vol_sma20"] = hist["Volume"].rolling(20).mean()

        def clean(val):
            try:
                f = float(val)
                return None if math.isnan(f) else round(f, 4)
            except:
                return None

        rows = []
        for date, row in hist.iterrows():
            rows.append({
                "date":        date.strftime("%Y-%m-%d"),
                "open":        clean(row["Open"]),
                "high":        clean(row["High"]),
                "low":         clean(row["Low"]),
                "close":       clean(row["Close"]),
                "volume":      int(row["Volume"]) if not math.isnan(float(row["Volume"])) else 0,
                "ema20":       clean(row["ema20"]),
                "ema50":       clean(row["ema50"]),
                "bb_upper":    clean(row["bb_upper"]),
                "bb_lower":    clean(row["bb_lower"]),
                "bb_mid":      clean(row["bb_mid"]),
                "rsi":         clean(row["rsi"]),
                "macd":        clean(row["macd"]),
                "macd_signal": clean(row["macd_signal"]),
                "macd_hist":   clean(row["macd_hist"]),
                "vol_sma20":   clean(row["vol_sma20"]),
            })

        current_price = clean(hist["Close"].iloc[-1])
        prev_close    = clean(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        change_pct    = round((current_price - prev_close) / prev_close * 100, 2) if prev_close else 0

        return {
            "ticker":        ticker,
            "period":        period,
            "current_price": current_price,
            "change_pct":    change_pct,
            "high_52w":      clean(hist["High"].max()),
            "low_52w":       clean(hist["Low"].min()),
            "avg_volume":    int(hist["Volume"].mean()),
            "data":          rows,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
