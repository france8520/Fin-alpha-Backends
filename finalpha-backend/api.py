"""
FinAlpha API — FastAPI backend
Endpoints:
  GET /analyze?ticker=AAPL  → risk metrics + FinBERT sentiment + Llama 3 summary
  GET /health               → health check for Render
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import requests as http_requests
import os
import uvicorn
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from threading import Lock
from groq import Groq

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
            raise ValueError(f"Empty data returned for {ticker}")
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "too many requests" in msg or "rate limit" in msg or "429" in msg
            if attempt < retries - 1 and is_rate_limit:
                wait = 2 ** (attempt + 1)
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
                return None
    return None

app = FastAPI(title="FinAlpha API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── API Keys & Models ─────────────────────────────────────────────────────────
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

logger.info("Using Hugging Face Inference API for FinBERT.")
if GROQ_API_KEY:
    logger.info("Groq API key found, Llama 3 summarization enabled.")
else:
    logger.warning("Groq API key not found, summarization will be disabled.")

# ── Helper: FinBERT sentiment via HF Inference API ───────────────────────────
def get_sentiment(texts: list[str]) -> dict:
    if not texts:
        return {"label": "neutral", "score": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}

    all_scores = {"positive": [], "negative": [], "neutral": []}
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    for text in texts[:10]:
        try:
            resp = http_requests.post(HF_API_URL, headers=headers, json={"inputs": text[:512]}, timeout=15)
            if resp.status_code != 200: continue
            results = resp.json()
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
        "label": label, "score": round(score, 4),
        "positive": round(avg["positive"], 4), "negative": round(avg["negative"], 4), "neutral": round(avg["neutral"], 4),
    }

# ── Helper: News summarization via Groq ──────────────────────────────────────
@lru_cache(maxsize=100)
def get_news_summary(news_items: tuple[str, ...]) -> str | None:
    if not GROQ_API_KEY or not news_items:
        return None

    client = Groq(api_key=GROQ_API_KEY)
    news_text = "\n".join(f"- {item}" for item in news_items)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a concise financial analyst. Summarize the key themes from these headlines in one brief paragraph (2-3 sentences) for an investor. Respond in the same language as the headlines."},
                {"role": "user", "content": f"Recent news headlines:\n{news_text}"}
            ],
            model="llama3-8b-8192",
            temperature=0.5, max_tokens=150,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[Groq] Failed to get summary: {e}")
        return None

# ── Helper: Risk metrics ──────────────────────────────────────────────────────
def calc_risk_metrics(ticker: str) -> dict:
    cached = cache_get(f"risk:{ticker}")
    if cached: return cached

    hist = yf_ticker_history(ticker)
    if hist.empty or len(hist) < 20:
        raise ValueError(f"Insufficient data for {ticker}")

    returns = hist["Close"].pct_change().dropna()
    vol = float(returns.std() * np.sqrt(252))

    try:
        spy_ret = yf_spy_returns()
        if spy_ret is not None:
            aligned = returns.align(spy_ret, join="inner")
            cov = np.cov(aligned[0], aligned[1])
            beta = float(cov[0, 1] / cov[1, 1])
        else: beta = 1.0
    except Exception: beta = 1.0

    rf = 0.05 / 252
    excess = returns - rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = float((returns.mean() * 252 - 0.05) / downside) if downside > 0 else 0.0
    var_95 = float(np.percentile(returns, 5))

    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = float(100 - 100 / (1 + rs.iloc[-1])) if rs.iloc[-1] != -1 else 0

    risk_score = min(100, int(vol * 60 + max(0, beta - 1) * 15 + max(0, 3.5 - sharpe) * 5 + max(0, rsi - 70) * 0.5))
    risk_label = "Low Risk" if risk_score < 30 else "Medium Risk" if risk_score < 60 else "High Risk"

    result = {
        "ticker": ticker.upper(), "price": round(float(hist["Close"].iloc[-1]), 2),
        "total_return": round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 2),
        "vol": f"{vol*100:.1f}%", "beta": round(beta, 2), "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2), "var": f"{var_95*100:.2f}%", "rsi": round(rsi, 1),
        "risk_score": risk_score, "risk_label": risk_label,
    }
    cache_set(f"risk:{ticker}", result)
    return result

# ── Helper: Sentiment → recommendation text ───────────────────────────────────
def sentiment_to_rec(sentiment: dict, risk: dict) -> str:
    label, score, rsi, risk_label = sentiment["label"], sentiment["score"], risk["rsi"], risk["risk_label"]
    if label == "positive" and score > 0.3 and rsi < 70: return f"Buy — Positive news sentiment ({score:+.2f}) with healthy momentum"
    if label == "positive" and rsi >= 70: return f"Hold — Positive sentiment but overbought RSI ({rsi}), wait for pullback"
    if label == "negative" and score < -0.3: return f"Sell / Avoid — Negative news sentiment ({score:+.2f}), {risk_label}"
    if label == "neutral": return f"Hold — Neutral sentiment ({score:+.2f}), monitor for clearer signal"
    return f"Hold — Mixed signals (sentiment: {score:+.2f}, RSI: {rsi})"

def normalize_news_item(item: dict) -> dict:
    content = item.get("content", {}) if isinstance(item, dict) else {}
    provider = content.get("provider", {})
    canonical_url = content.get("canonicalUrl", {})
    published_at = content.get("pubDate") or item.get("providerPublishTime")
    if isinstance(published_at, (int, float)):
        published_at = datetime.fromtimestamp(published_at, tz=timezone.utc).isoformat()
    return {
        "title": content.get("title") or item.get("title", ""),
        "summary": content.get("summary") or item.get("summary", ""),
        "publisher": (provider.get("displayName") if isinstance(provider, dict) else None) or item.get("publisher", ""),
        "link": (canonical_url.get("url") if isinstance(canonical_url, dict) else None) or item.get("link", ""),
        "published_at": published_at,
    }

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
        risk = calc_risk_metrics(ticker)
        stock = yf.Ticker(ticker)
        news = stock.news or []
        texts, news_items = [], []
        for item in news:
            normalized = normalize_news_item(item)
            text = (normalized["title"] + ". " + normalized["summary"]).strip()
            if text:
                texts.append(text)
                news_items.append(normalized)

        sentiment = get_sentiment(texts)
        rec = sentiment_to_rec(sentiment, risk)
        news_summary = get_news_summary(tuple(n["title"] for n in news_items))

        return {
            **risk,
            "sentiment": sentiment,
            "recommendation": rec,
            "news_summary": news_summary,
            "news_count": len(texts),
            "news_items": news_items[:6],
        }
    except ValueError as e:
        logger.error(f"[{ticker}] ValueError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[{ticker}] Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)