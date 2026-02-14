# Research & Sentiment Intelligence Dashboard — Technical Implementation Spec

> **Context**: This is a hackathon project. The research module is one piece of a larger finance platform. This document is a comprehensive prompt for a coding agent to build the research module end-to-end. Every decision has been made — the agent should execute, not deliberate.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Backend Implementation](#3-backend-implementation)
4. [Data Sources & API Integration](#4-data-sources--api-integration)
5. [Sentiment Analysis Pipeline](#5-sentiment-analysis-pipeline)
6. [Browser-Based Research Agents](#6-browser-based-research-agents)
7. [Frontend Implementation](#7-frontend-implementation)
8. [Data Models & Schemas](#8-data-models--schemas)
9. [API Endpoints](#9-api-endpoints)
10. [Environment & Configuration](#10-environment--configuration)
11. [File Structure](#11-file-structure)
12. [Implementation Order](#12-implementation-order)

---

## 1. Project Overview

### What This Is
A financial research dashboard that aggregates market data, sentiment analysis, prediction market odds, AI-generated research summaries, and institutional-grade data into a single Yahoo Finance-style interface. The user enters a ticker symbol and gets a comprehensive research page with actionable data.

### Core Value Proposition
"Why open 6 tabs when this does it all?" — One unified interface that combines:
- Real market data (prices, fundamentals, technicals)
- AI-powered research summaries with citations (Perplexity Sonar)
- Social sentiment from Reddit
- Prediction market implied probabilities (Kalshi, Polymarket)
- Institutional research data scraped via browser agents (Finviz, Morningstar)
- Professional-grade charting (TradingView-style)

### What Makes This Impressive
- Perplexity Sonar acts as an AI research analyst — summarizes current sentiment with cited sources
- Prediction market data is a unique signal most finance tools don't show
- TradingView-quality charts via `lightweight-charts` library
- Browser agent scraping institutional data demonstrates agentic AI capability
- Everything unified in one polished UI

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Ticker   │ │ Charts   │ │Sentiment │ │Predict │ │
│  │ Overview │ │ Module   │ │ Panel    │ │Markets │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP/REST
┌──────────────────────▼──────────────────────────────┐
│                 FastAPI Backend                       │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐ │
│  │ Market  │ │Sentiment │ │Browser │ │ Prediction│ │
│  │ Data    │ │ Pipeline │ │ Agent  │ │ Market    │ │
│  │ Service │ │ Service  │ │Service │ │ Service   │ │
│  └────┬────┘ └────┬─────┘ └───┬────┘ └─────┬─────┘ │
│       │           │           │             │        │
│  ┌────▼────┐ ┌────▼─────┐ ┌──▼─────┐ ┌────▼──────┐│
│  │yfinance │ │Perplexity│ │Finviz/ │ │Kalshi/    ││
│  │FRED API │ │Reddit    │ │Morning │ │Polymarket ││
│  │         │ │VADER     │ │star    │ │           ││
│  └─────────┘ └──────────┘ └────────┘ └───────────┘│
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  SQLite + Cache │
              │  (in-memory)    │
              └─────────────────┘
```

### Tech Stack

| Layer | Technology | Version / Notes |
|-------|-----------|----------------|
| **Backend Framework** | FastAPI | Python 3.11+, async endpoints |
| **Market Data** | `yfinance` | Free, no API key needed |
| **Macro Data** | FRED API | Free API key from fred.stlouisfed.org |
| **AI Research** | Perplexity Sonar API (`sonar-pro`) | `sonar-pro` model, returns cited summaries |
| **Reddit Data** | PRAW (Python Reddit API Wrapper) | Free Reddit API credentials |
| **Sentiment Scoring** | VADER (`nltk.sentiment.vader`) | No GPU, instant, good for social media text |
| **Browser Agent** | Playwright (Python) | Headless Chromium, scrape Finviz + Morningstar |
| **Prediction Markets** | Kalshi API + Polymarket Gamma API | Public endpoints, no auth needed for reads |
| **Database** | SQLite | Single file, zero config |
| **Caching** | `cachetools` (Python in-memory TTL cache) | Avoid re-fetching same ticker within 5 min |
| **Frontend Framework** | React 18 (Vite) | Fast HMR, TypeScript |
| **Charting** | `lightweight-charts` (TradingView) | Candlestick, line, area, volume, histogram |
| **Additional Charts** | Recharts | Sentiment gauges, bar charts, treemaps |
| **UI Components** | Tailwind CSS + shadcn/ui | Professional look with minimal effort |
| **HTTP Client (frontend)** | Axios or `fetch` | For calling backend API |
| **HTTP Client (backend)** | `httpx` (async) | For calling external APIs |

---

## 3. Backend Implementation

### 3.1 FastAPI Application Setup

Create the FastAPI app with CORS enabled for the React frontend. Use async throughout.

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Research Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3.2 Caching Strategy

Use `cachetools.TTLCache` to avoid hammering APIs. Cache per ticker symbol.

```python
from cachetools import TTLCache

# Cache config: max 100 entries, 5-minute TTL
market_data_cache = TTLCache(maxsize=100, ttl=300)
sentiment_cache = TTLCache(maxsize=100, ttl=600)      # 10 min for sentiment
research_cache = TTLCache(maxsize=50, ttl=900)         # 15 min for AI research
prediction_cache = TTLCache(maxsize=50, ttl=300)
finviz_cache = TTLCache(maxsize=100, ttl=1800)         # 30 min for scraped data
```

### 3.3 Service Layer Pattern

Each data source gets its own service module. Services are async where possible and handle their own error handling + caching. The API route layer just orchestrates calls to services and returns unified responses.

---

## 4. Data Sources & API Integration

### 4.1 Market Data — yfinance

`yfinance` is the primary market data source. It's free, requires no API key, and provides everything needed.

**Install**: `pip install yfinance`

**Data to extract per ticker**:

```python
import yfinance as yf

def get_ticker_data(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    info = ticker.info
    
    # Extract these fields:
    return {
        # Price Data
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose"),
        "open": info.get("open") or info.get("regularMarketOpen"),
        "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
        "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "volume": info.get("volume") or info.get("regularMarketVolume"),
        "avg_volume": info.get("averageVolume"),
        "market_cap": info.get("marketCap"),
        
        # Valuation Ratios
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("pegRatio"),
        "price_to_book": info.get("priceToBook"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        
        # Fundamentals
        "eps": info.get("trailingEps"),
        "forward_eps": info.get("forwardEps"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "profit_margin": info.get("profitMargins"),
        "revenue": info.get("totalRevenue"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "debt_to_equity": info.get("debtToEquity"),
        "return_on_equity": info.get("returnOnEquity"),
        "free_cash_flow": info.get("freeCashflow"),
        
        # Company Info
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "description": info.get("longBusinessSummary"),
        "website": info.get("website"),
        "employees": info.get("fullTimeEmployees"),
        "exchange": info.get("exchange"),
        "currency": info.get("currency"),
    }
```

**Historical price data for charts**:

```python
def get_price_history(symbol: str, period: str = "1y", interval: str = "1d") -> list[dict]:
    """
    Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    Intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    Note: intraday intervals (1m-90m) only work with short periods (≤7 days)
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    
    # Convert to list of dicts for JSON serialization
    # Format required by lightweight-charts:
    records = []
    for index, row in hist.iterrows():
        records.append({
            "time": index.strftime("%Y-%m-%d") if interval in ["1d","5d","1wk","1mo","3mo"] 
                    else int(index.timestamp()),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"]),
        })
    return records
```

**Technical indicators** — compute these server-side using `pandas`:

```python
import pandas as pd

def compute_technicals(hist_df: pd.DataFrame) -> dict:
    close = hist_df["Close"]
    
    # Simple Moving Averages
    sma_20 = close.rolling(window=20).mean().iloc[-1]
    sma_50 = close.rolling(window=50).mean().iloc[-1]
    sma_200 = close.rolling(window=200).mean().iloc[-1]
    
    # Exponential Moving Averages
    ema_12 = close.ewm(span=12).mean().iloc[-1]
    ema_26 = close.ewm(span=26).mean().iloc[-1]
    
    # RSI (14-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_current = rsi.iloc[-1]
    
    # MACD
    macd_line = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    signal_line = macd_line.ewm(span=9).mean()
    macd_histogram = macd_line - signal_line
    
    # Bollinger Bands (20-period, 2 std dev)
    bb_middle = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    # Volatility (30-day annualized)
    returns = close.pct_change()
    volatility_30d = returns.tail(30).std() * (252 ** 0.5)
    
    # Generate signal
    current_price = close.iloc[-1]
    signals = []
    if current_price > sma_50:
        signals.append("Above 50 SMA (Bullish)")
    else:
        signals.append("Below 50 SMA (Bearish)")
    if rsi_current > 70:
        signals.append("RSI Overbought")
    elif rsi_current < 30:
        signals.append("RSI Oversold")
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        signals.append("MACD Bullish Crossover")
    else:
        signals.append("MACD Bearish Crossover")
    
    return {
        "sma_20": round(sma_20, 2),
        "sma_50": round(sma_50, 2),
        "sma_200": round(sma_200, 2),
        "ema_12": round(ema_12, 2),
        "ema_26": round(ema_26, 2),
        "rsi": round(rsi_current, 2),
        "macd": round(macd_line.iloc[-1], 4),
        "macd_signal": round(signal_line.iloc[-1], 4),
        "macd_histogram": round(macd_histogram.iloc[-1], 4),
        "bollinger_upper": round(bb_upper.iloc[-1], 2),
        "bollinger_middle": round(bb_middle.iloc[-1], 2),
        "bollinger_lower": round(bb_lower.iloc[-1], 2),
        "volatility_30d": round(volatility_30d, 4),
        "signals": signals,
        
        # Full series for chart overlays
        "sma_50_series": [
            {"time": idx.strftime("%Y-%m-%d"), "value": round(v, 2)}
            for idx, v in close.rolling(window=50).mean().dropna().items()
        ],
        "sma_200_series": [
            {"time": idx.strftime("%Y-%m-%d"), "value": round(v, 2)}
            for idx, v in close.rolling(window=200).mean().dropna().items()
        ],
        "rsi_series": [
            {"time": idx.strftime("%Y-%m-%d"), "value": round(v, 2)}
            for idx, v in rsi.dropna().items()
        ],
        "macd_series": [
            {"time": idx.strftime("%Y-%m-%d"), "value": round(v, 4)}
            for idx, v in macd_line.dropna().items()
        ],
        "macd_signal_series": [
            {"time": idx.strftime("%Y-%m-%d"), "value": round(v, 4)}
            for idx, v in signal_line.dropna().items()
        ],
    }
```

### 4.2 Macro Data — FRED API

**Install**: `pip install fredapi`

Pull key macro indicators to display alongside ticker analysis. These provide context for the broader market environment.

```python
from fredapi import Fred

fred = Fred(api_key=FRED_API_KEY)

def get_macro_indicators() -> dict:
    """Fetch key macro indicators. Cache these for 1 hour."""
    return {
        "fed_funds_rate": {
            "value": fred.get_series("FEDFUNDS").iloc[-1],
            "name": "Federal Funds Rate",
            "unit": "%"
        },
        "cpi_yoy": {
            "value": fred.get_series("CPIAUCSL").pct_change(12).iloc[-1] * 100,
            "name": "CPI (YoY)",
            "unit": "%"
        },
        "unemployment": {
            "value": fred.get_series("UNRATE").iloc[-1],
            "name": "Unemployment Rate",
            "unit": "%"
        },
        "ten_year_treasury": {
            "value": fred.get_series("DGS10").dropna().iloc[-1],
            "name": "10Y Treasury Yield",
            "unit": "%"
        },
        "vix": {
            "value": fred.get_series("VIXCLS").dropna().iloc[-1],
            "name": "VIX (Fear Index)",
            "unit": ""
        },
        "gdp_growth": {
            "value": fred.get_series("A191RL1Q225SBEA").iloc[-1],
            "name": "GDP Growth (QoQ Annualized)",
            "unit": "%"
        }
    }
```

### 4.3 Perplexity Sonar API — AI Research Analyst

This is the centerpiece of the research feature. One API call returns a comprehensive, cited summary of current market sentiment and news for any ticker.

**API**: `https://api.perplexity.ai/chat/completions`
**Model**: `sonar-pro` (best quality, returns citations)
**Auth**: Bearer token

```python
import httpx

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

async def get_ai_research_summary(symbol: str, company_name: str) -> dict:
    """
    Calls Perplexity Sonar to get an AI-generated research summary.
    Returns structured sentiment + summary + citations.
    """
    prompt = f"""Analyze the current market sentiment and recent news for {company_name} ({symbol}). 
    
    Provide:
    1. OVERALL SENTIMENT: Rate as Strongly Bullish, Bullish, Neutral, Bearish, or Strongly Bearish
    2. SENTIMENT SCORE: A number from -100 (extremely bearish) to +100 (extremely bullish)
    3. KEY DRIVERS: What are the 3-5 most important factors driving the stock right now?
    4. RECENT NEWS: Summarize the 3-5 most impactful recent news items
    5. ANALYST CONSENSUS: What are analysts saying? Any recent upgrades/downgrades?
    6. RISKS: What are the top 2-3 risks to watch?
    7. CATALYSTS: What upcoming events could move the stock?
    
    Be specific with numbers, dates, and names. Focus on the last 1-2 weeks of activity."""
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            PERPLEXITY_API_URL,
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a senior equity research analyst. Provide detailed, data-driven analysis. Always include specific numbers, dates, and source attributions."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            timeout=30.0,
        )
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", [])  # Sonar returns source URLs
        
        return {
            "summary": content,
            "citations": citations,  # List of source URLs
            "model": "sonar-pro",
        }
```

**Important Perplexity Sonar notes for the agent**:
- The `citations` field in the response is a list of URLs that correspond to `[1]`, `[2]`, etc. in the response text
- Display these as clickable links in the UI next to the summary
- The API key goes in the `Authorization: Bearer <key>` header
- Use `sonar-pro` for best quality; `sonar` is cheaper but less detailed
- Rate limit: be aware of the plan's rate limits, cache aggressively

### 4.4 Reddit Sentiment — PRAW

**Install**: `pip install praw`

Pull recent posts and comments mentioning a ticker from finance subreddits.

```python
import praw
from datetime import datetime, timedelta

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent="research-dashboard/1.0"
)

FINANCE_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options", "stockmarket"]

def get_reddit_sentiment(symbol: str, limit: int = 25) -> dict:
    """
    Search Reddit for mentions of a ticker symbol.
    Returns posts with metadata + individual sentiment scores.
    """
    posts = []
    
    for subreddit_name in FINANCE_SUBREDDITS:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Search for the ticker symbol (with $ prefix and without)
        for query in [f"${symbol}", symbol]:
            try:
                results = subreddit.search(query, sort="new", time_filter="week", limit=limit // len(FINANCE_SUBREDDITS))
                for post in results:
                    posts.append({
                        "title": post.title,
                        "selftext": post.selftext[:500] if post.selftext else "",
                        "subreddit": subreddit_name,
                        "score": post.score,  # upvotes
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(post.created_utc).isoformat(),
                        "url": f"https://reddit.com{post.permalink}",
                        "author": str(post.author) if post.author else "[deleted]",
                    })
            except Exception:
                continue  # Skip if subreddit is unavailable
    
    # Deduplicate by URL
    seen = set()
    unique_posts = []
    for p in posts:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique_posts.append(p)
    
    # Sort by Reddit score (upvotes) descending
    unique_posts.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "posts": unique_posts[:20],  # Top 20 posts
        "total_found": len(unique_posts),
        "subreddits_searched": FINANCE_SUBREDDITS,
    }
```

### 4.5 VADER Sentiment Scoring

**Install**: `pip install nltk` (then download vader lexicon)

Run VADER on each Reddit post to get a sentiment score.

```python
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

def score_reddit_posts(posts: list[dict]) -> dict:
    """
    Run VADER sentiment on each Reddit post.
    Returns individual scores + aggregate metrics.
    """
    scored_posts = []
    all_compounds = []
    
    for post in posts:
        text = f"{post['title']}. {post['selftext']}"
        scores = sid.polarity_scores(text)
        compound = scores["compound"]  # -1 to +1
        all_compounds.append(compound)
        
        # Classify
        if compound >= 0.05:
            label = "Bullish"
        elif compound <= -0.05:
            label = "Bearish"
        else:
            label = "Neutral"
        
        scored_posts.append({
            **post,
            "sentiment_score": round(compound, 3),
            "sentiment_label": label,
            "sentiment_detail": {
                "positive": round(scores["pos"], 3),
                "negative": round(scores["neg"], 3),
                "neutral": round(scores["neu"], 3),
            }
        })
    
    # Aggregate
    if all_compounds:
        avg_sentiment = sum(all_compounds) / len(all_compounds)
        bullish_count = sum(1 for c in all_compounds if c >= 0.05)
        bearish_count = sum(1 for c in all_compounds if c <= -0.05)
        neutral_count = len(all_compounds) - bullish_count - bearish_count
    else:
        avg_sentiment = 0
        bullish_count = bearish_count = neutral_count = 0
    
    return {
        "posts": scored_posts,
        "aggregate": {
            "average_sentiment": round(avg_sentiment, 3),
            "sentiment_label": "Bullish" if avg_sentiment >= 0.05 else "Bearish" if avg_sentiment <= -0.05 else "Neutral",
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "total_posts": len(scored_posts),
            # Scale to -100 to +100 for display consistency with Perplexity
            "score_normalized": round(avg_sentiment * 100, 1),
        }
    }
```

### 4.6 Prediction Markets — Kalshi API

Kalshi provides event contracts. Fetch markets related to macro events (Fed rate decisions, economic indicators, etc.).

**API Base**: `https://api.elections.kalshi.com/trade-api/v2` (public, no auth needed for reading)

```python
async def get_kalshi_markets(query: str = "fed") -> list[dict]:
    """
    Search Kalshi for relevant prediction market contracts.
    Useful queries: "fed", "rate", "recession", "inflation", "gdp", "unemployment", "s&p"
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.elections.kalshi.com/trade-api/v2/markets",
            params={
                "status": "open",
                "series_ticker": "",  # leave empty for search
                "limit": 20,
            },
            timeout=15.0,
        )
        data = response.json()
        
        markets = []
        for market in data.get("markets", []):
            # Filter for finance-relevant markets
            title = market.get("title", "").lower()
            if any(kw in title for kw in ["fed", "rate", "inflation", "recession", "gdp", "s&p", "nasdaq", "unemployment", "cpi"]):
                yes_price = market.get("yes_ask", 0) or market.get("last_price", 0)
                markets.append({
                    "title": market.get("title"),
                    "ticker": market.get("ticker"),
                    "yes_price": yes_price,  # Probability in cents (50 = 50%)
                    "implied_probability": f"{yes_price}%",
                    "volume": market.get("volume"),
                    "open_interest": market.get("open_interest"),
                    "close_time": market.get("close_time"),
                    "category": market.get("category"),
                    "url": f"https://kalshi.com/markets/{market.get('ticker')}",
                })
        
        return markets
```

**Note for agent**: Kalshi's API may require different endpoints depending on current API version. If the above doesn't work, try:
- `https://trading-api.kalshi.com/trade-api/v2/markets`
- Check `https://trading-api.kalshi.com/trade-api/v2/exchange/status` to verify API is up
- The key data point is the `yes_price` which represents implied probability

### 4.7 Prediction Markets — Polymarket API

Polymarket uses the Gamma API for market data.

**API Base**: `https://gamma-api.polymarket.com`

```python
async def get_polymarket_markets(query: str = "federal reserve") -> list[dict]:
    """
    Search Polymarket for relevant prediction markets.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://gamma-api.polymarket.com/markets",
            params={
                "closed": "false",
                "limit": 20,
                "order": "volume",
                "ascending": "false",
            },
            timeout=15.0,
        )
        data = response.json()
        
        markets = []
        for market in data:
            question = market.get("question", "").lower()
            # Filter for finance-relevant markets
            finance_keywords = ["fed", "rate", "inflation", "recession", "gdp", "s&p", "nasdaq", 
                              "bitcoin", "crypto", "tariff", "treasury", "unemployment", "stock"]
            if any(kw in question for kw in finance_keywords):
                markets.append({
                    "question": market.get("question"),
                    "description": market.get("description", "")[:200],
                    "outcome_prices": market.get("outcomePrices", ""),  # JSON string of prices
                    "volume": market.get("volume"),
                    "liquidity": market.get("liquidity"),
                    "end_date": market.get("endDate"),
                    "image": market.get("image"),
                    "url": f"https://polymarket.com/event/{market.get('slug', '')}",
                })
        
        return markets
```

### 4.8 Browser Agent — Finviz Scraper

Use Playwright to scrape Finviz ticker pages. Finviz provides a dense overview including analyst ratings, target price, and institutional ownership — data that's hard to get from free APIs.

**Install**: `pip install playwright && playwright install chromium`

```python
from playwright.async_api import async_playwright

async def scrape_finviz(symbol: str) -> dict:
    """
    Scrape Finviz quote page for a ticker.
    Returns: analyst rating, target price, institutional data, sector performance, news headlines.
    """
    url = f"https://finviz.com/quote.ashx?t={symbol}&p=d"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Set a realistic user agent
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        
        try:
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")
            
            # Extract the snapshot table (contains all key metrics)
            # Finviz uses a table with class "snapshot-table2"
            snapshot_data = {}
            rows = await page.query_selector_all("table.snapshot-table2 tr")
            for row in rows:
                cells = await row.query_selector_all("td")
                cell_texts = [await c.inner_text() for c in cells]
                # Cells alternate: label, value, label, value, ...
                for i in range(0, len(cell_texts) - 1, 2):
                    label = cell_texts[i].strip()
                    value = cell_texts[i + 1].strip()
                    snapshot_data[label] = value
            
            # Extract analyst recommendations
            # Look for the "Recom" field in snapshot data
            analyst_recom = snapshot_data.get("Recom", "N/A")
            target_price = snapshot_data.get("Target Price", "N/A")
            
            # Extract recent news headlines from the news table
            news_items = []
            news_rows = await page.query_selector_all("table.fullview-news-outer tr")
            for row in news_rows[:10]:  # Top 10 news items
                link = await row.query_selector("a.tab-link-news")
                date_cell = await row.query_selector("td:first-child")
                if link:
                    title = await link.inner_text()
                    href = await link.get_attribute("href")
                    date_text = await date_cell.inner_text() if date_cell else ""
                    news_items.append({
                        "title": title.strip(),
                        "url": href,
                        "date": date_text.strip(),
                    })
            
            await browser.close()
            
            return {
                "source": "finviz",
                "url": url,
                "snapshot": snapshot_data,  # Full key-value table
                "analyst_recommendation": analyst_recom,  # 1.0 (Strong Buy) to 5.0 (Strong Sell)
                "target_price": target_price,
                "news": news_items,
                # Pull out specific useful fields:
                "highlights": {
                    "analyst_recom": analyst_recom,
                    "target_price": target_price,
                    "insider_ownership": snapshot_data.get("Insider Own", "N/A"),
                    "institutional_ownership": snapshot_data.get("Inst Own", "N/A"),
                    "short_float": snapshot_data.get("Short Float", "N/A"),
                    "rsi_14": snapshot_data.get("RSI (14)", "N/A"),
                    "rel_volume": snapshot_data.get("Rel Volume", "N/A"),
                    "perf_week": snapshot_data.get("Perf Week", "N/A"),
                    "perf_month": snapshot_data.get("Perf Month", "N/A"),
                    "perf_quarter": snapshot_data.get("Perf Quarter", "N/A"),
                    "perf_year": snapshot_data.get("Perf YTD", "N/A"),
                    "earnings_date": snapshot_data.get("Earnings", "N/A"),
                    "sma_20_distance": snapshot_data.get("SMA20", "N/A"),
                    "sma_50_distance": snapshot_data.get("SMA50", "N/A"),
                    "sma_200_distance": snapshot_data.get("SMA200", "N/A"),
                }
            }
        except Exception as e:
            await browser.close()
            return {"source": "finviz", "error": str(e), "url": url}
```

**Fallback**: If Playwright is too slow or Finviz blocks the scraper, use the `finvizfinance` Python package as a simpler alternative:

```python
# pip install finvizfinance
from finvizfinance.quote import finvizfinance as fvz

def get_finviz_simple(symbol: str) -> dict:
    stock = fvz(symbol)
    return {
        "fundamentals": stock.ticker_fundament(),  # Dict of all snapshot data
        "description": stock.ticker_description(),
        "news": stock.ticker_news(),  # DataFrame of news
        "analyst_ratings": stock.ticker_outer_ratings(),  # DataFrame of ratings
    }
```

---

## 5. Sentiment Analysis Pipeline

### Composite Sentiment Score

Combine Perplexity Sonar sentiment + Reddit VADER sentiment into a single composite score.

```python
def compute_composite_sentiment(
    perplexity_summary: str,
    reddit_aggregate: dict,
) -> dict:
    """
    Create a composite sentiment score from all sources.
    
    Perplexity: Parse the sentiment score from the AI summary (it returns a -100 to +100 score)
    Reddit: Use the normalized VADER score (-100 to +100)
    
    Weights:
    - Perplexity (AI Research): 60% — higher quality, broader scope
    - Reddit (Social): 40% — real-time retail sentiment
    """
    # Extract Perplexity sentiment score from the summary text
    # The prompt asks Sonar to return a SENTIMENT SCORE number
    # Parse it from the text (or default to 0 if not found)
    perplexity_score = extract_sentiment_score_from_text(perplexity_summary)  # Implement regex/parsing
    
    reddit_score = reddit_aggregate.get("score_normalized", 0)
    
    composite = (perplexity_score * 0.6) + (reddit_score * 0.4)
    
    if composite >= 40:
        label = "Strongly Bullish"
        color = "#22c55e"  # green-500
    elif composite >= 10:
        label = "Bullish"
        color = "#86efac"  # green-300
    elif composite >= -10:
        label = "Neutral"
        color = "#fbbf24"  # amber-400
    elif composite >= -40:
        label = "Bearish"
        color = "#f87171"  # red-400
    else:
        label = "Strongly Bearish"
        color = "#ef4444"  # red-500
    
    return {
        "composite_score": round(composite, 1),
        "label": label,
        "color": color,
        "breakdown": {
            "ai_research": {"score": perplexity_score, "weight": 0.6, "source": "Perplexity Sonar"},
            "reddit": {"score": reddit_score, "weight": 0.4, "source": "Reddit (VADER)"},
        }
    }
```

---

## 6. Browser-Based Research Agents

### Architecture

The browser agent is a Playwright-based async scraper that can be pointed at institutional research sites. For the hackathon, focus on **Finviz** (covered in 4.8 above). If time permits, add a **Morningstar** scraper:

```python
async def scrape_morningstar(symbol: str) -> dict:
    """
    Scrape Morningstar for fair value estimate, moat rating, star rating.
    URL pattern: https://www.morningstar.com/stocks/{exchange}/{symbol}/quote
    """
    # Morningstar requires determining the exchange prefix
    # Common: xnys (NYSE), xnas (NASDAQ)
    exchanges = ["xnas", "xnys"]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        for exchange in exchanges:
            url = f"https://www.morningstar.com/stocks/{exchange}/{symbol.lower()}/quote"
            try:
                await page.goto(url, timeout=15000)
                
                # Check if page loaded correctly
                title = await page.title()
                if "404" in title or "not found" in title.lower():
                    continue
                
                # Extract star rating
                star_rating = await page.eval_on_selector(
                    '[data-testid="star-rating"]', 
                    'el => el.getAttribute("aria-label")',
                    strict=False
                ) or "N/A"
                
                # Extract fair value estimate
                fair_value = await page.eval_on_selector(
                    'text=Fair Value', 
                    'el => el.parentElement?.querySelector("[class*=price]")?.innerText',
                    strict=False
                ) or "N/A"
                
                # Extract moat rating
                moat = await page.eval_on_selector(
                    'text=Economic Moat',
                    'el => el.parentElement?.querySelector("[class*=value]")?.innerText',
                    strict=False
                ) or "N/A"
                
                await browser.close()
                
                return {
                    "source": "morningstar",
                    "url": url,
                    "star_rating": star_rating,
                    "fair_value": fair_value,
                    "moat_rating": moat,
                }
            except Exception:
                continue
        
        await browser.close()
        return {"source": "morningstar", "error": "Could not find ticker on Morningstar"}
```

---

## 7. Frontend Implementation

### 7.1 Project Setup

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install lightweight-charts recharts axios tailwindcss @tailwindcss/vite lucide-react
```

### 7.2 Page Structure

The app has 2 main pages:

**Page 1: Home / Overview Dashboard** (`/`)
- Search bar at top (ticker search with autocomplete)
- Watchlist section: grid of ticker cards showing sparkline + price + % change
- Market overview: S&P 500, NASDAQ, DOW mini charts
- Macro indicators bar: Fed rate, CPI, VIX, 10Y yield (from FRED)
- Trending tickers: based on Reddit mention volume
- Prediction market highlights: top Kalshi/Polymarket events

**Page 2: Ticker Research Page** (`/ticker/:symbol`)
This is the main research page. Layout (top to bottom, left to right):

```
┌─────────────────────────────────────────────────────────────┐
│ HEADER: AAPL - Apple Inc. | $187.44 | +1.23 (+0.66%) | ▲   │
│ Sector: Technology | Industry: Consumer Electronics         │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────┐ ┌─────────────────────┐ │
│ │                                 │ │ KEY STATS           │ │
│ │   PRICE CHART                   │ │ P/E: 28.5           │ │
│ │   (Candlestick + Volume)        │ │ Fwd P/E: 26.1       │ │
│ │   Period toggles: 1D 1W 1M 3M  │ │ Beta: 1.28          │ │
│ │   6M 1Y 2Y 5Y                  │ │ Mkt Cap: $2.89T     │ │
│ │   Overlay: SMA50, SMA200, BB   │ │ EPS: $6.57           │ │
│ │                                 │ │ Div Yield: 0.55%    │ │
│ │                                 │ │ 52wk H: $199.62     │ │
│ │                                 │ │ 52wk L: $164.08     │ │
│ │                                 │ │ Vol: 55.2M          │ │
│ │                                 │ │ Avg Vol: 58.7M      │ │
│ └─────────────────────────────────┘ └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ TABS: [Overview] [Technicals] [Sentiment] [AI Research]     │
│       [Prediction Markets] [Financials] [News]              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ TAB CONTENT AREA (see below for each tab)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Tab Content Details

**Overview Tab:**
- Company description (from yfinance)
- Composite sentiment gauge (big circular gauge: -100 to +100)
- Signals summary: list of bullish/bearish signals from technicals + sentiment
- Finviz highlights: analyst rating, target price, insider/institutional ownership
- Morningstar: star rating, moat, fair value (if available)

**Technicals Tab:**
- RSI chart (separate chart below price, with overbought/oversold lines at 70/30)
- MACD chart (MACD line, signal line, histogram)
- Bollinger Bands overlay on the main price chart
- Table of technical values: RSI, MACD, SMA 20/50/200, BB upper/lower
- Technical signals with buy/sell indicators (green/red badges)

**Sentiment Tab:**
- Composite sentiment score (large number + gauge)
- Breakdown chart: bar chart showing Perplexity vs Reddit scores
- Sentiment over time: if data allows, show sentiment trend (line chart)
- Reddit posts list: scrollable card list with:
  - Post title (clickable link)
  - Subreddit badge
  - Upvotes
  - Sentiment color (green/red/gray bar)
  - Sentiment score
  - Time posted

**AI Research Tab:**
- Full Perplexity Sonar summary rendered as formatted markdown
- Citations displayed as numbered clickable links below the summary
- "Last updated" timestamp
- "Refresh" button to re-query Perplexity

**Prediction Markets Tab:**
- Split into two sections: Kalshi and Polymarket
- Each market displayed as a card:
  - Event title/question
  - Implied probability as a progress bar (e.g., 73% YES)
  - Volume traded
  - End date
  - Link to the market
- Filter/search within prediction markets

**Financials Tab:**
- Valuation ratios table: P/E, Forward P/E, PEG, P/B, P/S, EV/EBITDA
- Profitability: Profit margin, ROE, ROA
- Growth: Revenue growth, Earnings growth
- Balance sheet highlights: Debt/Equity, Current Ratio, Free Cash Flow
- Use color coding: green for metrics above industry average, red for below (if available)

**News Tab:**
- Combined news from Finviz scraper headlines + any Perplexity citations
- Each news item: headline (linked), source, date, sentiment badge
- Sorted by date (newest first)

### 7.4 Chart Implementation with Lightweight Charts

```typescript
// components/PriceChart.tsx
import { createChart, ColorType, IChartApi } from 'lightweight-charts';
import { useEffect, useRef, useState } from 'react';

interface PriceChartProps {
  data: {
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[];
  sma50?: { time: string; value: number }[];
  sma200?: { time: string; value: number }[];
}

const PERIOD_OPTIONS = [
  { label: "1D", period: "1d", interval: "5m" },
  { label: "1W", period: "5d", interval: "15m" },
  { label: "1M", period: "1mo", interval: "1h" },
  { label: "3M", period: "3mo", interval: "1d" },
  { label: "6M", period: "6mo", interval: "1d" },
  { label: "1Y", period: "1y", interval: "1d" },
  { label: "2Y", period: "2y", interval: "1wk" },
  { label: "5Y", period: "5y", interval: "1wk" },
];

export function PriceChart({ data, sma50, sma200 }: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0f172a' }, // slate-900
        textColor: '#94a3b8', // slate-400
      },
      grid: {
        vertLines: { color: '#1e293b' },  // slate-800
        horzLines: { color: '#1e293b' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      crosshair: {
        mode: 0, // Normal
      },
      timeScale: {
        borderColor: '#334155', // slate-700
      },
    });

    // Candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    });
    candlestickSeries.setData(data);

    // Volume histogram
    const volumeSeries = chart.addHistogramSeries({
      color: '#3b82f6',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });
    volumeSeries.setData(
      data.map(d => ({
        time: d.time,
        value: d.volume,
        color: d.close >= d.open ? '#22c55e40' : '#ef444440',
      }))
    );

    // SMA overlays
    if (sma50?.length) {
      const sma50Series = chart.addLineSeries({
        color: '#f59e0b', // amber
        lineWidth: 1,
        title: 'SMA 50',
      });
      sma50Series.setData(sma50);
    }

    if (sma200?.length) {
      const sma200Series = chart.addLineSeries({
        color: '#8b5cf6', // violet
        lineWidth: 1,
        title: 'SMA 200',
      });
      sma200Series.setData(sma200);
    }

    chart.timeScale().fitContent();
    chartRef.current = chart;

    // Resize handler
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data, sma50, sma200]);

  return <div ref={chartContainerRef} />;
}
```

### 7.5 Sentiment Gauge Component

```typescript
// components/SentimentGauge.tsx
// Use Recharts PieChart to create a half-circle gauge
import { PieChart, Pie, Cell } from 'recharts';

interface SentimentGaugeProps {
  score: number; // -100 to +100
  label: string;
  color: string;
}

export function SentimentGauge({ score, label, color }: SentimentGaugeProps) {
  // Convert -100..+100 to 0..200 for the gauge
  const normalized = score + 100; // 0 to 200
  const data = [
    { value: normalized },
    { value: 200 - normalized },
  ];
  
  return (
    <div className="flex flex-col items-center">
      <PieChart width={200} height={120}>
        <Pie
          data={data}
          cx={100}
          cy={100}
          startAngle={180}
          endAngle={0}
          innerRadius={60}
          outerRadius={80}
          dataKey="value"
        >
          <Cell fill={color} />
          <Cell fill="#1e293b" />
        </Pie>
      </PieChart>
      <div className="text-3xl font-bold" style={{ color }}>{score > 0 ? '+' : ''}{score}</div>
      <div className="text-sm text-slate-400">{label}</div>
    </div>
  );
}
```

### 7.6 Color Scheme & Design Language

Use a **dark theme** reminiscent of Bloomberg Terminal / TradingView:

```
Background:     #0f172a (slate-900)
Surface:        #1e293b (slate-800)
Border:         #334155 (slate-700)
Text Primary:   #f1f5f9 (slate-100)
Text Secondary: #94a3b8 (slate-400)
Green (Bullish):#22c55e (green-500)
Red (Bearish):  #ef4444 (red-500)
Amber (Neutral):#f59e0b (amber-500)
Blue (Accent):  #3b82f6 (blue-500)
Violet (SMA):   #8b5cf6 (violet-500)
```

---

## 8. Data Models & Schemas

### Pydantic Models (Backend)

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TickerOverview(BaseModel):
    symbol: str
    name: str
    sector: Optional[str]
    industry: Optional[str]
    exchange: Optional[str]
    currency: str = "USD"
    description: Optional[str]
    website: Optional[str]
    employees: Optional[int]

class PriceData(BaseModel):
    current_price: float
    previous_close: float
    open: float
    day_high: float
    day_low: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    volume: int
    avg_volume: int
    market_cap: Optional[float]
    change: float  # current - previous_close
    change_percent: float  # (change / previous_close) * 100

class ValuationMetrics(BaseModel):
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    peg_ratio: Optional[float]
    price_to_book: Optional[float]
    price_to_sales: Optional[float]
    ev_to_ebitda: Optional[float]

class Fundamentals(BaseModel):
    eps: Optional[float]
    forward_eps: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    profit_margin: Optional[float]
    return_on_equity: Optional[float]
    debt_to_equity: Optional[float]
    revenue: Optional[float]
    revenue_growth: Optional[float]
    earnings_growth: Optional[float]
    free_cash_flow: Optional[float]

class TechnicalIndicators(BaseModel):
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    sma_20: float
    sma_50: float
    sma_200: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    volatility_30d: float
    signals: list[str]

class SentimentResult(BaseModel):
    composite_score: float  # -100 to +100
    label: str  # Strongly Bullish, Bullish, Neutral, Bearish, Strongly Bearish
    color: str  # Hex color for UI
    breakdown: dict  # Per-source scores

class RedditPost(BaseModel):
    title: str
    selftext: str
    subreddit: str
    score: int
    num_comments: int
    created_utc: str
    url: str
    sentiment_score: float
    sentiment_label: str

class PredictionMarket(BaseModel):
    title: str
    source: str  # "kalshi" or "polymarket"
    implied_probability: float  # 0-100
    volume: Optional[float]
    end_date: Optional[str]
    url: str

class AIResearchSummary(BaseModel):
    summary: str  # Markdown-formatted research summary
    citations: list[str]  # List of source URLs
    timestamp: str

class FullTickerResearch(BaseModel):
    """The complete response for a ticker research page."""
    overview: TickerOverview
    price: PriceData
    valuation: ValuationMetrics
    fundamentals: Fundamentals
    technicals: TechnicalIndicators
    sentiment: SentimentResult
    ai_research: AIResearchSummary
    reddit_posts: list[RedditPost]
    prediction_markets: list[PredictionMarket]
    finviz_data: Optional[dict]
    morningstar_data: Optional[dict]
    macro_indicators: dict
    chart_data: list[dict]  # OHLCV for lightweight-charts
    last_updated: str
```

---

## 9. API Endpoints

### Backend REST API

```
GET  /api/ticker/{symbol}
     → Returns: FullTickerResearch (everything for the research page)
     → This is the main endpoint. It orchestrates all service calls concurrently.

GET  /api/ticker/{symbol}/chart?period=1y&interval=1d
     → Returns: { data: [...OHLCV], technicals: {...}, sma50_series: [...], sma200_series: [...] }
     → Called when user changes chart period

GET  /api/ticker/{symbol}/sentiment
     → Returns: { composite: SentimentResult, reddit: {...}, ai_summary: AIResearchSummary }
     → Can be refreshed independently

GET  /api/ticker/{symbol}/ai-research
     → Returns: AIResearchSummary
     → Called by "Refresh" button on AI Research tab

GET  /api/prediction-markets?query=fed
     → Returns: { kalshi: [...], polymarket: [...] }
     → For the prediction markets tab/section

GET  /api/macro
     → Returns: { indicators: {...} }
     → FRED macro data for the header bar

GET  /api/search?q=appl
     → Returns: { results: [{ symbol: "AAPL", name: "Apple Inc." }, ...] }
     → For ticker search autocomplete (use yfinance or a static list)
```

### Main Ticker Endpoint — Concurrent Fetching

The `/api/ticker/{symbol}` endpoint should fetch ALL data sources concurrently using `asyncio.gather`:

```python
from fastapi import FastAPI, HTTPException
import asyncio

@app.get("/api/ticker/{symbol}")
async def get_ticker_research(symbol: str):
    symbol = symbol.upper().strip()
    
    # Check cache first
    if symbol in market_data_cache:
        return market_data_cache[symbol]
    
    try:
        # Fetch everything concurrently
        (
            ticker_data,
            chart_data,
            reddit_data,
            ai_research,
            kalshi_markets,
            poly_markets,
            finviz_data,
            macro_data,
        ) = await asyncio.gather(
            asyncio.to_thread(get_ticker_data, symbol),          # yfinance (sync → thread)
            asyncio.to_thread(get_price_history, symbol),        # yfinance (sync → thread)
            asyncio.to_thread(get_reddit_sentiment, symbol),     # PRAW (sync → thread)
            get_ai_research_summary(symbol, ""),                  # Perplexity (async)
            get_kalshi_markets(),                                  # Kalshi (async)
            get_polymarket_markets(),                              # Polymarket (async)
            scrape_finviz(symbol),                                 # Playwright (async)
            asyncio.to_thread(get_macro_indicators),              # FRED (sync → thread)
            return_exceptions=True,  # Don't fail if one source errors
        )
        
        # Score Reddit sentiment
        if isinstance(reddit_data, dict) and "posts" in reddit_data:
            scored_reddit = score_reddit_posts(reddit_data["posts"])
        else:
            scored_reddit = {"posts": [], "aggregate": {"score_normalized": 0}}
        
        # Compute technicals
        hist_df = yf.Ticker(symbol).history(period="1y")
        technicals = compute_technicals(hist_df)
        
        # Compute composite sentiment
        sentiment = compute_composite_sentiment(
            ai_research.get("summary", "") if isinstance(ai_research, dict) else "",
            scored_reddit.get("aggregate", {}),
        )
        
        # Assemble response
        result = {
            "overview": {
                "symbol": symbol,
                "name": ticker_data.get("name", symbol) if isinstance(ticker_data, dict) else symbol,
                # ... map all fields
            },
            "price": ticker_data if isinstance(ticker_data, dict) else {},
            "chart_data": chart_data if isinstance(chart_data, list) else [],
            "technicals": technicals,
            "sentiment": sentiment,
            "ai_research": ai_research if isinstance(ai_research, dict) else {},
            "reddit_posts": scored_reddit.get("posts", []),
            "prediction_markets": {
                "kalshi": kalshi_markets if isinstance(kalshi_markets, list) else [],
                "polymarket": poly_markets if isinstance(poly_markets, list) else [],
            },
            "finviz": finviz_data if isinstance(finviz_data, dict) else {},
            "macro": macro_data if isinstance(macro_data, dict) else {},
            "last_updated": datetime.utcnow().isoformat(),
        }
        
        # Cache result
        market_data_cache[symbol] = result
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 10. Environment & Configuration

### Required API Keys / Credentials

Create a `.env` file:

```env
# Perplexity Sonar API
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxx

# Reddit API (create app at https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret

# FRED API (register at https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY=your_fred_key

# Kalshi and Polymarket — no auth needed for public read endpoints

# yfinance — no API key needed
```

### Python Dependencies

```
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
yfinance==0.2.40
praw==7.7.1
nltk==3.9
httpx==0.27.0
cachetools==5.4.0
fredapi==0.5.2
playwright==1.47.0
python-dotenv==1.0.1
pandas==2.2.0
pydantic==2.9.0
```

### Frontend Dependencies

```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.26.0",
    "lightweight-charts": "^4.2.0",
    "recharts": "^2.12.0",
    "axios": "^1.7.0",
    "lucide-react": "^0.441.0",
    "react-markdown": "^9.0.0"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.0.0",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.5.0",
    "@types/react": "^18.3.0",
    "vite": "^5.4.0"
  }
}
```

---

## 11. File Structure

```
project/
├── backend/
│   ├── main.py                    # FastAPI app, routes, CORS
│   ├── config.py                  # Environment variables, constants
│   ├── services/
│   │   ├── market_data.py         # yfinance: ticker data, price history, technicals
│   │   ├── sentiment.py           # VADER scoring, composite sentiment
│   │   ├── perplexity.py          # Perplexity Sonar API calls
│   │   ├── reddit.py              # PRAW: Reddit data fetching
│   │   ├── prediction_markets.py  # Kalshi + Polymarket API calls
│   │   ├── browser_agent.py       # Playwright: Finviz + Morningstar scrapers
│   │   ├── macro.py               # FRED API macro indicators
│   │   └── search.py              # Ticker search/autocomplete
│   ├── models/
│   │   └── schemas.py             # Pydantic models
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Router setup
│   │   ├── main.tsx               # Entry point
│   │   ├── index.css              # Tailwind imports + global styles
│   │   ├── api/
│   │   │   └── client.ts          # Axios instance + API call functions
│   │   ├── pages/
│   │   │   ├── HomePage.tsx       # Watchlist, market overview, search
│   │   │   └── TickerPage.tsx     # Full research page for a ticker
│   │   ├── components/
│   │   │   ├── charts/
│   │   │   │   ├── PriceChart.tsx       # Candlestick + volume (lightweight-charts)
│   │   │   │   ├── RSIChart.tsx         # RSI indicator chart
│   │   │   │   ├── MACDChart.tsx        # MACD indicator chart
│   │   │   │   └── SentimentGauge.tsx   # Half-circle gauge (recharts)
│   │   │   ├── ticker/
│   │   │   │   ├── TickerHeader.tsx     # Name, price, change, sector
│   │   │   │   ├── KeyStatsCard.tsx     # P/E, Beta, Market Cap, etc.
│   │   │   │   ├── TechnicalSignals.tsx # Buy/sell signal badges
│   │   │   │   └── PeriodSelector.tsx   # 1D, 1W, 1M, ... buttons
│   │   │   ├── sentiment/
│   │   │   │   ├── SentimentOverview.tsx # Composite gauge + breakdown
│   │   │   │   ├── RedditPostCard.tsx    # Individual Reddit post
│   │   │   │   └── RedditFeed.tsx        # Scrollable Reddit posts list
│   │   │   ├── research/
│   │   │   │   ├── AIResearchPanel.tsx   # Perplexity summary + citations
│   │   │   │   └── FinvizData.tsx        # Finviz scraped data display
│   │   │   ├── prediction/
│   │   │   │   ├── PredictionCard.tsx    # Single market card
│   │   │   │   └── PredictionMarkets.tsx # Kalshi + Polymarket sections
│   │   │   ├── macro/
│   │   │   │   └── MacroBar.tsx          # Horizontal macro indicator bar
│   │   │   ├── search/
│   │   │   │   └── TickerSearch.tsx      # Search bar with autocomplete
│   │   │   └── common/
│   │   │       ├── TabNav.tsx            # Tab navigation component
│   │   │       ├── LoadingSpinner.tsx
│   │   │       └── ErrorState.tsx
│   │   └── utils/
│   │       ├── formatters.ts     # Number formatting, currency, percentages
│   │       └── constants.ts      # Colors, period options, etc.
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── index.html
│
└── README.md
```

---

## 12. Implementation Order

Build in this order to have a working demo as early as possible:

### Phase 1: Core Data (Get data flowing)
1. Set up FastAPI backend with CORS
2. Implement `market_data.py` (yfinance ticker data + price history + technicals)
3. Create `/api/ticker/{symbol}` endpoint returning just market data
4. Set up React frontend with Vite + Tailwind
5. Build `TickerPage.tsx` with `PriceChart` (lightweight-charts) + `KeyStatsCard`
6. Verify: can enter a ticker and see a professional chart + stats

### Phase 2: Sentiment (The "wow" factor)
7. Implement `perplexity.py` (Sonar API call)
8. Implement `reddit.py` (PRAW) + `sentiment.py` (VADER scoring)
9. Compute composite sentiment
10. Build `SentimentGauge`, `AIResearchPanel`, `RedditFeed` components
11. Add Sentiment and AI Research tabs

### Phase 3: Prediction Markets + Browser Agent
12. Implement `prediction_markets.py` (Kalshi + Polymarket)
13. Implement `browser_agent.py` (Finviz scraper)
14. Build `PredictionMarkets` and `FinvizData` components
15. Add Prediction Markets tab

### Phase 4: Polish
16. Implement `macro.py` (FRED) + `MacroBar` component
17. Build `HomePage` with search, watchlist, market overview
18. Add all remaining tabs (Technicals, Financials, News)
19. Error handling, loading states, empty states
20. Make the concurrent fetching work with `asyncio.gather`
21. Final UI polish: responsive layout, transitions, hover states

### Phase 5: Demo Prep
22. Pre-load a few popular tickers (AAPL, TSLA, NVDA, MSFT, META) in cache
23. Test all data sources are returning data
24. Prepare a demo flow: search NVDA → show chart → sentiment → AI research → prediction markets
25. Have fallback/mock data ready in case any API is down during demo

---

## Links to All Tools & APIs

| Tool | URL | Notes |
|------|-----|-------|
| Perplexity Sonar API | https://docs.perplexity.ai/ | AI research summaries |
| Reddit API / PRAW | https://www.reddit.com/prefs/apps | Social sentiment |
| Kalshi API | https://trading-api.kalshi.com/trade-api/v2 | Prediction markets |
| Polymarket Gamma API | https://gamma-api.polymarket.com | Prediction markets |
| yfinance | https://pypi.org/project/yfinance/ | Market data (free) |
| FRED API | https://fred.stlouisfed.org/docs/api/fred/ | Macro indicators |
| Playwright | https://playwright.dev/python/ | Browser automation |
| Finviz | https://finviz.com | Institutional data (scrape target) |
| Morningstar | https://www.morningstar.com | Fair value / moat (scrape target) |
| Lightweight Charts | https://tradingview.github.io/lightweight-charts/ | TradingView charting |
| Recharts | https://recharts.org/ | React charts |
| Tailwind CSS | https://tailwindcss.com/ | Styling |
| FastAPI | https://fastapi.tiangolo.com/ | Backend framework |
| shadcn/ui | https://ui.shadcn.com/ | UI components |

---

## Critical Notes for the Coding Agent

1. **Always use `asyncio.gather` with `return_exceptions=True`** for the main ticker endpoint. If Reddit is down, the page should still load with market data + Perplexity.

2. **Every external API call must have a timeout** (10-15 seconds max). Never let a single slow API block the whole response.

3. **Cache everything.** The user will click between tabs — don't re-fetch data they already have.

4. **The frontend should show partial data as it loads.** Don't wait for all 8 data sources to resolve before showing anything. Show the chart + stats first (fastest), then sentiment, then AI research (slowest).

5. **Dark theme is non-negotiable.** This is a financial tool. Light themes look amateur for trading dashboards.

6. **The chart must be interactive.** Crosshair, tooltip with OHLCV on hover, period switching, overlay toggles. `lightweight-charts` does all of this out of the box.

7. **Perplexity Sonar is the star of the show.** Make the AI Research tab prominent and the summary beautifully formatted. Render it as markdown with `react-markdown`.

8. **Handle errors gracefully.** If a data source fails, show a subtle error state for that section, not a full page crash. Each component should handle its own error state.

9. **Numbers must be formatted properly.** Market cap as "$2.89T" not "2890000000000". Percentages with color (green for positive, red for negative). Use Intl.NumberFormat.

10. **The search must feel snappy.** Use a debounced input (300ms) and show results in a dropdown. Pre-load popular tickers.
