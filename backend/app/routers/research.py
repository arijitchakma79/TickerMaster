from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas import ResearchRequest
from app.services.browserbase_scraper import run_deep_research
from app.services.market_data import (
    fetch_advanced_stock_data,
    fetch_candles,
    fetch_metric,
    resolve_symbol_input,
    search_tickers,
)
from app.services.research_cache import get_cached_research, set_cached_research
from app.services.sentiment import get_x_sentiment, run_research

router = APIRouter(prefix="/research", tags=["research"])

_GENERAL_MOVER_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "AMD", "NFLX", "ORCL", "CRM", "ADBE", "INTC", "QCOM", "PLTR",
    "JPM", "BAC", "WFC", "GS", "V", "MA",
    "XOM", "CVX", "COP", "SLB",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
    "WMT", "COST", "HD", "LOW", "MCD", "NKE", "DIS", "UBER", "SHOP", "COIN",
]


@router.post("/analyze")
async def analyze_research(payload: ResearchRequest, request: Request):
    settings = request.app.state.settings
    return await run_research(payload, settings)


@router.get("/candles/{ticker}")
async def candles(ticker: str, period: str = "1mo", interval: str = "1d"):
    try:
        symbol = resolve_symbol_input(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    cache_key_hot = f"candles:{period}:{interval}:15m"
    cache_key_last = f"candles:{period}:{interval}:last"
    hot = get_cached_research(symbol, cache_key_hot)
    if hot:
        return hot
    try:
        points = fetch_candles(symbol, period=period, interval=interval)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        cached_last = get_cached_research(symbol, cache_key_last)
        if cached_last:
            return {**cached_last, "stale": True, "provider_error": str(exc)}
        raise HTTPException(status_code=502, detail=f"Market data provider unavailable for {ticker.upper()}: {exc}")
    payload = {"ticker": symbol, "points": points}
    set_cached_research(symbol, cache_key_hot, payload, ttl_minutes=15)
    set_cached_research(symbol, cache_key_last, payload, ttl_minutes=24 * 60)
    return payload


@router.get("/advanced/{ticker}")
async def advanced(ticker: str):
    try:
        return fetch_advanced_stock_data(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/x-sentiment/{ticker}")
async def x_sentiment(ticker: str, request: Request):
    settings = request.app.state.settings
    return await get_x_sentiment(ticker, settings)


@router.post("/deep/{ticker}")
async def deep_research(ticker: str, request: Request):
    settings = request.app.state.settings
    return await run_deep_research(ticker, settings)


@router.get("/quote/{ticker}")
async def quote(ticker: str):
    try:
        return fetch_metric(ticker).model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/search/tickers")
async def ticker_lookup(
    query: str = Query(..., min_length=1, max_length=80),
    limit: int = Query(8, ge=1, le=20),
):
    results = await search_tickers(query, limit=limit)
    return {"query": query, "results": [item.model_dump() for item in results]}


@router.get("/movers")
async def market_movers(limit: int = Query(5, ge=3, le=15)):
    cache_key = f"movers:{limit}:5m"
    cached = get_cached_research("GLOBAL", cache_key)
    if cached:
        return cached

    async def load_metric(symbol: str):
        try:
            metric = await asyncio.to_thread(fetch_metric, symbol)
            return metric.model_dump()
        except Exception:
            return None

    metrics = [
        item for item in await asyncio.gather(*[load_metric(symbol) for symbol in _GENERAL_MOVER_UNIVERSE]) if item
    ]
    winners = sorted(
        [item for item in metrics if float(item.get("change_percent", 0.0)) > 0],
        key=lambda item: float(item.get("change_percent", 0.0)),
        reverse=True,
    )[:limit]
    losers = sorted(
        [item for item in metrics if float(item.get("change_percent", 0.0)) < 0],
        key=lambda item: float(item.get("change_percent", 0.0)),
    )[:limit]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(metrics),
        "winners": winners,
        "losers": losers,
        "tickers": metrics,
    }
    set_cached_research("GLOBAL", cache_key, payload, ttl_minutes=5)
    return payload
