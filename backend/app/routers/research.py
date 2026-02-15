from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas import ResearchChatRequest, ResearchChatResponse, ResearchRequest
from app.services.browserbase_scraper import run_deep_research
from app.services.llm import generate_openai_commentary
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


def _normalize_timeframe(value: str | None) -> str:
    if not value:
        return "7d"
    raw = value.strip().lower()
    aliases = {
        "1d": "24h",
        "24hr": "24h",
        "24hrs": "24h",
        "day": "24h",
        "week": "7d",
        "1w": "7d",
        "month": "30d",
        "1m": "30d",
        "3m": "90d",
        "6m": "180d",
        "year": "1y",
        "12m": "1y",
    }
    canonical = aliases.get(raw, raw)
    allowed = {"24h", "7d", "30d", "60d", "90d", "180d", "1y", "2y", "5y", "10y", "max"}
    return canonical if canonical in allowed else "7d"


def _infer_ticker_from_prompt(prompt: str) -> str | None:
    tokens = re.findall(r"\$?[A-Z]{1,5}", prompt.upper())
    if not tokens:
        return None
    ignored = {"A", "I", "THE", "AND", "OR", "TO", "FOR", "OF", "IN", "ON", "AT", "IS", "ARE"}
    for token in tokens:
        clean = token.lstrip("$").strip()
        if not clean or clean in ignored:
            continue
        return clean
    return None


def _fallback_chat_response(
    ticker: str,
    research_payload,
    deep_payload: dict | None,
) -> tuple[str, list[str]]:
    recommendation = str(research_payload.recommendation).replace("_", " ")
    sentiment = float(research_payload.aggregate_sentiment)
    lines: list[str] = [
        f"{ticker} composite signal: {recommendation} ({sentiment:+.2f}).",
    ]
    for narrative in list(research_payload.narratives or [])[:4]:
        text = str(narrative).strip()
        if text:
            lines.append(f"- {text}")
    if deep_payload:
        for bullet in list(deep_payload.get("deep_bullets") or [])[:3]:
            text = str(bullet).strip()
            if text:
                lines.append(f"- {text}")

    source_labels = [str(item.source) for item in list(research_payload.source_breakdown or []) if getattr(item, "source", None)]
    if deep_payload:
        source_labels.extend([str(item) for item in list(deep_payload.get("sources") or []) if item])
    deduped_sources = list(dict.fromkeys(source_labels))
    return "\n".join(lines), deduped_sources


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


@router.post("/chat", response_model=ResearchChatResponse)
async def research_chat(payload: ResearchChatRequest, request: Request):
    settings = request.app.state.settings
    timeframe = _normalize_timeframe(payload.timeframe)

    raw_ticker = (payload.ticker or "").strip()
    resolved_ticker: str | None = None
    if raw_ticker:
        try:
            resolved_ticker = resolve_symbol_input(raw_ticker)
        except ValueError:
            if not payload.auto_fetch_if_missing:
                raise HTTPException(status_code=422, detail="Ticker could not be resolved.")

    if not resolved_ticker and payload.auto_fetch_if_missing:
        inferred = _infer_ticker_from_prompt(payload.prompt)
        if inferred:
            try:
                resolved_ticker = resolve_symbol_input(inferred)
            except ValueError:
                resolved_ticker = inferred

    ticker = (resolved_ticker or "AAPL").upper().strip()

    research_payload = await run_research(
        ResearchRequest(ticker=ticker, timeframe=timeframe, include_prediction_markets=True),
        settings,
    )
    deep_payload = await run_deep_research(ticker, settings) if payload.include_deep else None

    context = {
        "ticker": ticker,
        "timeframe": timeframe,
        "prompt": payload.prompt,
        "aggregate_sentiment": research_payload.aggregate_sentiment,
        "recommendation": research_payload.recommendation,
        "narratives": list(research_payload.narratives or [])[:6],
        "source_summaries": [
            {
                "source": row.source,
                "score": row.score,
                "summary": row.summary,
            }
            for row in list(research_payload.source_breakdown or [])[:5]
        ],
        "prediction_markets": list(research_payload.prediction_markets or [])[:3],
        "deep_research": deep_payload if payload.include_deep else None,
    }

    if settings.openai_api_key:
        out = await generate_openai_commentary(payload.prompt, context, settings)
        source_labels = [str(item.source) for item in list(research_payload.source_breakdown or []) if getattr(item, "source", None)]
        if deep_payload:
            source_labels.extend([str(item) for item in list(deep_payload.get("sources") or []) if item])
        return ResearchChatResponse(
            ticker=ticker,
            response=str(out.get("response") or "").strip() or f"No response generated for {ticker}.",
            model=str(out.get("model") or "gpt-4o-mini"),
            generated_at=str(out.get("generated_at") or datetime.now(timezone.utc).isoformat()),
            context_refreshed=True,
            sources=list(dict.fromkeys(source_labels)),
        )

    fallback_text, source_labels = _fallback_chat_response(ticker, research_payload, deep_payload)
    return ResearchChatResponse(
        ticker=ticker,
        response=fallback_text,
        model="research-context-template",
        generated_at=datetime.now(timezone.utc).isoformat(),
        context_refreshed=True,
        sources=source_labels,
    )


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
