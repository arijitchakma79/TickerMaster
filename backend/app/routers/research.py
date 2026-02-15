from __future__ import annotations

import asyncio
import math
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

_MOVER_SNAPSHOT_KEY = "movers:snapshot:1h:v1"
_MOVER_SNAPSHOT_LAST_KEY = "movers:snapshot:last:v1"


def _coerce_metric_payload(raw: dict | None) -> dict | None:
    if not isinstance(raw, dict):
        return None
    ticker = str(raw.get("ticker") or "").upper().strip()
    try:
        price = float(raw.get("price"))
        change_percent = float(raw.get("change_percent"))
    except (TypeError, ValueError):
        return None
    if not ticker or not math.isfinite(price) or not math.isfinite(change_percent):
        return None
    out: dict[str, float | str | None] = {
        "ticker": ticker,
        "price": round(price, 2),
        "change_percent": round(change_percent, 2),
        "pe_ratio": _round_or_none(raw.get("pe_ratio")) if isinstance(raw.get("pe_ratio"), (int, float)) else None,
        "beta": _round_or_none(raw.get("beta")) if isinstance(raw.get("beta"), (int, float)) else None,
        "volume": raw.get("volume"),
        "market_cap": raw.get("market_cap"),
    }
    return out


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), 4)


def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None
    window = values[-period:]
    return sum(window) / period


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period or period <= 0:
        return None
    alpha = 2.0 / (period + 1.0)
    ema_value = sum(values[:period]) / period
    for price in values[period:]:
        ema_value = (price * alpha) + (ema_value * (1 - alpha))
    return ema_value


def _stddev(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) <= period:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for idx in range(1, len(values)):
        delta = values[idx] - values[idx - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for idx in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[idx]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[idx]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _vwap(closes: list[float], highs: list[float], lows: list[float], volumes: list[float]) -> float | None:
    if not closes or not highs or not lows or not volumes:
        return None
    n = min(len(closes), len(highs), len(lows), len(volumes))
    total_pv = 0.0
    total_vol = 0.0
    for idx in range(n):
        typical_price = (highs[idx] + lows[idx] + closes[idx]) / 3.0
        volume = max(volumes[idx], 0.0)
        total_pv += typical_price * volume
        total_vol += volume
    if total_vol <= 0:
        return None
    return total_pv / total_vol


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    if len(highs) <= period or len(lows) <= period or len(closes) <= period:
        return None
    true_ranges: list[float] = []
    for idx in range(1, len(closes)):
        high = highs[idx]
        low = lows[idx]
        prev_close = closes[idx - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return None
    atr_value = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        atr_value = ((atr_value * (period - 1)) + tr) / period
    return atr_value


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
        points = await asyncio.to_thread(fetch_candles, symbol, period, interval)
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
        symbol = resolve_symbol_input(ticker)
        return fetch_advanced_stock_data(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/indicators/{ticker}")
async def indicators(ticker: str, period: str = "6mo", interval: str = "1d"):
    try:
        symbol = resolve_symbol_input(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        points = fetch_candles(symbol, period=period, interval=interval)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Market data provider unavailable for {symbol}: {exc}")

    closes = [float(point.close) for point in points if point.close is not None]
    highs = [float(point.high) for point in points if point.high is not None]
    lows = [float(point.low) for point in points if point.low is not None]
    volumes = [float(point.volume or 0.0) for point in points]

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = (ema12 - ema26) if ema12 is not None and ema26 is not None else None
    macd_series: list[float] = []
    for idx in range(26, len(closes) + 1):
        short_ema = _ema(closes[:idx], 12)
        long_ema = _ema(closes[:idx], 26)
        if short_ema is None or long_ema is None:
            continue
        macd_series.append(short_ema - long_ema)
    macd_signal = _ema(macd_series, 9)
    macd_hist = (macd_line - macd_signal) if macd_line is not None and macd_signal is not None else None

    bb_mid = _sma(closes, 20)
    bb_std = _stddev(closes[-20:]) if len(closes) >= 20 else None
    bb_upper = (bb_mid + (2 * bb_std)) if bb_mid is not None and bb_std is not None else None
    bb_lower = (bb_mid - (2 * bb_std)) if bb_mid is not None and bb_std is not None else None

    latest = {
        "sma20": _round_or_none(_sma(closes, 20)),
        "sma50": _round_or_none(_sma(closes, 50)),
        "sma200": _round_or_none(_sma(closes, 200)),
        "ema21": _round_or_none(_ema(closes, 21)),
        "ema50": _round_or_none(_ema(closes, 50)),
        "vwap": _round_or_none(_vwap(closes, highs, lows, volumes)),
        "rsi14": _round_or_none(_rsi(closes, 14)),
        "macd_line": _round_or_none(macd_line),
        "macd_signal": _round_or_none(macd_signal),
        "macd_hist": _round_or_none(macd_hist),
        "bb_upper": _round_or_none(bb_upper),
        "bb_mid": _round_or_none(bb_mid),
        "bb_lower": _round_or_none(bb_lower),
        "atr14": _round_or_none(_atr(highs, lows, closes, 14)),
    }

    return {
        "ticker": symbol,
        "period": period,
        "interval": interval,
        "latest": latest,
        "available": [key for key, value in latest.items() if value is not None],
    }


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

    # Try to use cached research first for faster responses
    cache_key = f"research:v8:{timeframe}:1"
    cached = get_cached_research(ticker, cache_key)
    research_payload = None
    context_refreshed = False

    if cached:
        # Use cached research data
        from app.schemas import ResearchResponse
        research_payload = ResearchResponse(**cached)
    else:
        # No cache - run research (but skip prediction markets for chat to be faster)
        research_payload = await run_research(
            ResearchRequest(ticker=ticker, timeframe=timeframe, include_prediction_markets=False),
            settings,
        )
        context_refreshed = True

    # Only run deep research if explicitly requested
    deep_payload = None
    if payload.include_deep:
        deep_payload = await run_deep_research(ticker, settings)

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
            context_refreshed=context_refreshed,
            sources=list(dict.fromkeys(source_labels)),
        )

    fallback_text, source_labels = _fallback_chat_response(ticker, research_payload, deep_payload)
    return ResearchChatResponse(
        ticker=ticker,
        response=fallback_text,
        model="research-context-template",
        generated_at=datetime.now(timezone.utc).isoformat(),
        context_refreshed=context_refreshed,
        sources=source_labels,
    )


@router.get("/quote/{ticker}")
async def quote(ticker: str):
    try:
        symbol = resolve_symbol_input(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    cache_key_last = "quote:last"
    try:
        metric = await asyncio.to_thread(fetch_metric, symbol)
    except Exception as exc:
        cached_last = get_cached_research(symbol, cache_key_last)
        if cached_last:
            return {**cached_last, "stale": True, "provider_error": str(exc)}
        raise HTTPException(status_code=502, detail=f"Market data provider unavailable for {symbol}: {exc}")
    payload = metric.model_dump()
    set_cached_research(symbol, cache_key_last, payload, ttl_minutes=24 * 60)
    return payload


@router.get("/search/tickers")
async def ticker_lookup(
    query: str = Query(..., min_length=1, max_length=80),
    limit: int = Query(8, ge=1, le=20),
):
    results = await search_tickers(query, limit=limit)
    return {"query": query, "results": [item.model_dump() for item in results]}


@router.get("/movers")
async def market_movers(limit: int = Query(5, ge=3, le=15)):
    cached_snapshot = get_cached_research("GLOBAL", _MOVER_SNAPSHOT_KEY)
    if cached_snapshot and isinstance(cached_snapshot.get("tickers"), list):
        metrics = [
            item
            for item in (_coerce_metric_payload(raw if isinstance(raw, dict) else None) for raw in cached_snapshot.get("tickers", []))
            if item
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
        return {
            **cached_snapshot,
            "universe_size": len(metrics),
            "winners": winners,
            "losers": losers,
            "tickers": metrics,
        }

    async def load_metric(symbol: str):
        try:
            metric = await asyncio.to_thread(fetch_metric, symbol)
            return metric.model_dump()
        except Exception:
            cached_last = get_cached_research(symbol, "quote:last")
            if isinstance(cached_last, dict):
                payload = _coerce_metric_payload(cached_last)
                if payload:
                    payload["stale"] = True
                    return payload
            return None

    semaphore = asyncio.Semaphore(10)

    async def load_metric_limited(symbol: str):
        async with semaphore:
            return await load_metric(symbol)

    universe = list(dict.fromkeys([str(symbol).upper().strip() for symbol in _GENERAL_MOVER_UNIVERSE if str(symbol).strip()]))
    metrics = [item for item in await asyncio.gather(*[load_metric_limited(symbol) for symbol in universe]) if item]
    if not metrics:
        cached_last_snapshot = get_cached_research("GLOBAL", _MOVER_SNAPSHOT_LAST_KEY)
        if cached_last_snapshot:
            return {**cached_last_snapshot, "stale": True}

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
        "target_universe_size": len(universe),
    }
    set_cached_research("GLOBAL", _MOVER_SNAPSHOT_KEY, payload, ttl_minutes=60)
    set_cached_research("GLOBAL", _MOVER_SNAPSHOT_LAST_KEY, payload, ttl_minutes=7 * 24 * 60)
    return payload
