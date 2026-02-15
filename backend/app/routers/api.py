from __future__ import annotations

import asyncio
import base64
import binascii
import re
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.schemas import ResearchRequest
from app.services.agent_logger import get_recent_activity, log_agent_activity
from app.services.browserbase_scraper import run_deep_research
from app.services.llm import parse_tracker_instruction, tracker_agent_chat_response, tracker_context_query_response
from app.services.macro import get_macro_indicators
from app.services.market_data import fetch_candles, fetch_metric, resolve_symbol_input
from app.services.prediction_markets import fetch_kalshi_markets, fetch_polymarket_markets
from app.services.research_cache import get_cached_research, set_cached_research
from app.services.sentiment import get_x_sentiment, run_research
from app.services.tracker_csv import (
    append_agent_response_csv,
    read_agent_response_csv_tail,
    read_alert_context_csv_tail,
)
from app.services.tracker_repository import tracker_repo
from app.services.user_context import get_user_id_from_request
from app.services.user_preferences import get_favorites as get_user_favorites
from app.services.user_preferences import get_watchlist as get_user_watchlist
from app.services.user_preferences import set_favorites as set_user_favorites
from app.services.user_preferences import set_watchlist as set_user_watchlist
from app.schemas import SimulationStartRequest

router = APIRouter(prefix="/api", tags=["api"])


class PokeInboundRequest(BaseModel):
    message: str


class TrackerAgentCreateRequest(BaseModel):
    symbol: str | None = None
    name: str | None = None
    triggers: dict[str, Any] = Field(default_factory=dict)
    auto_simulate: bool = False
    create_prompt: str | None = None


class TrackerAgentPatchRequest(BaseModel):
    status: str | None = None
    name: str | None = None
    triggers: dict[str, Any] | None = None
    auto_simulate: bool | None = None


class TrackerEmitAlertRequest(BaseModel):
    symbol: str
    trigger_reason: str
    narrative: str | None = None
    market_snapshot: dict[str, Any] = Field(default_factory=dict)
    investigation_data: dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    agent_id: str | None = None
    simulation_id: str | None = None


class TrackerNLCreateRequest(BaseModel):
    prompt: str
    user_id: str | None = None


class TrackerAgentInteractRequest(BaseModel):
    message: str
    user_id: str | None = None


class TrackerContextQueryRequest(BaseModel):
    question: str
    user_id: str | None = None
    run_limit: int = 40
    history_limit: int = 40
    csv_limit: int = 120


def normalize_timeframe(value: str | None) -> str:
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


def sanitize_tracker_triggers(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    out: dict[str, Any] = {}

    def _num(key: str, default: float, low: float, high: float) -> float:
        value = raw.get(key, default)
        if isinstance(value, bool):
            return default
        try:
            parsed = float(value)
        except Exception:
            parsed = default
        return max(low, min(high, parsed))

    out["price_change_pct"] = _num("price_change_pct", 2.0, 0.1, 25.0)
    out["volume_spike_ratio"] = _num("volume_spike_ratio", 1.8, 1.0, 10.0)
    out["sentiment_bearish_threshold"] = _num("sentiment_bearish_threshold", -0.25, -1.0, 0.2)
    out["sentiment_bullish_threshold"] = _num("sentiment_bullish_threshold", 0.25, -0.2, 1.0)
    out["x_bearish_threshold"] = _num("x_bearish_threshold", -0.25, -1.0, 0.2)
    out["poll_interval_seconds"] = int(_num("poll_interval_seconds", 120.0, 30.0, 3600.0))
    out["report_interval_seconds"] = int(_num("report_interval_seconds", float(out["poll_interval_seconds"]), 30.0, 86400.0))
    out["simulate_on_alert"] = bool(raw.get("simulate_on_alert", False))

    schedule_mode = str(raw.get("schedule_mode") or "realtime").strip().lower()
    if schedule_mode not in {"realtime", "hourly", "daily", "custom"}:
        schedule_mode = "realtime"
    out["schedule_mode"] = schedule_mode
    if schedule_mode == "hourly":
        out["poll_interval_seconds"] = 3600
        out["report_interval_seconds"] = max(out["report_interval_seconds"], 3600)
    elif schedule_mode == "daily":
        out["poll_interval_seconds"] = 86400
        out["report_interval_seconds"] = max(out["report_interval_seconds"], 86400)

    if schedule_mode == "daily":
        daily_run_time_raw = str(raw.get("daily_run_time") or "09:30").strip()
        daily_match = re.match(r"^(\d{1,2}):(\d{2})$", daily_run_time_raw)
        if daily_match:
            hh = max(0, min(23, int(daily_match.group(1))))
            mm = max(0, min(59, int(daily_match.group(2))))
            out["daily_run_time"] = f"{hh:02d}:{mm:02d}"
        else:
            out["daily_run_time"] = "09:30"

    timezone_raw = str(raw.get("timezone") or "America/New_York").strip()
    try:
        ZoneInfo(timezone_raw)
        out["timezone"] = timezone_raw
    except Exception:
        out["timezone"] = "America/New_York"

    start_at_raw = raw.get("start_at")
    if start_at_raw:
        token = str(start_at_raw).strip()
        if token:
            normalized_start_at: str | None = None
            try:
                stamp = datetime.fromisoformat(token.replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=ZoneInfo(out["timezone"]))
                normalized_start_at = stamp.astimezone(timezone.utc).isoformat()
            except Exception:
                normalized_start_at = None
            if normalized_start_at:
                out["start_at"] = normalized_start_at

    baseline_mode = str(raw.get("baseline_mode") or "prev_close").strip().lower()
    if baseline_mode not in {"prev_close", "last_check", "last_alert", "session_open"}:
        baseline_mode = "prev_close"
    out["baseline_mode"] = baseline_mode

    tool_mode = str(raw.get("tool_mode") or "auto").strip().lower()
    if tool_mode not in {"auto", "manual"}:
        tool_mode = "auto"
    out["tool_mode"] = tool_mode

    report_mode_raw = str(raw.get("report_mode") or "").strip().lower()
    report_mode_aliases = {
        "alerts": "triggers_only",
        "alerts_only": "triggers_only",
        "trigger": "triggers_only",
        "trigger_only": "triggers_only",
        "report": "periodic",
        "reports": "periodic",
        "periodic_report": "periodic",
        "mixed": "hybrid",
    }
    report_mode = report_mode_aliases.get(report_mode_raw, report_mode_raw or "triggers_only")
    if report_mode not in {"triggers_only", "periodic", "hybrid"}:
        report_mode = "triggers_only"
    out["report_mode"] = report_mode

    allowed_tools = {
        "price",
        "volume",
        "sentiment",
        "news",
        "prediction_markets",
        "deep_research",
        "simulation",
    }
    raw_tools = raw.get("tools")
    if isinstance(raw_tools, list):
        cleaned_tools = []
        seen_tools: set[str] = set()
        for item in raw_tools:
            token = str(item).strip().lower()
            if token not in allowed_tools or token in seen_tools:
                continue
            seen_tools.add(token)
            cleaned_tools.append(token)
        if cleaned_tools:
            out["tools"] = cleaned_tools

    allowed_sources = {"perplexity", "x", "reddit", "prediction_markets", "deep"}
    raw_sources = raw.get("research_sources")
    if isinstance(raw_sources, list):
        cleaned_sources = []
        seen_sources: set[str] = set()
        for item in raw_sources:
            token = str(item).strip().lower()
            if token not in allowed_sources or token in seen_sources:
                continue
            seen_sources.add(token)
            cleaned_sources.append(token)
        if cleaned_sources:
            out["research_sources"] = cleaned_sources

    allowed_channels = {"twilio", "poke"}
    raw_channels = raw.get("notify_channels")
    if isinstance(raw_channels, list):
        cleaned_channels = []
        seen_channels: set[str] = set()
        for item in raw_channels:
            token = str(item).strip().lower()
            if token not in allowed_channels or token in seen_channels:
                continue
            seen_channels.add(token)
            cleaned_channels.append(token)
        if cleaned_channels:
            out["notify_channels"] = cleaned_channels

    phone_raw = str(raw.get("notify_phone") or "").strip()
    if phone_raw:
        out["notify_phone"] = phone_raw[:40]

    if "research_timeframe" in raw and isinstance(raw.get("research_timeframe"), str):
        out["research_timeframe"] = normalize_timeframe(str(raw["research_timeframe"]))
    return out


def _extract_phone_candidate(text: str) -> str | None:
    match = re.search(r"(\+?\d[\d\-\s\(\)]{8,}\d)", text or "")
    if not match:
        return None
    phone = match.group(1).strip()
    return phone[:40] if phone else None


def _extract_interval_seconds(text: str) -> int | None:
    match = re.search(r"every\s+(\d+)\s*(second|sec|minute|min|hour|hr)s?\b", text or "", flags=re.IGNORECASE)
    if not match:
        return None
    magnitude = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("sec"):
        return max(30, magnitude)
    if unit.startswith("hour") or unit == "hr":
        return max(30, magnitude * 3600)
    return max(30, magnitude * 60)


def _extract_explicit_research_sources(text: str) -> tuple[list[str], bool]:
    normalized = re.sub(r"[-_/]+", " ", text.lower())
    lower = f" {normalized} "
    found: list[str] = []
    if " reddit " in lower:
        found.append("reddit")
    if " perplexity " in lower:
        found.append("perplexity")
    if " x " in lower or " twitter " in lower:
        found.append("x")
    if " prediction market" in lower or " polymarket" in lower or " kalshi" in lower:
        found.append("prediction_markets")
    if " deep research" in lower or " browserbase" in lower:
        found.append("deep")
    found = list(dict.fromkeys(found))

    exclusive_markers = {" only ", " just ", " strictly ", " exclusively "}
    is_exclusive = any(marker in lower for marker in exclusive_markers)
    return found, is_exclusive


def _extract_explicit_tools(text: str) -> list[str]:
    lower = f" {text.lower()} "
    tools: list[str] = []
    if any(token in lower for token in {" price ", " drops ", " up ", " down ", " move ", "%"}):
        tools.append("price")
    if " volume " in lower:
        tools.append("volume")
    if " sentiment " in lower:
        tools.append("sentiment")
    if any(token in lower for token in {" news ", " catalyst ", " catalysts "}):
        tools.append("news")
    if any(token in lower for token in {" prediction market", " polymarket", " kalshi"}):
        tools.append("prediction_markets")
    if any(token in lower for token in {" deep research", " browserbase"}):
        tools.append("deep_research")
    if any(token in lower for token in {" simulate ", " simulation ", " backtest ", " scenario ", " sandbox "}):
        tools.append("simulation")
    return list(dict.fromkeys(tools))


def _extract_daily_run_time(text: str) -> str | None:
    match = re.search(r"\bat\s+(\d{1,2}):(\d{2})\s*(am|pm)?\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    ampm = (match.group(3) or "").lower().strip()
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return f"{hour:02d}:{minute:02d}"


def _extract_timezone_hint(text: str) -> str | None:
    lower = text.lower()
    aliases = {
        " est ": "America/New_York",
        " edt ": "America/New_York",
        " eastern ": "America/New_York",
        " pst ": "America/Los_Angeles",
        " pdt ": "America/Los_Angeles",
        " pacific ": "America/Los_Angeles",
        " cst ": "America/Chicago",
        " central ": "America/Chicago",
        " mst ": "America/Denver",
        " mountain ": "America/Denver",
        " utc ": "UTC",
        " gmt ": "UTC",
    }
    padded = f" {lower} "
    for needle, zone in aliases.items():
        if needle in padded:
            return zone

    for candidate in re.findall(r"\b[A-Za-z_]+/[A-Za-z_]+(?:/[A-Za-z_]+)?\b", text):
        try:
            ZoneInfo(candidate)
            return candidate
        except Exception:
            continue
    return None


def _parse_clock(hour_token: str, minute_token: str, ampm_token: str | None = None) -> tuple[int, int]:
    hour = int(hour_token)
    minute = int(minute_token)
    ampm = (ampm_token or "").strip().lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return hour, minute


def _extract_start_at(prompt: str, timezone_name: str) -> str | None:
    lower = prompt.lower()

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = timezone.utc
    now = datetime.now(timezone.utc).astimezone(tz)

    if any(token in lower for token in {"start now", "start immediately", "run now", "immediately"}):
        return now.astimezone(timezone.utc).isoformat()

    relative = re.search(r"start\s+in\s+(\d+)\s*(minute|min|hour|hr|day|week)s?\b", lower)
    if relative:
        magnitude = int(relative.group(1))
        unit = relative.group(2)
        if unit.startswith("min"):
            delta = timedelta(minutes=magnitude)
        elif unit.startswith("hour") or unit == "hr":
            delta = timedelta(hours=magnitude)
        elif unit.startswith("day"):
            delta = timedelta(days=magnitude)
        else:
            delta = timedelta(weeks=magnitude)
        return (now + delta).astimezone(timezone.utc).isoformat()

    date_match = re.search(
        r"start\s+(?:on\s+)?(\d{4}-\d{2}-\d{2})(?:[ t](\d{1,2}):(\d{2})(?:[:](\d{2}))?\s*(am|pm)?)?",
        lower,
        flags=re.IGNORECASE,
    )
    if date_match:
        date_token = date_match.group(1)
        hh = date_match.group(2) or "09"
        mm = date_match.group(3) or "30"
        ss = date_match.group(4) or "00"
        try:
            hour, minute = _parse_clock(hh, mm, date_match.group(5))
            second = max(0, min(59, int(ss)))
            target_local = datetime.fromisoformat(f"{date_token}T{hour:02d}:{minute:02d}:{second:02d}").replace(tzinfo=tz)
            return target_local.astimezone(timezone.utc).isoformat()
        except Exception:
            pass

    tomorrow_match = re.search(r"start\s+tomorrow(?:\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm)?)?", lower, flags=re.IGNORECASE)
    if tomorrow_match:
        hh = tomorrow_match.group(1) or "09"
        mm = tomorrow_match.group(2) or "30"
        hour, minute = _parse_clock(hh, mm, tomorrow_match.group(3))
        target_local = (now + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        return target_local.astimezone(timezone.utc).isoformat()

    time_match = re.search(r"start\s+(?:at|from)\s+(\d{1,2}):(\d{2})\s*(am|pm)?", lower, flags=re.IGNORECASE)
    if time_match:
        hour, minute = _parse_clock(time_match.group(1), time_match.group(2), time_match.group(3))
        target_local = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target_local <= now:
            target_local = target_local + timedelta(days=1)
        return target_local.astimezone(timezone.utc).isoformat()

    return None


def _apply_prompt_overrides(prompt: str, raw_triggers: dict[str, Any] | None) -> dict[str, Any]:
    lower = (prompt or "").lower()
    merged = dict(raw_triggers or {})

    explicit_interval = _extract_interval_seconds(prompt)
    if explicit_interval is not None:
        merged["schedule_mode"] = "custom"
        merged["poll_interval_seconds"] = explicit_interval
        merged["report_interval_seconds"] = explicit_interval
    elif "realtime" in lower or "real time" in lower:
        # Product default for realtime crawler behavior.
        merged["schedule_mode"] = "realtime"
        merged.setdefault("poll_interval_seconds", 120)
        merged.setdefault("report_interval_seconds", 120)
    elif "hourly" in lower:
        merged["schedule_mode"] = "hourly"
        merged["poll_interval_seconds"] = 3600
        merged["report_interval_seconds"] = 3600
    elif "daily" in lower:
        merged["schedule_mode"] = "daily"
        merged["poll_interval_seconds"] = 86400
        merged.setdefault("report_interval_seconds", 86400)
        daily_run_time = _extract_daily_run_time(prompt)
        if daily_run_time:
            merged["daily_run_time"] = daily_run_time

    wants_report = any(token in lower for token in {"report every", "summary every", "update every", "send report", "periodic report"})
    wants_alert = any(token in lower for token in {"alert", "trigger", "notify me when", "if price", "if it moves"})
    if wants_report and wants_alert:
        merged["report_mode"] = "hybrid"
    elif wants_report:
        merged["report_mode"] = "periodic"
    elif any(token in lower for token in {"trigger only", "alerts only", "only when"}) and not wants_report:
        merged["report_mode"] = "triggers_only"

    phone = _extract_phone_candidate(prompt)
    if phone:
        merged["notify_phone"] = phone
        channels = merged.get("notify_channels")
        if not isinstance(channels, list):
            channels = []
        tokens = [str(item).strip().lower() for item in channels if str(item).strip()]
        if "twilio" not in tokens:
            tokens.append("twilio")
        merged["notify_channels"] = tokens

    # Baseline comparison defaults to prev close unless user specifies another anchor.
    if "previous close" in lower or "prev close" in lower:
        merged["baseline_mode"] = "prev_close"
    elif "from open" in lower or "since open" in lower or "session open" in lower:
        merged["baseline_mode"] = "session_open"
    elif "from last check" in lower:
        merged["baseline_mode"] = "last_check"
    elif "from last alert" in lower:
        merged["baseline_mode"] = "last_alert"

    # Timezone hints.
    timezone_hint = _extract_timezone_hint(prompt)
    if timezone_hint:
        merged["timezone"] = timezone_hint

    start_at = _extract_start_at(prompt, str(merged.get("timezone") or "America/New_York"))
    if start_at:
        merged["start_at"] = start_at

    explicit_sources, exclusive_sources = _extract_explicit_research_sources(prompt)
    explicit_tools = _extract_explicit_tools(prompt)
    if explicit_tools:
        merged["tool_mode"] = "manual"
        existing_tools = merged.get("tools")
        existing_tokens = [str(item).strip().lower() for item in existing_tools] if isinstance(existing_tools, list) else []
        merged["tools"] = list(dict.fromkeys(existing_tokens + explicit_tools))
        if "simulation" in explicit_tools:
            merged["simulate_on_alert"] = True

    if explicit_sources:
        existing_sources = merged.get("research_sources")
        existing_tokens = [str(item).strip().lower() for item in existing_sources] if isinstance(existing_sources, list) else []
        merged["research_sources"] = explicit_sources if exclusive_sources else list(dict.fromkeys(existing_tokens + explicit_sources))
        # If user explicitly asks for one source only, pin to sentiment/news tools.
        if exclusive_sources:
            merged["tool_mode"] = "manual"
            base_tools = [tool for tool in explicit_tools if tool in {"price", "volume", "simulation"}]
            sentiment_stack = ["sentiment", "news"]
            merged["tools"] = list(dict.fromkeys(base_tools + sentiment_stack))

    if "24/7" in lower or "24x7" in lower or "always on" in lower:
        merged["schedule_mode"] = "realtime"
        merged["poll_interval_seconds"] = int(merged.get("poll_interval_seconds") or 120)
        merged["report_interval_seconds"] = int(merged.get("report_interval_seconds") or merged["poll_interval_seconds"])

    if any(token in lower for token in {"sentiment analysis", "market sentiment", "sentiment report", "sentiment summary"}):
        merged.setdefault("tool_mode", "manual")
        tools = merged.get("tools")
        existing_tools = [str(item).strip().lower() for item in tools] if isinstance(tools, list) else []
        merged["tools"] = list(dict.fromkeys(existing_tools + ["sentiment", "news"]))
        if any(token in lower for token in {"notify", "notification", "send me", "text me", "sms"}):
            merged["report_mode"] = "periodic" if merged.get("report_mode") != "hybrid" else "hybrid"

    if any(
        token in lower
        for token in {
            "bad thing",
            "bad news",
            "negative news",
            "bearish news",
            "something bad",
            "investigate",
            "search",
        }
    ):
        merged.setdefault("tool_mode", "manual")
        tools = merged.get("tools")
        existing_tools = [str(item).strip().lower() for item in tools] if isinstance(tools, list) else []
        merged["tools"] = list(dict.fromkeys(existing_tools + ["news", "sentiment"]))
        sources = merged.get("research_sources")
        existing_sources = [str(item).strip().lower() for item in sources] if isinstance(sources, list) else []
        merged["research_sources"] = list(dict.fromkeys(existing_sources + ["perplexity"]))
        merged["sentiment_bearish_threshold"] = max(-0.1, float(merged.get("sentiment_bearish_threshold") or -0.25))
        if merged.get("report_mode") == "triggers_only":
            merged["report_mode"] = "hybrid"

    if any(token in lower for token in {"llm decide", "auto decide tools", "decide tools", "decide what to use", "use any tool", "auto tool"}):
        merged["tool_mode"] = "auto"
        merged.pop("tools", None)

    # Default explicit schedule when user says "real-time" preferences without interval.
    if str(merged.get("schedule_mode") or "").lower() == "realtime":
        merged.setdefault("poll_interval_seconds", 120)
        merged.setdefault("report_interval_seconds", 120)
    return merged


def _chart_params_from_message(text: str) -> tuple[str, str]:
    lower = text.lower()
    period = "6mo"
    interval = "1d"
    if "1y" in lower or "year" in lower:
        period = "1y"
    elif "2y" in lower:
        period = "2y"
    elif "5y" in lower:
        period = "5y"
    elif "10y" in lower:
        period = "10y"
    elif "1mo" in lower or "month" in lower:
        period = "1mo"
    elif "3mo" in lower:
        period = "3mo"
    elif "max" in lower:
        period = "max"

    if "1wk" in lower or "weekly" in lower:
        interval = "1wk"
    elif "1mo interval" in lower or "monthly" in lower:
        interval = "1mo"
    return period, interval


@router.get("/health")
async def api_health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ticker/{symbol}/quote")
async def ticker_quote(symbol: str) -> dict[str, Any]:
    try:
        resolved_symbol = resolve_symbol_input(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    key = "quote:5m"
    last_key = "quote:last"
    cached = get_cached_research(resolved_symbol, key)
    if cached:
        return cached
    try:
        metric = await asyncio.to_thread(fetch_metric, resolved_symbol)
    except Exception as exc:
        last = get_cached_research(resolved_symbol, last_key)
        if last:
            return {**last, "stale": True, "provider_error": str(exc)}
        raise HTTPException(status_code=502, detail=f"Market data provider unavailable for {resolved_symbol.upper()}: {exc}")
    payload = metric.model_dump()
    set_cached_research(resolved_symbol, key, payload, ttl_minutes=5)
    set_cached_research(resolved_symbol, last_key, payload, ttl_minutes=24 * 60)
    return payload


@router.get("/ticker/{symbol}/ai-research")
async def ticker_ai_research(symbol: str, request: Request, timeframe: str = "7d") -> dict[str, Any]:
    frame = normalize_timeframe(timeframe)
    cache_key = f"ai_research:{frame}:15m"
    cached = get_cached_research(symbol, cache_key)
    if cached:
        return cached
    settings = request.app.state.settings
    data = await run_research(ResearchRequest(ticker=symbol, timeframe=frame, include_prediction_markets=True), settings)
    payload = {
        "ticker": symbol.upper(),
        "timeframe": frame,
        "summary": "\n".join(data.narratives),
        "source_breakdown": [entry.model_dump() for entry in data.source_breakdown],
        "citations": [link.model_dump() for link in data.tool_links],
        "recommendation": data.recommendation,
    }
    set_cached_research(symbol, cache_key, payload, ttl_minutes=15)
    return payload


@router.get("/ticker/{symbol}/sentiment")
async def ticker_sentiment(symbol: str, request: Request, timeframe: str = "7d") -> dict[str, Any]:
    frame = normalize_timeframe(timeframe)
    cache_key = f"sentiment:{frame}:10m"
    cached = get_cached_research(symbol, cache_key)
    if cached:
        return cached
    settings = request.app.state.settings
    data = await run_research(ResearchRequest(ticker=symbol, timeframe=frame, include_prediction_markets=True), settings)
    breakdown = {entry.source: entry.score for entry in data.source_breakdown}
    payload = {
        "ticker": symbol.upper(),
        "timeframe": frame,
        "composite_score": data.aggregate_sentiment,
        "recommendation": data.recommendation,
        "breakdown": breakdown,
        "weights": {"perplexity": 0.45, "reddit": 0.30, "x": 0.25},
    }
    set_cached_research(symbol, cache_key, payload, ttl_minutes=10)
    return payload


@router.get("/ticker/{symbol}/x-sentiment")
async def ticker_x_sentiment(symbol: str, request: Request) -> dict[str, Any]:
    cache_key = "x_sentiment:5m"
    cached = get_cached_research(symbol, cache_key)
    if cached:
        return cached
    settings = request.app.state.settings
    payload = await get_x_sentiment(symbol, settings)
    set_cached_research(symbol, cache_key, payload, ttl_minutes=5)
    return payload


@router.get("/prediction-markets")
async def prediction_markets(request: Request, query: str = "fed") -> dict[str, Any]:
    cache_key = f"prediction_markets:{query.lower()}:10m"
    cached = get_cached_research(query, cache_key)
    if cached:
        return cached
    settings = request.app.state.settings
    kalshi, poly = await asyncio.gather(
        fetch_kalshi_markets(query, settings),
        fetch_polymarket_markets(query, settings),
    )
    payload = {"query": query, "kalshi": kalshi, "polymarket": poly}
    set_cached_research(query, cache_key, payload, ttl_minutes=10)
    return payload


@router.get("/ticker/{symbol}")
async def ticker_full(symbol: str, request: Request, timeframe: str = "7d") -> dict[str, Any]:
    frame = normalize_timeframe(timeframe)
    cache_key = f"ticker_bundle:{frame}:5m"
    cached = get_cached_research(symbol, cache_key)
    if cached:
        return cached
    settings = request.app.state.settings
    metric_task = asyncio.to_thread(fetch_metric, symbol)
    research_task = run_research(ResearchRequest(ticker=symbol, timeframe=frame, include_prediction_markets=True), settings)
    macro_task = get_macro_indicators(settings)
    deep_task = run_deep_research(symbol, settings)

    metric, research, macro, deep = await asyncio.gather(metric_task, research_task, macro_task, deep_task)
    payload = {
        "symbol": symbol.upper(),
        "timeframe": frame,
        "quote": metric.model_dump(),
        "research": research.model_dump(),
        "macro": macro,
        "deep_research": deep,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    set_cached_research(symbol, cache_key, payload, ttl_minutes=5)
    return payload


@router.post("/research/deep/{symbol}")
async def deep_research(symbol: str, request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    return await run_deep_research(symbol, settings)


def _require_tracker_user_id(request: Request, explicit_user_id: str | None = None) -> str:
    resolved_user_id = explicit_user_id or get_user_id_from_request(request)
    if not resolved_user_id:
        raise HTTPException(status_code=401, detail="Sign in required to persist tracker agents.")
    return resolved_user_id


def _require_agent_start_time(triggers: dict[str, Any]) -> None:
    start_at = str(triggers.get("start_at") or "").strip()
    if not start_at:
        raise HTTPException(
            status_code=422,
            detail="start_at is required. Set a start time in the future or 'start now' in your instruction.",
        )


_SYMBOL_STOPWORDS = {
    "THE",
    "FROM",
    "ONLY",
    "WITH",
    "NEWS",
    "AND",
    "FOR",
    "WHEN",
    "THEN",
    "ALERT",
    "ALERTS",
    "TRACK",
    "MONITOR",
    "WATCH",
    "REPORT",
    "REPORTS",
    "SENTIMENT",
    "SIM",
}

_COMPANY_ALIAS_TO_SYMBOL = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
    "AMAZON": "AMZN",
    "META": "META",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "NETFLIX": "NFLX",
    "AMD": "AMD",
    "INTEL": "INTC",
    "PALANTIR": "PLTR",
    "SP500": "SPY",
    "S&P 500": "SPY",
    "S P 500": "SPY",
}


def _extract_symbol_candidates_from_prompt(prompt: str) -> list[str]:
    text = prompt or ""
    candidates: list[str] = []

    def _push(raw: str) -> None:
        token = re.sub(r"\s+", " ", str(raw or "")).strip(" \t\r\n.,;:!?")
        if not token:
            return
        if token.upper() in _SYMBOL_STOPWORDS:
            return
        if token not in candidates:
            candidates.append(token)

    for match in re.findall(r"\$([A-Za-z][A-Za-z0-9.\-]{0,9})\b", text):
        _push(match)
    for match in re.findall(r"\(([A-Za-z][A-Za-z0-9.\-]{0,9})\)", text):
        _push(match)
    for match in re.findall(r"\b([A-Z]{1,5})\b", text.upper()):
        _push(match)

    phrase_patterns = [
        r"(?:track|monitor|watch|follow|analy[sz]e|research|cover)\s+([A-Za-z0-9&.\-\s]{2,90}?)(?:\s+(?:from|for|with|and|at|on|if|when|where|that|to)\b|$)",
        r"(?:for|about)\s+([A-Za-z0-9&.\-\s]{2,90}?)(?:\s+(?:from|with|and|at|on|if|when|where|that|to)\b|$)",
    ]
    for pattern in phrase_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            _push(match)

    return candidates


def _resolve_symbol_candidate(candidate: str) -> str:
    token = str(candidate or "").strip()
    if not token:
        raise ValueError("Ticker input is empty.")
    alias_key = re.sub(r"[^A-Za-z0-9&]+", " ", token).strip().upper()
    if alias_key in _COMPANY_ALIAS_TO_SYMBOL:
        return _COMPANY_ALIAS_TO_SYMBOL[alias_key]

    if re.fullmatch(r"[A-Za-z]{4,}", token):
        try:
            return resolve_symbol_input(f"{token} stock")
        except ValueError:
            pass
    return resolve_symbol_input(token)


def _infer_symbol_from_prompt(prompt: str) -> str | None:
    for candidate in _extract_symbol_candidates_from_prompt(prompt):
        try:
            return _resolve_symbol_candidate(candidate)
        except Exception:
            continue
    return None


def _resolve_tracker_symbol(
    *,
    explicit_symbol: str | None,
    create_prompt: str,
    parsed_prompt: dict[str, Any] | None,
) -> str:
    if explicit_symbol and str(explicit_symbol).strip():
        try:
            return _resolve_symbol_candidate(str(explicit_symbol))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    inferred = _infer_symbol_from_prompt(create_prompt)
    if inferred:
        return inferred

    parsed_prompt = parsed_prompt or {}
    parsed_symbol = str(parsed_prompt.get("symbol") or "").strip().upper()
    if parsed_symbol:
        prompt_upper = create_prompt.upper()
        looks_like_fallback = parsed_symbol == "AAPL" and ("AAPL" not in prompt_upper and "APPLE" not in prompt_upper)
        if not looks_like_fallback:
            try:
                return _resolve_symbol_candidate(parsed_symbol)
            except ValueError:
                pass

    parsed_name = str(parsed_prompt.get("name") or "").strip()
    if parsed_name:
        name_hint = re.sub(r"\b(associate|tracker|agent)\b", "", parsed_name, flags=re.IGNORECASE).strip()
        if name_hint:
            try:
                return _resolve_symbol_candidate(name_hint)
            except ValueError:
                pass

    if create_prompt:
        raise HTTPException(
            status_code=422,
            detail="Could not infer ticker symbol from instruction. Include a ticker or company name in the prompt.",
        )
    raise HTTPException(
        status_code=422,
        detail="Ticker symbol is required unless create_prompt includes a ticker or company name.",
    )


@router.post("/tracker/agents")
async def create_tracker_agent(payload: TrackerAgentCreateRequest, request: Request, user_id: str | None = None) -> dict[str, Any]:
    resolved_user_id = _require_tracker_user_id(request, user_id)
    create_prompt = str(payload.create_prompt or "").strip()
    parsed_prompt: dict[str, Any] = {}
    if create_prompt:
        settings = request.app.state.settings
        try:
            maybe_parsed = await parse_tracker_instruction(settings, create_prompt)
            if isinstance(maybe_parsed, dict):
                parsed_prompt = maybe_parsed
        except Exception:
            parsed_prompt = {}

    prompt_triggers = (
        _apply_prompt_overrides(
            create_prompt,
            parsed_prompt.get("triggers") if isinstance(parsed_prompt.get("triggers"), dict) else {},
        )
        if create_prompt
        else {}
    )
    merged_triggers = {
        **prompt_triggers,
        **(payload.triggers if isinstance(payload.triggers, dict) else {}),
    }
    clean_triggers = sanitize_tracker_triggers(merged_triggers)
    _require_agent_start_time(clean_triggers)
    symbol = _resolve_tracker_symbol(
        explicit_symbol=payload.symbol,
        create_prompt=create_prompt,
        parsed_prompt=parsed_prompt,
    )
    parsed_name = str(parsed_prompt.get("name") or "").strip() if isinstance(parsed_prompt, dict) else ""
    name = str(payload.name or "").strip() or parsed_name or f"{symbol} Tracker Agent"
    auto_simulate = bool(
        payload.auto_simulate
        or (bool(parsed_prompt.get("auto_simulate")) if isinstance(parsed_prompt, dict) else False)
        or bool(clean_triggers.get("simulate_on_alert"))
    )
    try:
        # Strict persistence prevents local-memory ghost agents that never land in Supabase.
        agent = tracker_repo.create_agent(
            user_id=resolved_user_id,
            symbol=symbol,
            name=name,
            triggers=clean_triggers,
            auto_simulate=auto_simulate,
            strict_persistence=True,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        tracker_repo.create_history(
            user_id=resolved_user_id,
            agent_id=str(agent.get("id")),
            event_type="create_prompt" if create_prompt else "system_update",
            raw_prompt=create_prompt or None,
            parsed_intent=parsed_prompt if parsed_prompt else {"intent": "create_agent", "symbol": symbol},
            trigger_snapshot=agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {},
            note=(
                "Agent created via structured tracker create endpoint."
                if not create_prompt
                else "Agent created via structured tracker create endpoint with initial manager prompt."
            ),
            strict_persistence=True,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    await log_agent_activity(
        module="tracker",
        agent_name=str(agent.get("name") or f"{symbol} Tracker"),
        action=f"Tracker agent deployed for {str(agent.get('symbol') or symbol).upper()}",
        status="success",
        user_id=resolved_user_id,
        details={
            "agent_id": agent.get("id"),
            "symbol": str(agent.get("symbol") or symbol).upper(),
            "description": "Agent is actively monitoring configured trigger thresholds.",
            "triggers": agent.get("triggers") or merged_triggers,
            "create_prompt": create_prompt or None,
            "parsed_prompt": parsed_prompt or None,
        },
    )
    return agent


@router.get("/tracker/agents")
async def list_tracker_agents(request: Request, user_id: str | None = None) -> list[dict[str, Any]]:
    resolved_user_id = _require_tracker_user_id(request, user_id)
    return tracker_repo.list_agents(user_id=resolved_user_id)


@router.patch("/tracker/agents/{agent_id}")
async def patch_tracker_agent(agent_id: str, payload: TrackerAgentPatchRequest, request: Request, user_id: str | None = None) -> dict[str, Any]:
    resolved_user_id = _require_tracker_user_id(request, user_id)
    existing = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    if existing is None:
        return {"error": "not_found"}
    updates = payload.model_dump(exclude_none=True)
    if "triggers" in updates and isinstance(updates.get("triggers"), dict):
        existing_triggers = sanitize_tracker_triggers(existing.get("triggers")) if isinstance(existing.get("triggers"), dict) else {}
        merged_triggers = {
            **existing_triggers,
            **sanitize_tracker_triggers(updates["triggers"]),
        }
        _require_agent_start_time(merged_triggers)
        updates["triggers"] = merged_triggers
    item = tracker_repo.update_agent(user_id=resolved_user_id, agent_id=agent_id, updates=updates)
    if item is None:
        return {"error": "not_found"}
    tracker_repo.create_history(
        user_id=resolved_user_id,
        agent_id=agent_id,
        event_type="system_update",
        parsed_intent={"intent": "update_agent"},
        trigger_snapshot=item.get("triggers") if isinstance(item.get("triggers"), dict) else {},
        note="Agent updated via structured tracker patch endpoint.",
    )
    await log_agent_activity(
        module="tracker",
        agent_name=str(item.get("name") or "Tracker Agent"),
        action=f"Agent configuration updated for {str(item.get('symbol') or '').upper()}",
        status="success",
        user_id=resolved_user_id,
        details={
            "agent_id": item.get("id"),
            "symbol": str(item.get("symbol") or "").upper(),
            "description": "Monitoring profile updated and running with new settings.",
            "updates": updates,
        },
    )
    return item


@router.get("/tracker/agents/{agent_id}")
async def get_tracker_agent(agent_id: str, request: Request, user_id: str | None = None) -> dict[str, Any]:
    resolved_user_id = _require_tracker_user_id(request, user_id)
    item = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return item


@router.get("/tracker/agents/{agent_id}/detail")
async def get_tracker_agent_detail(agent_id: str, request: Request, user_id: str | None = None) -> dict[str, Any]:
    resolved_user_id = _require_tracker_user_id(request, user_id)
    agent = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    symbol = str(agent.get("symbol", "")).upper()
    if symbol:
        try:
            quote = await asyncio.to_thread(fetch_metric, symbol)
        except Exception:
            quote = None
    else:
        quote = None

    alerts = tracker_repo.list_alerts(
        user_id=resolved_user_id,
        agent_id=agent_id,
        limit=20,
    )
    alert_context = tracker_repo.list_alert_context(
        user_id=resolved_user_id,
        agent_id=agent_id,
        limit=20,
    )
    thesis = tracker_repo.get_thesis(user_id=resolved_user_id, agent_id=agent_id)
    recent_runs = tracker_repo.list_runs(user_id=resolved_user_id, agent_id=agent_id, limit=20)
    activity = await get_recent_activity(limit=200, module="tracker")
    agent_name = str(agent.get("name", ""))
    actions = [
        row
        for row in activity
        if symbol in str(row.get("action", "")).upper()
        or agent_name.lower() in str(row.get("agent_name", "")).lower()
    ][:30]

    return {
        "agent": agent,
        "market": quote.model_dump() if quote else None,
        "recent_alerts": alerts,
        "recent_alert_context": alert_context,
        "recent_runs": recent_runs,
        "thesis": thesis,
        "recent_actions": actions,
    }


@router.get("/tracker/agents/{agent_id}/history")
async def get_tracker_agent_history(agent_id: str, request: Request, user_id: str | None = None, limit: int = 20) -> dict[str, Any]:
    resolved_user_id = _require_tracker_user_id(request, user_id)
    item = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    history = tracker_repo.list_history(user_id=resolved_user_id, agent_id=agent_id, limit=min(max(limit, 1), 200))
    return {"items": history}


@router.get("/tracker/agents/{agent_id}/context")
async def get_tracker_agent_context(
    agent_id: str,
    request: Request,
    user_id: str | None = None,
    run_limit: int = 40,
    history_limit: int = 40,
    csv_limit: int = 120,
) -> dict[str, Any]:
    resolved_user_id = _require_tracker_user_id(request, user_id)
    agent = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    runs = tracker_repo.list_runs(user_id=resolved_user_id, agent_id=agent_id, limit=min(max(run_limit, 1), 300))
    history = tracker_repo.list_history(user_id=resolved_user_id, agent_id=agent_id, limit=min(max(history_limit, 1), 300))
    alert_context = tracker_repo.list_alert_context(user_id=resolved_user_id, agent_id=agent_id, limit=min(max(run_limit, 1), 300))
    thesis = tracker_repo.get_thesis(user_id=resolved_user_id, agent_id=agent_id)
    csv_tail = await asyncio.to_thread(
        read_agent_response_csv_tail,
        user_id=resolved_user_id,
        agent_id=agent_id,
        limit=min(max(csv_limit, 1), 1000),
    )
    alert_context_csv = await asyncio.to_thread(
        read_alert_context_csv_tail,
        user_id=resolved_user_id,
        agent_id=agent_id,
        limit=min(max(csv_limit, 1), 1000),
    )
    return {
        "agent": agent,
        "thesis": thesis,
        "runs": runs,
        "history": history,
        "csv_export": csv_tail,
        "alert_context": alert_context,
        "alert_context_csv": alert_context_csv,
    }


@router.post("/tracker/agents/{agent_id}/context-query")
async def query_tracker_agent_context(
    agent_id: str,
    payload: TrackerContextQueryRequest,
    request: Request,
) -> dict[str, Any]:
    resolved_user_id = _require_tracker_user_id(request, payload.user_id)
    agent = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    runs = tracker_repo.list_runs(user_id=resolved_user_id, agent_id=agent_id, limit=min(max(payload.run_limit, 1), 300))
    history = tracker_repo.list_history(user_id=resolved_user_id, agent_id=agent_id, limit=min(max(payload.history_limit, 1), 300))
    alert_context = tracker_repo.list_alert_context(
        user_id=resolved_user_id,
        agent_id=agent_id,
        limit=min(max(payload.run_limit, 1), 300),
    )
    thesis = tracker_repo.get_thesis(user_id=resolved_user_id, agent_id=agent_id)
    csv_tail = await asyncio.to_thread(
        read_agent_response_csv_tail,
        user_id=resolved_user_id,
        agent_id=agent_id,
        limit=min(max(payload.csv_limit, 1), 1000),
    )
    alert_context_csv = await asyncio.to_thread(
        read_alert_context_csv_tail,
        user_id=resolved_user_id,
        agent_id=agent_id,
        limit=min(max(payload.csv_limit, 1), 1000),
    )
    context = {
        "agent": {
            "id": str(agent.get("id")),
            "name": str(agent.get("name") or ""),
            "symbol": str(agent.get("symbol") or ""),
            "status": str(agent.get("status") or ""),
            "triggers": agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {},
        },
        "thesis": thesis or {},
        "runs": runs,
        "history": history,
        "csv_rows": csv_tail.get("rows") if isinstance(csv_tail, dict) else [],
        "alert_context": alert_context,
        "alert_context_rows": alert_context_csv.get("rows") if isinstance(alert_context_csv, dict) else [],
    }
    settings = request.app.state.settings
    answer = await tracker_context_query_response(settings, question=payload.question, context=context)
    await log_agent_activity(
        module="tracker",
        agent_name=str(agent.get("name") or "Tracker Agent"),
        action=f"Context query executed for {str(agent.get('symbol') or '').upper()}",
        status="success",
        user_id=resolved_user_id,
        details={
            "agent_id": agent_id,
            "symbol": str(agent.get("symbol") or "").upper(),
            "description": "Answered manager query against stored tracker history, runs, and CSV logs.",
            "question": payload.question,
        },
    )
    return {
        "ok": True,
        "answer": answer,
        "context_meta": {
            "run_count": len(runs),
            "history_count": len(history),
            "csv_rows": len(csv_tail.get("rows") or []),
            "alert_context_count": len(alert_context),
            "alert_context_csv_rows": len(alert_context_csv.get("rows") or []),
            "csv_path": csv_tail.get("path"),
            "csv_bucket": csv_tail.get("bucket"),
            "alert_context_csv_path": alert_context_csv.get("path"),
            "alert_context_csv_bucket": alert_context_csv.get("bucket"),
        },
    }


@router.delete("/tracker/agents/{agent_id}")
async def delete_tracker_agent(agent_id: str, request: Request, user_id: str | None = None) -> dict[str, Any]:
    resolved_user_id = _require_tracker_user_id(request, user_id)
    existing = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    ok = tracker_repo.delete_agent(user_id=resolved_user_id, agent_id=agent_id)
    if ok:
        tracker_repo.create_history(
            user_id=resolved_user_id,
            agent_id=agent_id,
            event_type="system_update",
            parsed_intent={"intent": "delete_agent"},
            trigger_snapshot={},
            note="Agent deleted.",
        )
        await log_agent_activity(
            module="tracker",
            agent_name=str((existing or {}).get("name") or "Tracker Agent"),
            action=f"Agent stopped for {str((existing or {}).get('symbol') or '').upper()}",
            status="success",
            user_id=resolved_user_id,
            details={
                "agent_id": agent_id,
                "symbol": str((existing or {}).get("symbol") or "").upper(),
                "description": "Agent was deleted and is no longer monitoring this market.",
            },
        )
    return {"ok": ok}


@router.post("/tracker/agents/nl-create")
async def create_tracker_agent_nl(payload: TrackerNLCreateRequest, request: Request) -> dict[str, Any]:
    resolved_user_id = _require_tracker_user_id(request, payload.user_id)
    settings = request.app.state.settings
    parsed = await parse_tracker_instruction(settings, payload.prompt)

    symbol = _resolve_tracker_symbol(
        explicit_symbol=None,
        create_prompt=payload.prompt,
        parsed_prompt=parsed if isinstance(parsed, dict) else {},
    )
    name = str(parsed.get("name") or f"{symbol} Associate")
    triggers = sanitize_tracker_triggers(
        _apply_prompt_overrides(
            payload.prompt,
            parsed.get("triggers") if isinstance(parsed.get("triggers"), dict) else {},
        )
    )
    _require_agent_start_time(triggers)
    auto_simulate = bool(parsed.get("auto_simulate", False) or triggers.get("simulate_on_alert"))
    try:
        agent = tracker_repo.create_agent(
            user_id=resolved_user_id,
            symbol=symbol,
            name=name,
            triggers=triggers,
            auto_simulate=auto_simulate,
            strict_persistence=True,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        tracker_repo.create_history(
            user_id=resolved_user_id,
            agent_id=str(agent.get("id")),
            event_type="create_prompt",
            raw_prompt=payload.prompt,
            parsed_intent=parsed if isinstance(parsed, dict) else {},
            trigger_snapshot=agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {},
            note="Agent created from prompt.",
            strict_persistence=True,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    await log_agent_activity(
        module="tracker",
        agent_name=name,
        action=f"Created agent from natural-language instruction for {symbol}",
        status="success",
        user_id=resolved_user_id,
        details={
            "agent_id": agent.get("id"),
            "symbol": symbol,
            "description": "LLM parsed manager instruction into a deployable tracker associate profile.",
            "raw_prompt": payload.prompt,
            "parsed": parsed,
        },
    )
    return {"ok": True, "agent": agent, "parsed": parsed}


@router.post("/tracker/agents/{agent_id}/interact")
async def interact_tracker_agent(agent_id: str, payload: TrackerAgentInteractRequest, request: Request) -> dict[str, Any]:
    resolved_user_id = _require_tracker_user_id(request, payload.user_id)
    agent = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    if agent is None:
        return {"ok": False, "error": "not_found"}

    settings = request.app.state.settings
    symbol = str(agent.get("symbol", "")).upper()
    metric = await asyncio.to_thread(fetch_metric, symbol) if symbol else None
    research = await run_research(
        ResearchRequest(ticker=symbol or "AAPL", timeframe="7d", include_prediction_markets=False),
        settings,
    )
    research_state = {
        "aggregate_sentiment": research.aggregate_sentiment,
        "recommendation": research.recommendation,
        "breakdown": {entry.source: entry.score for entry in research.source_breakdown},
    }
    market_state = metric.model_dump() if metric else {}
    recent_history = tracker_repo.list_history(user_id=resolved_user_id, agent_id=agent_id, limit=12)
    recent_runs = tracker_repo.list_runs(user_id=resolved_user_id, agent_id=agent_id, limit=12)
    recent_alerts = tracker_repo.list_alerts(user_id=resolved_user_id, agent_id=agent_id, limit=12)
    recent_alert_context = tracker_repo.list_alert_context(user_id=resolved_user_id, agent_id=agent_id, limit=12)
    thesis = tracker_repo.get_thesis(user_id=resolved_user_id, agent_id=agent_id)
    latest_instruction = next(
        (
            str(row.get("raw_prompt") or "").strip()
            for row in recent_history
            if str(row.get("event_type") or "") in {"manager_instruction", "create_prompt"} and str(row.get("raw_prompt") or "").strip()
        ),
        "",
    )
    memory_context = {
        "latest_instruction": latest_instruction or None,
        "recent_history": recent_history[:8],
        "recent_runs": recent_runs[:8],
        "recent_alerts": recent_alerts[:8],
        "recent_alert_context": recent_alert_context[:8],
        "thesis": thesis or {},
    }
    reply = await tracker_agent_chat_response(
        settings,
        agent,
        market_state,
        research_state,
        payload.message,
        memory_context=memory_context,
    )
    parsed = await parse_tracker_instruction(settings, payload.message)
    intent = str(parsed.get("intent") or "")
    if isinstance(parsed.get("triggers"), dict):
        parsed["triggers"] = _apply_prompt_overrides(payload.message, parsed.get("triggers"))
    elif payload.message.strip():
        interval = _extract_interval_seconds(payload.message)
        wants_report = any(token in payload.message.lower() for token in {"report every", "summary every", "update every", "send report"})
        if interval is not None or wants_report:
            parsed["triggers"] = _apply_prompt_overrides(payload.message, {})
    message_lower = payload.message.lower()
    tool_outputs: dict[str, Any] = {}

    if any(token in message_lower for token in {"chart", "candlestick", "price action", "technical"}):
        period, interval = _chart_params_from_message(payload.message)
        points = await asyncio.to_thread(fetch_candles, symbol, period, interval)
        tool_outputs["chart"] = {
            "period": period,
            "interval": interval,
            "points": [point.model_dump() for point in points[-260:]],
        }
        await log_agent_activity(
            module="tracker",
            agent_name=str(agent.get("name") or "Tracker Associate"),
            action=f"Fetched chart data for {symbol} ({period}/{interval})",
            status="success",
            user_id=resolved_user_id,
            details={
                "agent_id": agent_id,
                "symbol": symbol,
                "description": "Generated chart dataset in response to manager request.",
                "period": period,
                "interval": interval,
                "points": len(tool_outputs["chart"]["points"]),
            },
        )

    if any(token in message_lower for token in {"research", "sentiment", "twitter", "x ", "reddit", "news"}):
        tool_outputs["research"] = research_state
        await log_agent_activity(
            module="tracker",
            agent_name=str(agent.get("name") or "Tracker Associate"),
            action=f"Pulled sentiment research snapshot for {symbol}",
            status="success",
            user_id=resolved_user_id,
            details={
                "agent_id": agent_id,
                "symbol": symbol,
                "description": "Included composite and source sentiment in agent response.",
                "research": research_state,
            },
        )

    if any(token in message_lower for token in {"simulate", "sandbox", "backtest", "scenario", "crash"}):
        orchestrator = request.app.state.orchestrator
        start_req = SimulationStartRequest(
            ticker=symbol,
            user_id=resolved_user_id,
            duration_seconds=90,
            initial_price=float(metric.price if metric else 100.0),
            starting_cash=100_000,
            volatility=0.03 if "crash" in message_lower else 0.02,
            agents=[],
        )
        sim = await orchestrator.start(start_req)
        tool_outputs["simulation"] = {"session_id": sim.session_id, "ticker": sim.ticker}
        await log_agent_activity(
            module="tracker",
            agent_name=str(agent.get("name") or "Tracker Associate"),
            action=f"Launched simulation scenario for {symbol}",
            status="success",
            user_id=resolved_user_id,
            details={
                "agent_id": agent_id,
                "symbol": symbol,
                "description": "Triggered simulation tool from manager chat instruction.",
                "session_id": sim.session_id,
            },
        )
    if intent == "update_agent" and isinstance(parsed.get("triggers"), dict):
        new_triggers = {**sanitize_tracker_triggers(agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {}), **sanitize_tracker_triggers(parsed["triggers"])}
        agent = tracker_repo.update_agent(
            user_id=resolved_user_id,
            agent_id=agent_id,
            updates={"triggers": new_triggers},
        ) or agent
    elif intent == "pause_agent":
        agent = tracker_repo.update_agent(user_id=resolved_user_id, agent_id=agent_id, updates={"status": "paused"}) or agent
    elif intent == "resume_agent":
        agent = tracker_repo.update_agent(user_id=resolved_user_id, agent_id=agent_id, updates={"status": "active"}) or agent

    try:
        tracker_repo.create_history(
            user_id=resolved_user_id,
            agent_id=agent_id,
            event_type="manager_instruction",
            raw_prompt=payload.message,
            parsed_intent=parsed if isinstance(parsed, dict) else {},
            trigger_snapshot=agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {},
            tool_outputs=tool_outputs,
            note=f"Manager instruction processed with intent {intent or 'conversation'}.",
            strict_persistence=True,
        )
        tracker_repo.create_history(
            user_id=resolved_user_id,
            agent_id=agent_id,
            event_type="agent_response",
            raw_prompt=None,
            parsed_intent={"intent": intent or "conversation"},
            trigger_snapshot=agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {},
            tool_outputs=tool_outputs,
            note=str(reply.get("response") or ""),
            strict_persistence=True,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    csv_export: dict[str, str] | None = None
    csv_export_error: str | None = None
    try:
        csv_export = await asyncio.to_thread(
            append_agent_response_csv,
            user_id=resolved_user_id,
            agent_id=agent_id,
            symbol=symbol,
            agent_name=str(agent.get("name") or "Tracker Associate"),
            manager_instruction=payload.message,
            response_text=str(reply.get("response") or ""),
            generated_at=str(reply.get("generated_at") or datetime.now(timezone.utc).isoformat()),
            intent=intent or "conversation",
            parsed_intent=parsed if isinstance(parsed, dict) else {},
            trigger_snapshot=agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {},
            tool_outputs=tool_outputs,
        )
    except Exception as exc:
        csv_export_error = str(exc)
        await log_agent_activity(
            module="tracker",
            agent_name=str(agent.get("name") or "Tracker Associate"),
            action=f"Failed CSV export for {symbol}",
            status="error",
            user_id=resolved_user_id,
            details={
                "agent_id": agent_id,
                "symbol": symbol,
                "description": "Could not append agent response row to tracker CSV archive.",
                "error": csv_export_error,
            },
        )

    if csv_export:
        tool_outputs["csv_export"] = csv_export

    await log_agent_activity(
        module="tracker",
        agent_name=str(agent.get("name") or "Tracker Associate"),
        action=f"Processed manager interaction for {symbol}",
        status="success",
        user_id=resolved_user_id,
        details={
            "agent_id": agent_id,
            "symbol": symbol,
            "description": "Agent read instruction, updated plan if needed, and returned a briefing.",
            "manager_message": payload.message,
            "intent": intent or "conversation",
        },
    )
    return {
        "ok": True,
        "agent": agent,
        "reply": reply,
        "parsed_intent": parsed,
        "market_state": market_state,
        "research_state": research_state,
        "tool_outputs": tool_outputs,
        "csv_export_error": csv_export_error,
    }


@router.get("/tracker/alerts")
async def list_tracker_alerts(request: Request, user_id: str | None = None, agent_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    resolved_user_id = _require_tracker_user_id(request, user_id)
    return tracker_repo.list_alerts(user_id=resolved_user_id, agent_id=agent_id, limit=min(max(limit, 1), 100))


@router.post("/tracker/emit-alert")
async def emit_tracker_alert(payload: TrackerEmitAlertRequest, request: Request) -> dict[str, Any]:
    row = tracker_repo.create_alert(
        symbol=payload.symbol,
        trigger_reason=payload.trigger_reason,
        narrative=payload.narrative,
        market_snapshot=payload.market_snapshot,
        investigation_data=payload.investigation_data,
        user_id=payload.user_id,
        agent_id=payload.agent_id,
        simulation_id=payload.simulation_id,
    )
    manager = request.app.state.ws_manager
    await manager.broadcast(
        {
            "channel": "tracker",
            "type": "new_alert",
            "data": row,
        },
        channel="tracker",
    )
    await log_agent_activity(
        module="tracker",
        agent_name=f"{payload.symbol.upper()} Tracker",
        action=f"External alert emitted for {payload.symbol.upper()}",
        status="success",
    )
    return {"ok": True, "alert": row}


@router.get("/user/profile")
async def get_profile(request: Request) -> dict[str, Any]:
    from app.services.database import get_supabase

    user_id = get_user_id_from_request(request)
    if not user_id:
        return {"user_id": None, "profile": None, "watchlist": [], "note": "Pass x-user-id header to load persisted profile."}

    client = get_supabase()
    if client is None:
        return {"user_id": user_id, "profile": None, "watchlist": [], "note": "Supabase client unavailable."}

    profile = None
    watchlist: list[str] = []
    favorites: list[str] = []
    try:
        profile = client.table("profiles").select("*").eq("id", user_id).single().execute().data
    except Exception:
        profile = None
    watchlist = get_user_watchlist(user_id)
    favorites = get_user_favorites(user_id)

    username_locked = False
    require_username_setup = True
    if isinstance(profile, dict):
        display_name = str(profile.get("display_name") or "").strip()
        email = str(profile.get("email") or "").strip()
        require_username_setup = _needs_username_setup(profile)
        username_locked = bool(display_name and (not email or display_name.lower() != email.lower()))

    return {
        "user_id": user_id,
        "profile": profile,
        "watchlist": watchlist,
        "favorites": favorites,
        "require_username_setup": require_username_setup,
        "username_locked": username_locked,
    }


class UserPrefsRequest(BaseModel):
    display_name: str | None = None
    avatar_data_url: str | None = None
    phone_number: str | None = None
    poke_enabled: bool | None = None
    tutorial_completed: bool | None = None
    watchlist: list[str] | None = None
    favorites: list[str] | None = None


class NotificationPreferencesRequest(BaseModel):
    phone_number: str | None = None
    email: str | None = None
    preferred_channel: str | None = None
    alert_frequency: str | None = None
    price_alerts: bool | None = None
    volume_alerts: bool | None = None
    simulation_summary: bool | None = None
    quiet_start: str | None = None
    quiet_end: str | None = None


def _notification_defaults(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = profile or {}
    return {
        "phone_number": str(profile.get("phone_number") or "").strip() or None,
        "email": str(profile.get("email") or "").strip() or None,
        "preferred_channel": "push",
        "alert_frequency": "realtime",
        "price_alerts": True,
        "volume_alerts": True,
        "simulation_summary": True,
        "quiet_start": "22:00:00",
        "quiet_end": "07:00:00",
    }


def _sanitize_notification_preferences(raw: dict[str, Any], *, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = existing or {}
    merged = {**_notification_defaults(existing), **existing, **raw}

    phone_raw = str(merged.get("phone_number") or "").strip()
    if phone_raw:
        digits = "".join(ch for ch in phone_raw if ch.isdigit())
        if len(digits) < 10:
            raise HTTPException(status_code=422, detail="Phone number must include at least 10 digits.")
        merged["phone_number"] = phone_raw[:40]
    else:
        merged["phone_number"] = None

    email_raw = str(merged.get("email") or "").strip()
    if email_raw and "@" not in email_raw:
        raise HTTPException(status_code=422, detail="Email must be valid.")
    merged["email"] = email_raw[:320] if email_raw else None

    channel = str(merged.get("preferred_channel") or "push").strip().lower()
    if channel not in {"sms", "email", "push"}:
        raise HTTPException(status_code=422, detail="preferred_channel must be one of: sms, email, push.")
    merged["preferred_channel"] = channel

    frequency = str(merged.get("alert_frequency") or "realtime").strip().lower()
    if frequency not in {"realtime", "hourly", "daily"}:
        raise HTTPException(status_code=422, detail="alert_frequency must be one of: realtime, hourly, daily.")
    merged["alert_frequency"] = frequency

    merged["price_alerts"] = bool(merged.get("price_alerts", True))
    merged["volume_alerts"] = bool(merged.get("volume_alerts", True))
    merged["simulation_summary"] = bool(merged.get("simulation_summary", True))

    def _clock(label: str, fallback: str) -> str:
        token = str(merged.get(label) or fallback).strip()
        match = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", token)
        if not match:
            raise HTTPException(status_code=422, detail=f"{label} must be HH:MM or HH:MM:SS.")
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3) or "0")
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
            raise HTTPException(status_code=422, detail=f"{label} is outside valid clock range.")
        return f"{hour:02d}:{minute:02d}:{second:02d}"

    merged["quiet_start"] = _clock("quiet_start", "22:00:00")
    merged["quiet_end"] = _clock("quiet_end", "07:00:00")
    return {
        "phone_number": merged["phone_number"],
        "email": merged["email"],
        "preferred_channel": merged["preferred_channel"],
        "alert_frequency": merged["alert_frequency"],
        "price_alerts": merged["price_alerts"],
        "volume_alerts": merged["volume_alerts"],
        "simulation_summary": merged["simulation_summary"],
        "quiet_start": merged["quiet_start"],
        "quiet_end": merged["quiet_end"],
    }


def _needs_username_setup(profile: dict[str, Any] | None) -> bool:
    if not isinstance(profile, dict):
        return True
    display_name = str(profile.get("display_name") or "").strip()
    email = str(profile.get("email") or "").strip()
    if not display_name:
        return True
    return bool(email and display_name.lower() == email.lower())


_AVATAR_DATA_URL_RE = re.compile(r"^data:image/([A-Za-z0-9.+-]+);base64,(.+)$", re.DOTALL)


def _decode_avatar_data_url(data_url: str) -> tuple[bytes, str]:
    match = _AVATAR_DATA_URL_RE.match(data_url.strip())
    if not match:
        raise HTTPException(status_code=422, detail="Avatar payload must be a base64 image data URL.")
    ext = match.group(1).lower()
    if ext == "jpeg":
        ext = "jpg"
    if ext not in {"jpg", "png", "webp", "gif"}:
        raise HTTPException(status_code=422, detail="Unsupported avatar image format.")
    encoded = match.group(2).strip()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=422, detail="Avatar image payload is invalid.") from exc
    if len(raw) > 1_500_000:
        raise HTTPException(status_code=422, detail="Avatar image is too large.")
    return raw, ext


def _ensure_avatar_bucket(client: Any, bucket: str) -> None:
    try:
        existing = client.storage.list_buckets()
        for item in existing or []:
            item_id = str(getattr(item, "id", "") or "")
            if item_id == bucket:
                return
    except Exception:
        # Continue and attempt create directly.
        pass
    try:
        client.storage.create_bucket(
            bucket,
            options={"public": True, "file_size_limit": 2_000_000, "allowed_mime_types": ["image/jpeg", "image/png", "image/webp", "image/gif"]},
        )
    except Exception:
        # Bucket may already exist or policy may block create.
        pass


def _upload_avatar_to_supabase(client: Any, bucket: str, user_id: str, data_url: str) -> str:
    raw, ext = _decode_avatar_data_url(data_url)
    mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
    _ensure_avatar_bucket(client, bucket)
    path = f"{user_id}/avatar-{uuid4().hex}.{ext}"
    try:
        client.storage.from_(bucket).upload(
            path,
            raw,
            {"content-type": mime, "upsert": "true"},
        )
        return str(client.storage.from_(bucket).get_public_url(path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not upload avatar image to Supabase storage.") from exc


@router.patch("/user/preferences")
async def patch_preferences(payload: UserPrefsRequest, request: Request) -> dict[str, Any]:
    from app.services.database import get_supabase

    user_id = get_user_id_from_request(request)
    if not user_id:
        return {"ok": False, "error": "missing_user_id"}

    client = get_supabase()
    if client is None:
        return {"ok": False, "error": "supabase_unavailable"}
    settings = request.app.state.settings

    updates = payload.model_dump(exclude_none=True)
    watchlist = updates.pop("watchlist", None)
    favorites = updates.pop("favorites", None)
    avatar_data_url = updates.pop("avatar_data_url", None)
    requested_display_name = updates.get("display_name")

    existing_profile: dict[str, Any] | None = None
    try:
        existing_profile = client.table("profiles").select("id,email,display_name,avatar_url").eq("id", user_id).single().execute().data
    except Exception:
        existing_profile = None

    if isinstance(requested_display_name, str):
        normalized_display_name = requested_display_name.strip()
        if not normalized_display_name:
            raise HTTPException(status_code=422, detail="Username cannot be empty.")
        if len(normalized_display_name) < 3:
            raise HTTPException(status_code=422, detail="Username must be at least 3 characters.")
        if len(normalized_display_name) > 24:
            raise HTTPException(status_code=422, detail="Username must be 24 characters or fewer.")
        existing_display_name = str((existing_profile or {}).get("display_name") or "").strip()
        existing_email = str((existing_profile or {}).get("email") or "").strip()
        username_locked = bool(
            existing_display_name
            and (not existing_email or existing_display_name.lower() != existing_email.lower())
        )
        if username_locked and normalized_display_name != existing_display_name:
            raise HTTPException(status_code=409, detail="Username is locked and cannot be changed.")
        updates["display_name"] = normalized_display_name

    requested_phone = updates.get("phone_number")
    if isinstance(requested_phone, str):
        normalized_phone = requested_phone.strip()
        if normalized_phone:
            digits = "".join(ch for ch in normalized_phone if ch.isdigit())
            if len(digits) < 10:
                raise HTTPException(status_code=422, detail="Phone number must include at least 10 digits.")
            updates["phone_number"] = normalized_phone[:40]
        else:
            updates["phone_number"] = None

    if isinstance(avatar_data_url, str):
        normalized_avatar = avatar_data_url.strip()
        if normalized_avatar:
            if not normalized_avatar.startswith("data:image/"):
                raise HTTPException(status_code=422, detail="Avatar payload must be an image data URL.")
            if len(normalized_avatar) > 2_000_000:
                raise HTTPException(status_code=422, detail="Avatar image is too large.")
            updates["avatar_url"] = _upload_avatar_to_supabase(
                client,
                settings.supabase_avatar_bucket,
                user_id,
                normalized_avatar,
            )

    profile = None
    if updates:
        try:
            updated_rows = (
                client.table("profiles")
                .update(updates)
                .eq("id", user_id)
                .execute()
                .data
                or []
            )
            if not updated_rows:
                profile = (
                    client.table("profiles")
                    .insert({"id": user_id, **updates})
                    .execute()
                    .data
                )
            else:
                profile = updated_rows
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Could not save profile preferences.") from exc

    if isinstance(watchlist, list):
        set_user_watchlist(user_id, watchlist)
    if isinstance(favorites, list):
        set_user_favorites(user_id, favorites)

    final_profile = None
    try:
        final_profile = client.table("profiles").select("*").eq("id", user_id).single().execute().data
    except Exception:
        final_profile = None
    return {
        "ok": True,
        "profile": final_profile or profile,
        "require_username_setup": _needs_username_setup(final_profile if isinstance(final_profile, dict) else existing_profile),
    }


@router.get("/user/notification-preferences")
async def get_notification_preferences(request: Request) -> dict[str, Any]:
    from app.services.database import get_supabase

    user_id = get_user_id_from_request(request)
    if not user_id:
        return {"user_id": None, "preferences": None, "note": "Sign in to load notification preferences."}

    client = get_supabase()
    if client is None:
        return {"user_id": user_id, "preferences": None, "note": "Supabase client unavailable."}

    profile: dict[str, Any] | None = None
    try:
        profile = client.table("profiles").select("email,phone_number").eq("id", user_id).single().execute().data
    except Exception:
        profile = None

    row: dict[str, Any] | None = None
    try:
        row = (
            client.table("notification_preferences")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
            .data
        )
    except Exception:
        row = None

    defaults = _notification_defaults(profile if isinstance(profile, dict) else None)
    if isinstance(row, dict):
        preferences = {
            **defaults,
            **row,
            "quiet_start": str(row.get("quiet_start") or defaults["quiet_start"]),
            "quiet_end": str(row.get("quiet_end") or defaults["quiet_end"]),
        }
        preferences.pop("user_id", None)
        preferences.pop("created_at", None)
        preferences.pop("updated_at", None)
    else:
        preferences = defaults

    return {"user_id": user_id, "preferences": preferences}


@router.put("/user/notification-preferences")
async def put_notification_preferences(payload: NotificationPreferencesRequest, request: Request) -> dict[str, Any]:
    from app.services.database import get_supabase

    user_id = get_user_id_from_request(request)
    if not user_id:
        return {"ok": False, "error": "missing_user_id"}

    client = get_supabase()
    if client is None:
        return {"ok": False, "error": "supabase_unavailable"}

    existing: dict[str, Any] | None = None
    try:
        existing = (
            client.table("notification_preferences")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
            .data
        )
    except Exception:
        existing = None

    clean = _sanitize_notification_preferences(
        payload.model_dump(exclude_none=True),
        existing=existing if isinstance(existing, dict) else None,
    )
    upsert_payload = {
        "user_id": user_id,
        **clean,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        saved = (
            client.table("notification_preferences")
            .update(upsert_payload)
            .eq("user_id", user_id)
            .execute()
            .data
            or []
        )
        if not saved:
            saved = (
                client.table("notification_preferences")
                .insert(upsert_payload)
                .execute()
                .data
                or []
            )
        preferences = (saved[0] if saved else upsert_payload) or upsert_payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not save notification preferences.") from exc

    # Keep profile phone/email in sync for existing notification routing fallbacks.
    sync_updates: dict[str, Any] = {}
    if "phone_number" in clean:
        sync_updates["phone_number"] = clean["phone_number"]
    if "email" in clean and clean["email"]:
        sync_updates["email"] = clean["email"]
    if sync_updates:
        try:
            client.table("profiles").update(sync_updates).eq("id", user_id).execute()
        except Exception:
            pass

    preferences.pop("user_id", None)
    preferences.pop("created_at", None)
    preferences.pop("updated_at", None)
    preferences["quiet_start"] = str(preferences.get("quiet_start") or clean["quiet_start"])
    preferences["quiet_end"] = str(preferences.get("quiet_end") or clean["quiet_end"])
    return {"ok": True, "preferences": preferences}


class FavoriteStocksRequest(BaseModel):
    symbols: list[str]


@router.get("/user/favorites")
async def get_favorite_stocks(request: Request) -> dict[str, Any]:
    user_id = get_user_id_from_request(request)
    if not user_id:
        return {"user_id": None, "favorites": [], "note": "Sign in to persist favorites."}
    return {"user_id": user_id, "favorites": get_user_favorites(user_id)}


@router.put("/user/favorites")
async def put_favorite_stocks(payload: FavoriteStocksRequest, request: Request) -> dict[str, Any]:
    user_id = get_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    favorites = set_user_favorites(user_id, payload.symbols)
    return {"user_id": user_id, "favorites": favorites}


@router.get("/user/trades")
async def get_user_trades(request: Request, limit: int = 200) -> dict[str, Any]:
    from app.services.database import get_supabase

    user_id = get_user_id_from_request(request)
    if not user_id:
        return {"user_id": None, "trades": [], "note": "Pass x-user-id header to load persisted trades."}

    client = get_supabase()
    if client is None:
        return {"user_id": user_id, "trades": [], "note": "Supabase client unavailable."}

    trades: list[dict[str, Any]] = []
    try:
        rows = (
            client.table("simulations")
            .select("id,results,created_at,completed_at,status")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
            .data
            or []
        )
        for row in rows:
            results = row.get("results") or {}
            for trade in (results.get("trades") or []):
                if isinstance(trade, dict):
                    trades.append({"simulation_id": row.get("id"), **trade})
        trades = trades[: max(1, min(limit, 1000))]
    except Exception:
        trades = []

    return {"user_id": user_id, "trades": trades}


@router.get("/agents/activity")
async def agents_activity(limit: int = 50, module: str | None = None) -> dict[str, Any]:
    data = await get_recent_activity(limit=min(max(limit, 1), 200), module=module)
    return {"items": data}


@router.websocket("/agents/ws")
async def agents_activity_ws(websocket: WebSocket):
    manager = websocket.app.state.ws_manager
    await manager.connect(websocket, channels={"agents"})
    try:
        while True:
            raw = await websocket.receive_text()
            if raw.lower().strip() in {"ping", "heartbeat"}:
                await websocket.send_json({"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()})
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)


@router.post("/poke/inbound")
async def poke_inbound(payload: PokeInboundRequest, request: Request) -> dict[str, Any]:
    message = payload.message.strip()
    lowered = message.lower()
    settings = request.app.state.settings

    if lowered.startswith("research "):
        symbol = message.split(maxsplit=1)[1].strip().upper()
        research = await run_research(ResearchRequest(ticker=symbol, timeframe="7d", include_prediction_markets=True), settings)
        await log_agent_activity(module="research", agent_name="Poke Inbound", action=f"Ran research command for {symbol}")
        return {"ok": True, "command": "research", "symbol": symbol, "summary": research.narratives[:2]}

    if lowered.startswith("simulate "):
        await log_agent_activity(module="simulation", agent_name="Poke Inbound", action="Requested simulation launch", status="pending")
        return {"ok": True, "command": "simulate", "message": "Simulation command accepted."}

    if lowered.startswith("track "):
        await log_agent_activity(module="tracker", agent_name="Poke Inbound", action="Requested tracker deployment", status="pending")
        return {"ok": True, "command": "track", "message": "Tracker command accepted."}

    if lowered == "status":
        activity = await get_recent_activity(limit=5)
        return {"ok": True, "command": "status", "recent_activity": activity}

    return {"ok": False, "message": "Unsupported command. Try: research NVDA, simulate crash for TSLA, track AMZN with 3% alert, status"}
