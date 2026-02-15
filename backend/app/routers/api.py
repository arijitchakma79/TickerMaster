from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.schemas import ResearchRequest
from app.services.agent_logger import get_recent_activity, log_agent_activity
from app.services.browserbase_scraper import run_deep_research
from app.services.llm import parse_tracker_instruction, tracker_agent_chat_response
from app.services.macro import get_macro_indicators
from app.services.market_data import fetch_candles, fetch_metric, resolve_symbol_input
from app.services.prediction_markets import fetch_kalshi_markets, fetch_polymarket_markets
from app.services.research_cache import get_cached_research, set_cached_research
from app.services.sentiment import get_x_sentiment, run_research
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
    symbol: str
    name: str
    triggers: dict[str, Any] = Field(default_factory=dict)
    auto_simulate: bool = False


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

    if "research_timeframe" in raw and isinstance(raw.get("research_timeframe"), str):
        out["research_timeframe"] = normalize_timeframe(str(raw["research_timeframe"]))
    return out


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


@router.post("/tracker/agents")
async def create_tracker_agent(payload: TrackerAgentCreateRequest, request: Request, user_id: str | None = None) -> dict[str, Any]:
    resolved_user_id = user_id or get_user_id_from_request(request)
    # Prioritize authenticated/explicit user id for FK-safe persistence.
    # Query param fallback retained for local testing.
    agent = tracker_repo.create_agent(
        user_id=resolved_user_id,
        symbol=payload.symbol,
        name=payload.name,
        triggers=sanitize_tracker_triggers(payload.triggers),
        auto_simulate=payload.auto_simulate,
    )
    await log_agent_activity(
        module="tracker",
        agent_name=str(agent.get("name") or f"{payload.symbol.upper()} Tracker"),
        action=f"Tracker agent deployed for {str(agent.get('symbol') or payload.symbol).upper()}",
        status="success",
        user_id=resolved_user_id,
        details={
            "agent_id": agent.get("id"),
            "symbol": str(agent.get("symbol") or payload.symbol).upper(),
            "description": "Agent is actively monitoring configured trigger thresholds.",
            "triggers": agent.get("triggers") or payload.triggers,
        },
    )
    return agent


@router.get("/tracker/agents")
async def list_tracker_agents(request: Request, user_id: str | None = None) -> list[dict[str, Any]]:
    resolved_user_id = user_id or get_user_id_from_request(request)
    return tracker_repo.list_agents(user_id=resolved_user_id)


@router.patch("/tracker/agents/{agent_id}")
async def patch_tracker_agent(agent_id: str, payload: TrackerAgentPatchRequest, request: Request, user_id: str | None = None) -> dict[str, Any]:
    resolved_user_id = user_id or get_user_id_from_request(request)
    updates = payload.model_dump(exclude_none=True)
    if "triggers" in updates and isinstance(updates.get("triggers"), dict):
        updates["triggers"] = sanitize_tracker_triggers(updates["triggers"])
    item = tracker_repo.update_agent(user_id=resolved_user_id, agent_id=agent_id, updates=updates)
    if item is None:
        return {"error": "not_found"}
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
    resolved_user_id = user_id or get_user_id_from_request(request)
    item = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return item


@router.get("/tracker/agents/{agent_id}/detail")
async def get_tracker_agent_detail(agent_id: str, request: Request, user_id: str | None = None) -> dict[str, Any]:
    resolved_user_id = user_id or get_user_id_from_request(request)
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
        "recent_actions": actions,
    }


@router.delete("/tracker/agents/{agent_id}")
async def delete_tracker_agent(agent_id: str, request: Request, user_id: str | None = None) -> dict[str, Any]:
    resolved_user_id = user_id or get_user_id_from_request(request)
    existing = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    ok = tracker_repo.delete_agent(user_id=resolved_user_id, agent_id=agent_id)
    if ok:
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
    resolved_user_id = payload.user_id or get_user_id_from_request(request)
    settings = request.app.state.settings
    parsed = await parse_tracker_instruction(settings, payload.prompt)

    symbol = str(parsed.get("symbol") or "AAPL").upper().strip()
    name = str(parsed.get("name") or f"{symbol} Associate")
    triggers = sanitize_tracker_triggers(parsed.get("triggers") if isinstance(parsed.get("triggers"), dict) else {})
    auto_simulate = bool(parsed.get("auto_simulate", False))
    agent = tracker_repo.create_agent(
        user_id=resolved_user_id,
        symbol=symbol,
        name=name,
        triggers=triggers,
        auto_simulate=auto_simulate,
    )
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
    resolved_user_id = payload.user_id or get_user_id_from_request(request)
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
    reply = await tracker_agent_chat_response(settings, agent, market_state, research_state, payload.message)
    parsed = await parse_tracker_instruction(settings, payload.message)
    intent = str(parsed.get("intent") or "")
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
    }


@router.get("/tracker/alerts")
async def list_tracker_alerts(request: Request, user_id: str | None = None, agent_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    resolved_user_id = user_id or get_user_id_from_request(request)
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
    poke_enabled: bool | None = None
    tutorial_completed: bool | None = None
    watchlist: list[str] | None = None
    favorites: list[str] | None = None


def _needs_username_setup(profile: dict[str, Any] | None) -> bool:
    if not isinstance(profile, dict):
        return True
    display_name = str(profile.get("display_name") or "").strip()
    email = str(profile.get("email") or "").strip()
    if not display_name:
        return True
    return bool(email and display_name.lower() == email.lower())


@router.patch("/user/preferences")
async def patch_preferences(payload: UserPrefsRequest, request: Request) -> dict[str, Any]:
    from app.services.database import get_supabase

    user_id = get_user_id_from_request(request)
    if not user_id:
        return {"ok": False, "error": "missing_user_id"}

    client = get_supabase()
    if client is None:
        return {"ok": False, "error": "supabase_unavailable"}

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

    if isinstance(avatar_data_url, str):
        normalized_avatar = avatar_data_url.strip()
        if normalized_avatar:
            if not normalized_avatar.startswith("data:image/"):
                raise HTTPException(status_code=422, detail="Avatar payload must be an image data URL.")
            if len(normalized_avatar) > 2_000_000:
                raise HTTPException(status_code=422, detail="Avatar image is too large.")
            updates["avatar_url"] = normalized_avatar

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
