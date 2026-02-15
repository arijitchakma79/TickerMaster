from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.schemas import AlertConfig, TrackerWatchlistRequest
from app.services.market_data import fetch_watchlist_metrics
from app.services.tracker_repository import tracker_repo
from app.services.user_context import get_user_id_from_request
from app.services.user_preferences import get_watchlist as get_persisted_watchlist
from app.services.user_preferences import set_watchlist as set_persisted_watchlist

router = APIRouter(prefix="/tracker", tags=["tracker"])
_SYMBOL_PATTERN = r"^[A-Za-z0-9.\-]{1,12}$"


def _require_user_id(request: Request, explicit_user_id: str | None = None) -> str:
    resolved = explicit_user_id or get_user_id_from_request(request)
    if not resolved:
        raise HTTPException(status_code=401, detail="Authentication required")
    return resolved


class TrackerAgentCreateRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12, pattern=_SYMBOL_PATTERN)
    name: str = Field(min_length=1, max_length=120)
    triggers: dict[str, Any] = Field(default_factory=dict)
    auto_simulate: bool = False


class TrackerAgentPatchRequest(BaseModel):
    status: str | None = None
    name: str | None = None
    triggers: dict[str, Any] | None = None
    auto_simulate: bool | None = None


class TrackerEmitAlertRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12, pattern=_SYMBOL_PATTERN)
    trigger_reason: str = Field(min_length=1, max_length=400)
    narrative: str | None = None
    market_snapshot: dict[str, Any] = Field(default_factory=dict)
    investigation_data: dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    agent_id: str | None = None


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


def _sanitize_tracker_triggers(raw: dict[str, Any] | None) -> dict[str, Any]:
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
        out["research_timeframe"] = _normalize_timeframe(str(raw["research_timeframe"]))
    return out


@router.get("/snapshot")
async def snapshot(request: Request, tickers: str | None = None):
    explicit: list[str] = []
    if isinstance(tickers, str) and tickers.strip():
        seen: set[str] = set()
        for raw in tickers.split(","):
            symbol = raw.strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            explicit.append(symbol)
        explicit = explicit[:120]

    user_id = get_user_id_from_request(request)
    if explicit:
        metrics = await asyncio.to_thread(fetch_watchlist_metrics, explicit)
        alerts = tracker_repo.list_alerts(user_id=user_id, limit=8) if user_id else []
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tickers": [item.model_dump() for item in metrics],
            "alerts_triggered": alerts,
        }
    if user_id:
        watchlist = get_persisted_watchlist(user_id)
        metrics = await asyncio.to_thread(fetch_watchlist_metrics, watchlist) if watchlist else []
        alerts = tracker_repo.list_alerts(user_id=user_id, limit=8)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tickers": [item.model_dump() for item in metrics],
            "alerts_triggered": alerts,
        }
    tracker = request.app.state.tracker
    return await tracker.snapshot()


@router.post("/watchlist")
async def set_watchlist(payload: TrackerWatchlistRequest, request: Request):
    user_id = get_user_id_from_request(request)
    if user_id:
        return {"watchlist": set_persisted_watchlist(user_id, payload.tickers)}
    tracker = request.app.state.tracker
    return {"watchlist": tracker.set_watchlist(payload.tickers)}


@router.get("/watchlist")
async def get_watchlist(request: Request):
    user_id = get_user_id_from_request(request)
    if user_id:
        watchlist = get_persisted_watchlist(user_id)
        return {"watchlist": watchlist}
    tracker = request.app.state.tracker
    return {"watchlist": tracker.list_watchlist()}


@router.post("/alerts")
async def add_alert(payload: AlertConfig, request: Request):
    tracker = request.app.state.tracker
    tracker.add_alert(payload)
    return {"alerts": [alert.model_dump() for alert in tracker.list_alerts()]}


@router.get("/alerts")
async def get_alerts(request: Request):
    tracker = request.app.state.tracker
    return {"alerts": [alert.model_dump() for alert in tracker.list_alerts()]}


@router.post("/poll")
async def poll_now(request: Request):
    tracker = request.app.state.tracker
    return await tracker.poll_once()


@router.post("/agents")
async def create_tracker_agent(payload: TrackerAgentCreateRequest, request: Request, user_id: str | None = None):
    resolved_user_id = _require_user_id(request, explicit_user_id=user_id)
    return tracker_repo.create_agent(
        user_id=resolved_user_id,
        symbol=payload.symbol,
        name=payload.name,
        triggers=_sanitize_tracker_triggers(payload.triggers),
        auto_simulate=payload.auto_simulate,
    )


@router.get("/agents")
async def list_tracker_agents(request: Request, user_id: str | None = None):
    resolved_user_id = _require_user_id(request, explicit_user_id=user_id)
    return tracker_repo.list_agents(user_id=resolved_user_id)


@router.patch("/agents/{agent_id}")
async def patch_tracker_agent(agent_id: str, payload: TrackerAgentPatchRequest, request: Request, user_id: str | None = None):
    resolved_user_id = _require_user_id(request, explicit_user_id=user_id)
    existing = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    if existing is None:
        return {"error": "not_found"}
    updates = payload.model_dump(exclude_none=True)
    if "triggers" in updates and isinstance(updates.get("triggers"), dict):
        base_triggers = existing.get("triggers") if isinstance(existing.get("triggers"), dict) else {}
        merged_triggers = {**base_triggers, **updates["triggers"]}
        updates["triggers"] = _sanitize_tracker_triggers(merged_triggers)
    item = tracker_repo.update_agent(user_id=resolved_user_id, agent_id=agent_id, updates=updates)
    return item or {"error": "not_found"}


@router.delete("/agents/{agent_id}")
async def delete_tracker_agent(agent_id: str, request: Request, user_id: str | None = None):
    resolved_user_id = _require_user_id(request, explicit_user_id=user_id)
    return {"ok": tracker_repo.delete_agent(user_id=resolved_user_id, agent_id=agent_id)}


@router.post("/emit-alert")
async def emit_alert(payload: TrackerEmitAlertRequest, request: Request):
    resolved_user_id = _require_user_id(request, explicit_user_id=payload.user_id)
    row = tracker_repo.create_alert(
        symbol=payload.symbol,
        trigger_reason=payload.trigger_reason,
        narrative=payload.narrative,
        market_snapshot=payload.market_snapshot,
        investigation_data=payload.investigation_data,
        user_id=resolved_user_id,
        agent_id=payload.agent_id,
    )
    manager = request.app.state.ws_manager
    await manager.broadcast({"channel": "tracker", "type": "new_alert", "data": row}, channel="tracker")
    return {"ok": True, "alert": row}
