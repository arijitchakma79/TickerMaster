from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.routers.api import sanitize_tracker_triggers
from app.schemas import AlertConfig, TrackerWatchlistRequest
from app.services.market_data import fetch_watchlist_metrics
from app.services.tracker_repository import tracker_repo
from app.services.user_context import get_user_id_from_request
from app.services.user_preferences import get_watchlist as get_persisted_watchlist
from app.services.user_preferences import set_watchlist as set_persisted_watchlist

router = APIRouter(prefix="/tracker", tags=["tracker"])


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


@router.get("/snapshot")
async def snapshot(request: Request):
    user_id = get_user_id_from_request(request)
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
    resolved_user_id = user_id or get_user_id_from_request(request)
    clean_triggers = sanitize_tracker_triggers(payload.triggers)
    if not str(clean_triggers.get("start_at") or "").strip():
        raise HTTPException(status_code=422, detail="start_at is required.")
    return tracker_repo.create_agent(
        user_id=resolved_user_id,
        symbol=payload.symbol,
        name=payload.name,
        triggers=clean_triggers,
        auto_simulate=payload.auto_simulate,
    )


@router.get("/agents")
async def list_tracker_agents(request: Request, user_id: str | None = None):
    resolved_user_id = user_id or get_user_id_from_request(request)
    return tracker_repo.list_agents(user_id=resolved_user_id)


@router.patch("/agents/{agent_id}")
async def patch_tracker_agent(agent_id: str, payload: TrackerAgentPatchRequest, request: Request, user_id: str | None = None):
    resolved_user_id = user_id or get_user_id_from_request(request)
    existing = tracker_repo.get_agent(user_id=resolved_user_id, agent_id=agent_id)
    if existing is None:
        return {"error": "not_found"}
    updates = payload.model_dump(exclude_none=True)
    if "triggers" in updates and isinstance(updates["triggers"], dict):
        existing_triggers = sanitize_tracker_triggers(existing.get("triggers")) if isinstance(existing.get("triggers"), dict) else {}
        merged_triggers = {**existing_triggers, **sanitize_tracker_triggers(updates["triggers"])}
        if not str(merged_triggers.get("start_at") or "").strip():
            raise HTTPException(status_code=422, detail="start_at is required.")
        updates["triggers"] = merged_triggers
    item = tracker_repo.update_agent(user_id=resolved_user_id, agent_id=agent_id, updates=updates)
    return item or {"error": "not_found"}


@router.delete("/agents/{agent_id}")
async def delete_tracker_agent(agent_id: str, request: Request, user_id: str | None = None):
    resolved_user_id = user_id or get_user_id_from_request(request)
    return {"ok": tracker_repo.delete_agent(user_id=resolved_user_id, agent_id=agent_id)}


@router.post("/emit-alert")
async def emit_alert(payload: TrackerEmitAlertRequest, request: Request):
    row = tracker_repo.create_alert(
        symbol=payload.symbol,
        trigger_reason=payload.trigger_reason,
        narrative=payload.narrative,
        market_snapshot=payload.market_snapshot,
        investigation_data=payload.investigation_data,
        user_id=payload.user_id,
        agent_id=payload.agent_id,
    )
    manager = request.app.state.ws_manager
    await manager.broadcast({"channel": "tracker", "type": "new_alert", "data": row}, channel="tracker")
    return {"ok": True, "alert": row}
