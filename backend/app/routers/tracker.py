from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import AlertConfig, TrackerWatchlistRequest

router = APIRouter(prefix="/tracker", tags=["tracker"])


@router.get("/snapshot")
async def snapshot(request: Request):
    tracker = request.app.state.tracker
    return await tracker.snapshot()


@router.post("/watchlist")
async def set_watchlist(payload: TrackerWatchlistRequest, request: Request):
    tracker = request.app.state.tracker
    return {"watchlist": tracker.set_watchlist(payload.tickers)}


@router.get("/watchlist")
async def get_watchlist(request: Request):
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
