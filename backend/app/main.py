from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import api, chat, research, simulation, system, tracker
from app.services.activity_stream import set_ws_manager
from app.services.simulation import SimulationOrchestrator
from app.services.tracker import TrackerService
from app.ws_manager import WSManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ws_manager = WSManager()
    orchestrator = SimulationOrchestrator(settings, ws_manager)
    tracker_service = TrackerService(settings, ws_manager)

    app.state.settings = settings
    app.state.ws_manager = ws_manager
    app.state.orchestrator = orchestrator
    app.state.tracker = tracker_service
    set_ws_manager(ws_manager)

    try:
        await tracker_service.start()
    except Exception:
        logger.exception("Failed to start tracker service")

    try:
        yield
    finally:
        for session_id in list(orchestrator.sessions.keys()):
            try:
                await orchestrator.stop(session_id)
            except Exception:
                logger.exception("Failed to stop simulation session %s", session_id)


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-User-Id"],
)

app.include_router(system.router)
app.include_router(api.router)
app.include_router(research.router)
app.include_router(simulation.router)
app.include_router(tracker.router)
app.include_router(chat.router)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    allowed_channels = {"global", "simulation", "tracker", "agents"}
    channels_param = websocket.query_params.get("channels", "global,simulation,tracker")
    requested_channels = {channel.strip() for channel in channels_param.split(",") if channel.strip()}
    channels = {channel for channel in requested_channels if channel in allowed_channels} or {"global"}

    manager: WSManager = websocket.app.state.ws_manager
    await manager.connect(websocket, channels=channels)

    await manager.broadcast(
        {
            "channel": "global",
            "type": "socket_join",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "channels": sorted(channels),
        }
    )

    try:
        while True:
            raw = await websocket.receive_text()
            if raw.lower().strip() in {"ping", "heartbeat"}:
                await websocket.send_json({"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                await manager.broadcast(
                    {
                        "channel": "global",
                        "type": "user_event",
                        "payload": raw,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        logger.exception("Unhandled websocket stream error")
        await manager.disconnect(websocket)


@app.get("/")
async def root():
    return {"app": settings.app_name, "status": "running"}
