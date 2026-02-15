from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import api, chat, research, simulation, system, tracker
from app.services.activity_stream import set_ws_manager
from app.services.mcp_tool_router import shutdown_tracker_mcp_router
from app.services.simulation import SimulationOrchestrator
from app.services.tracker import TrackerService
from app.services.tracker_csv import ensure_tracker_storage_buckets
from app.ws_manager import WSManager

logger = logging.getLogger(__name__)

# Avoid noisy WinError 10054 callback traces from Proactor transport shutdown
# when local clients disconnect abruptly after successful responses.
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ws_manager = WSManager()
    orchestrator = SimulationOrchestrator(settings, ws_manager)
    tracker_service = TrackerService(settings, ws_manager, orchestrator=orchestrator)

    app.state.settings = settings
    app.state.ws_manager = ws_manager
    app.state.orchestrator = orchestrator
    app.state.tracker = tracker_service
    set_ws_manager(ws_manager)
    try:
        buckets_ready = await asyncio.to_thread(ensure_tracker_storage_buckets)
        if not buckets_ready:
            logger.warning("One or more tracker storage buckets are not ready on startup.")
    except Exception:
        logger.exception("Failed to ensure Supabase tracker storage buckets on startup.")

    try:
        await tracker_service.start()
    except Exception:
        logger.exception("Failed to start tracker service")

    try:
        yield
    finally:
        try:
            await tracker_service.stop()
        except Exception:
            logger.exception("Failed to stop tracker service")

        session_ids: list[Any]
        try:
            session_ids = list(orchestrator.sessions.keys())  # type: ignore[union-attr]
        except Exception:
            session_ids = list(orchestrator.sessions)  # type: ignore[arg-type]

        for session_id in session_ids:
            try:
                await orchestrator.stop(session_id)
            except Exception:
                logger.exception("Failed to stop simulation session %s", session_id)

        try:
            await shutdown_tracker_mcp_router()
        except Exception:
            logger.exception("Failed to shutdown tracker MCP router")


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
