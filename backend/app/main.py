from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import chat, research, simulation, system, tracker
from app.services.simulation import SimulationOrchestrator
from app.services.tracker import TrackerService
from app.ws_manager import WSManager


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

    await tracker_service.start()

    yield

    for session in list(orchestrator.sessions):
        await orchestrator.stop(session)


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(research.router)
app.include_router(simulation.router)
app.include_router(tracker.router)
app.include_router(chat.router)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    channels_param = websocket.query_params.get("channels", "global,simulation,tracker")
    channels = {channel.strip() for channel in channels_param.split(",") if channel.strip()}

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
        await manager.disconnect(websocket)


@app.get("/")
async def root():
    return {"app": settings.app_name, "status": "running"}
