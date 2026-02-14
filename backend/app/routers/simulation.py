from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.schemas import SimulationStartRequest
from app.services.modal_integration import modal_cron_health, spin_modal_sandbox

router = APIRouter(prefix="/simulation", tags=["simulation"])


class SandboxRequest(BaseModel):
    prompt: str
    session_id: str


@router.post("/start")
async def start_simulation(payload: SimulationStartRequest, request: Request):
    orchestrator = request.app.state.orchestrator
    return await orchestrator.start(payload)


@router.post("/stop/{session_id}")
async def stop_simulation(session_id: str, request: Request):
    orchestrator = request.app.state.orchestrator
    stopped = await orchestrator.stop(session_id)
    if not stopped:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True, "session_id": session_id}


@router.get("/state/{session_id}")
async def simulation_state(session_id: str, request: Request):
    orchestrator = request.app.state.orchestrator
    state = orchestrator.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@router.get("/sessions")
async def list_simulation_sessions(request: Request):
    orchestrator = request.app.state.orchestrator
    return {"sessions": orchestrator.list()}


@router.post("/modal/sandbox")
async def modal_sandbox(payload: SandboxRequest, request: Request):
    settings = request.app.state.settings
    return await spin_modal_sandbox(settings, payload.prompt, payload.session_id)


@router.get("/modal/cron-health")
async def modal_status(request: Request):
    settings = request.app.state.settings
    return await modal_cron_health(settings)
