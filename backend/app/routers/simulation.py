from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.schemas import SimulationStartRequest
from app.services.simulation_store import attach_modal_sandbox
from app.services.modal_integration import modal_cron_health, spin_modal_sandbox
from app.services.simulation_agents import delete_simulation_agent, list_simulation_agents, set_simulation_agents
from app.services.user_context import get_user_id_from_request

router = APIRouter(prefix="/simulation", tags=["simulation"])


class SandboxRequest(BaseModel):
    prompt: str
    session_id: str


class SimulationAgentEditor(BaseModel):
    risk: int = Field(..., ge=0, le=100)
    tempo: int = Field(..., ge=0, le=100)
    style: int = Field(..., ge=0, le=100)
    news: int = Field(..., ge=0, le=100)


class SimulationAgentEntry(BaseModel):
    config: dict
    iconEmoji: str | None = None
    editor: SimulationAgentEditor | None = None


class SimulationAgentsRequest(BaseModel):
    agents: list[SimulationAgentEntry] = Field(default_factory=list)


@router.post("/start")
async def start_simulation(payload: SimulationStartRequest, request: Request):
    orchestrator = request.app.state.orchestrator
    if not payload.user_id:
        payload.user_id = get_user_id_from_request(request)
    return await orchestrator.start(payload)


@router.post("/stop/{session_id}")
async def stop_simulation(session_id: str, request: Request):
    orchestrator = request.app.state.orchestrator
    stopped = await orchestrator.stop(session_id)
    if not stopped:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True, "session_id": session_id}


@router.post("/pause/{session_id}")
async def pause_simulation(session_id: str, request: Request):
    orchestrator = request.app.state.orchestrator
    state = await orchestrator.pause(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found or not running")
    return state


@router.post("/resume/{session_id}")
async def resume_simulation(session_id: str, request: Request):
    orchestrator = request.app.state.orchestrator
    state = await orchestrator.resume(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found or not running")
    return state


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


@router.get("/agents")
async def get_simulation_agents(request: Request):
    user_id = get_user_id_from_request(request)
    if not user_id:
        return {"agents": [], "note": "Authentication required for persisted simulation agents."}
    return {"agents": list_simulation_agents(user_id)}


@router.put("/agents")
async def put_simulation_agents(payload: SimulationAgentsRequest, request: Request):
    user_id = get_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    entries = [item.model_dump() for item in payload.agents]
    return {"agents": set_simulation_agents(user_id, entries)}


@router.delete("/agents/{agent_name}")
async def remove_simulation_agent(agent_name: str, request: Request):
    user_id = get_user_id_from_request(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return {"agents": delete_simulation_agent(user_id, agent_name)}


@router.post("/modal/sandbox")
async def modal_sandbox(payload: SandboxRequest, request: Request):
    settings = request.app.state.settings
    orchestrator = request.app.state.orchestrator
    runtime = orchestrator.sessions.get(payload.session_id)
    metadata = {
        "source": "simulation_page",
    }
    if runtime:
        metadata["ticker"] = runtime.ticker
        metadata["tick"] = runtime.tick
        metadata["running"] = runtime.running

    result = await spin_modal_sandbox(settings, payload.prompt, payload.session_id, metadata=metadata)
    sandbox_id = result.get("sandbox_id")
    if sandbox_id and runtime and runtime.simulation_record_id:
        attach_modal_sandbox(runtime.simulation_record_id, str(sandbox_id))
    return result


@router.get("/modal/cron-health")
async def modal_status(request: Request):
    settings = request.app.state.settings
    return await modal_cron_health(settings)
