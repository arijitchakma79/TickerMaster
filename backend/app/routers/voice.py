from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.services.voice_agent import (
    VoiceAgentConfigError,
    VoiceAgentRuntimeError,
    run_voice_agent_turn,
)

router = APIRouter(prefix="/api/voice", tags=["voice"])


def _parse_history(raw: str | None) -> list[dict[str, Any]]:
    if raw is None or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("history must be valid JSON") from exc

    if not isinstance(parsed, list):
        raise ValueError("history must be a JSON array")

    out: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            out.append(item)
    return out


@router.post("/turn")
async def voice_turn(
    request: Request,
    audio: UploadFile = File(...),
    history: str | None = Form(default=None),
):
    try:
        history_payload = _parse_history(history)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    settings = request.app.state.settings
    request_user_id = request.headers.get("x-user-id")
    try:
        return await run_voice_agent_turn(
            audio_bytes=audio_bytes,
            filename=audio.filename or "utterance.webm",
            content_type=audio.content_type,
            history=history_payload,
            request_user_id=request_user_id,
            settings=settings,
        )
    except VoiceAgentConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VoiceAgentRuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
