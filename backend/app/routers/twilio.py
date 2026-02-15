from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from uuid import uuid4
from xml.sax.saxutils import escape, quoteattr

import httpx
from fastapi import APIRouter, Form, Request
from fastapi.responses import Response

from app.services.voice_agent import (
    VoiceAgentConfigError,
    VoiceAgentRuntimeError,
    run_voice_agent_turn,
)

router = APIRouter(prefix="/api/twilio", tags=["twilio"])

MAX_CALL_HISTORY = 20
CALL_STATE_TTL_SECONDS = 60 * 60
AUDIO_TTL_SECONDS = 15 * 60


@dataclass
class _CallSessionState:
    history: list[dict[str, str]]
    updated_at: float


@dataclass
class _AudioBlob:
    payload: bytes
    mime_type: str
    expires_at: float


_CALL_SESSIONS: dict[str, _CallSessionState] = {}
_AUDIO_CACHE: dict[str, _AudioBlob] = {}
_STATE_LOCK = asyncio.Lock()


def _twiml(parts: list[str]) -> Response:
    xml = '<?xml version="1.0" encoding="UTF-8"?><Response>' + "".join(parts) + "</Response>"
    return Response(content=xml, media_type="application/xml")


def _record_prompt_twiml(recording_action_url: str, prompt: str) -> Response:
    action = quoteattr(recording_action_url)
    prompt_text = escape(prompt)
    return _twiml(
        [
            f"<Say>{prompt_text}</Say>",
            f"<Record action={action} method=\"POST\" playBeep=\"true\" maxLength=\"30\" timeout=\"4\" trim=\"trim-silence\" />",
            "<Say>I did not hear anything. Goodbye.</Say>",
            "<Hangup/>",
        ]
    )


async def _prune_state(now: float) -> None:
    stale_calls = [sid for sid, session in _CALL_SESSIONS.items() if now - session.updated_at > CALL_STATE_TTL_SECONDS]
    for sid in stale_calls:
        _CALL_SESSIONS.pop(sid, None)

    stale_audio = [audio_id for audio_id, blob in _AUDIO_CACHE.items() if blob.expires_at <= now]
    for audio_id in stale_audio:
        _AUDIO_CACHE.pop(audio_id, None)


async def _get_call_history(call_sid: str) -> list[dict[str, str]]:
    now = time.time()
    async with _STATE_LOCK:
        await _prune_state(now)
        state = _CALL_SESSIONS.get(call_sid)
        if state is None:
            return []
        state.updated_at = now
        return list(state.history)


async def _store_call_history(call_sid: str, history: list[dict[str, str]]) -> None:
    now = time.time()
    normalized = history[-MAX_CALL_HISTORY:]
    async with _STATE_LOCK:
        await _prune_state(now)
        _CALL_SESSIONS[call_sid] = _CallSessionState(history=normalized, updated_at=now)


async def _store_audio_blob(audio_bytes: bytes, mime_type: str) -> str:
    now = time.time()
    audio_id = uuid4().hex
    async with _STATE_LOCK:
        await _prune_state(now)
        _AUDIO_CACHE[audio_id] = _AudioBlob(
            payload=audio_bytes,
            mime_type=mime_type or "audio/mpeg",
            expires_at=now + AUDIO_TTL_SECONDS,
        )
    return audio_id


async def _get_audio_blob(audio_id: str) -> _AudioBlob | None:
    now = time.time()
    async with _STATE_LOCK:
        await _prune_state(now)
        return _AUDIO_CACHE.get(audio_id)


async def _download_recording_bytes(recording_url: str, request: Request) -> tuple[bytes, str]:
    settings = request.app.state.settings
    account_sid = settings.twilio_account_sid
    auth_token = settings.twilio_auth_token
    auth: tuple[str, str] | None = None
    if account_sid and auth_token:
        auth = (account_sid, auth_token)

    candidates = []
    cleaned = recording_url.strip()
    if not cleaned:
        raise VoiceAgentRuntimeError("Twilio recording URL is empty")
    if cleaned.endswith(".wav") or cleaned.endswith(".mp3"):
        candidates.append(cleaned)
    else:
        candidates.extend([f"{cleaned}.wav", cleaned])

    last_status = "download_error"
    async with httpx.AsyncClient(timeout=90.0) as client:
        for candidate in candidates:
            response = await client.get(candidate, auth=auth)
            if response.is_success and response.content:
                content_type = response.headers.get("content-type", "audio/wav")
                return response.content, content_type
            last_status = str(response.status_code)

    raise VoiceAgentRuntimeError(f"Could not download Twilio recording ({last_status})")


@router.post("/voice/incoming")
async def twilio_voice_incoming(request: Request, call_sid: str = Form(default="", alias="CallSid")):
    if call_sid:
        await _store_call_history(call_sid, [])
    record_url = str(request.url_for("twilio_voice_recording"))
    return _record_prompt_twiml(
        record_url,
        "Welcome to TickerMaster. Ask a market question after the beep.",
    )


@router.post("/voice/recording")
async def twilio_voice_recording(
    request: Request,
    call_sid: str = Form(default="", alias="CallSid"),
    recording_url: str = Form(default="", alias="RecordingUrl"),
):
    record_url = str(request.url_for("twilio_voice_recording"))
    if not recording_url.strip():
        return _record_prompt_twiml(record_url, "I did not receive audio. Please ask your question after the beep.")

    try:
        recording_bytes, content_type = await _download_recording_bytes(recording_url, request)
    except Exception:
        return _record_prompt_twiml(record_url, "I could not access that recording. Please try again.")

    history = await _get_call_history(call_sid) if call_sid else []
    settings = request.app.state.settings
    try:
        result = await run_voice_agent_turn(
            audio_bytes=recording_bytes,
            filename="twilio-call.wav",
            content_type=content_type,
            history=history,
            request_user_id=settings.x_user_id or None,
            settings=settings,
        )
    except VoiceAgentConfigError:
        return _record_prompt_twiml(record_url, "Voice agent configuration is incomplete. Please try again later.")
    except VoiceAgentRuntimeError:
        return _record_prompt_twiml(record_url, "I hit an error while processing that. Please ask again after the beep.")

    if call_sid:
        updated_history = [
            *history,
            {"role": "user", "content": str(result.get("transcript") or "").strip()},
            {"role": "assistant", "content": str(result.get("response") or "").strip()},
        ]
        await _store_call_history(call_sid, updated_history)

    audio_base64 = str(result.get("audio_base64") or "")
    audio_mime_type = str(result.get("audio_mime_type") or "audio/mpeg")
    try:
        audio_bytes = base64.b64decode(audio_base64, validate=True)
    except Exception:
        return _record_prompt_twiml(record_url, "I generated a reply but audio playback failed. Please ask again.")

    audio_id = await _store_audio_blob(audio_bytes, audio_mime_type)
    audio_url = str(request.url_for("twilio_voice_audio", audio_id=audio_id))

    return _twiml(
        [
            f"<Play>{escape(audio_url)}</Play>",
            "<Pause length=\"1\"/>",
            "<Say>You can ask another question after the beep.</Say>",
            f"<Record action={quoteattr(record_url)} method=\"POST\" playBeep=\"true\" maxLength=\"30\" timeout=\"4\" trim=\"trim-silence\" />",
            "<Say>No further audio received. Goodbye.</Say>",
            "<Hangup/>",
        ]
    )


@router.get("/voice/audio/{audio_id}")
async def twilio_voice_audio(audio_id: str):
    blob = await _get_audio_blob(audio_id)
    if blob is None:
        return Response(status_code=404)
    return Response(content=blob.payload, media_type=blob.mime_type)
