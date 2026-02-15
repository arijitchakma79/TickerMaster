from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from uuid import uuid4
from xml.sax.saxutils import escape, quoteattr

import httpx
from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import Response

from app.services.voice_agent import (
    VoiceAgentConfigError,
    VoiceAgentRuntimeError,
    run_voice_agent_turn,
    synthesize_speech_with_elevenlabs,
)

router = APIRouter(prefix="/api/twilio", tags=["twilio"])

MAX_CALL_HISTORY = 20
CALL_STATE_TTL_SECONDS = 60 * 60
AUDIO_TTL_SECONDS = 15 * 60
TURN_TIMEOUT_SECONDS = 120
POLL_PAUSE_SECONDS = 1
RECORD_MAX_LENGTH_SECONDS = 30
RECORD_TIMEOUT_SECONDS = 4


@dataclass
class _CallSessionState:
    history: list[dict[str, str]]
    updated_at: float
    pending_turn_id: str | None = None
    pending_started_at: float | None = None
    ready_audio_id: str | None = None
    ready_error: str | None = None


@dataclass
class _AudioBlob:
    payload: bytes
    mime_type: str
    expires_at: float


_CALL_SESSIONS: dict[str, _CallSessionState] = {}
_AUDIO_CACHE: dict[str, _AudioBlob] = {}
_STATE_LOCK = asyncio.Lock()
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()
_BROKER_GREETING_AUDIO_ID: str | None = None
_BROKER_GREETING_LOCK = asyncio.Lock()


def _twiml(parts: list[str]) -> Response:
    xml = '<?xml version="1.0" encoding="UTF-8"?><Response>' + "".join(parts) + "</Response>"
    return Response(content=xml, media_type="application/xml")


def _record_verb(recording_action_url: str) -> str:
    action = quoteattr(recording_action_url)
    return (
        f"<Record action={action} method=\"POST\" playBeep=\"true\" "
        f"maxLength=\"{RECORD_MAX_LENGTH_SECONDS}\" timeout=\"{RECORD_TIMEOUT_SECONDS}\" trim=\"do-not-trim\" />"
    )


def _record_prompt_twiml(recording_action_url: str) -> Response:
    return _twiml(
        [
            _record_verb(recording_action_url),
            "<Hangup/>",
        ]
    )


def _incoming_prompt_twiml(recording_action_url: str, greeting_audio_url: str | None) -> Response:
    parts: list[str] = []
    if greeting_audio_url:
        parts.append(f"<Play>{escape(greeting_audio_url)}</Play>")
    parts.extend([_record_verb(recording_action_url), "<Hangup/>"])
    return _twiml(parts)


def _processing_twiml(poll_url: str) -> Response:
    escaped_url = escape(poll_url)
    return _twiml(
        [
            f"<Pause length=\"{POLL_PAUSE_SECONDS}\"/>",
            f"<Redirect method=\"POST\">{escaped_url}</Redirect>",
        ]
    )


def _continue_conversation_twiml(*, audio_url: str, record_url: str) -> Response:
    return _twiml(
        [
            f"<Play>{escape(audio_url)}</Play>",
            _record_verb(record_url),
            "<Hangup/>",
        ]
    )


def _schedule_background(task: asyncio.Task[Any]) -> None:
    _BACKGROUND_TASKS.add(task)

    def _cleanup(completed: asyncio.Task[Any]) -> None:
        _BACKGROUND_TASKS.discard(completed)
        try:
            completed.result()
        except Exception:
            # Errors are translated into call state and surfaced to the caller via TwiML.
            pass

    task.add_done_callback(_cleanup)


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
        _CALL_SESSIONS[call_sid] = _CallSessionState(
            history=normalized,
            updated_at=now,
            pending_turn_id=None,
            pending_started_at=None,
            ready_audio_id=None,
            ready_error=None,
        )


async def _start_pending_turn(call_sid: str, turn_id: str) -> bool:
    now = time.time()
    async with _STATE_LOCK:
        await _prune_state(now)
        state = _CALL_SESSIONS.get(call_sid)
        if state is None:
            state = _CallSessionState(history=[], updated_at=now)
            _CALL_SESSIONS[call_sid] = state

        if state.pending_turn_id and not state.ready_audio_id and not state.ready_error:
            return False

        state.pending_turn_id = turn_id
        state.pending_started_at = now
        state.ready_audio_id = None
        state.ready_error = None
        state.updated_at = now
        return True


async def _finish_turn_success(call_sid: str, turn_id: str, transcript: str, response: str, audio_id: str) -> None:
    now = time.time()
    async with _STATE_LOCK:
        await _prune_state(now)
        state = _CALL_SESSIONS.get(call_sid)
        if state is None or state.pending_turn_id != turn_id:
            return

        cleaned_transcript = transcript.strip()
        if cleaned_transcript:
            state.history.append({"role": "user", "content": cleaned_transcript})
        cleaned_response = response.strip()
        if cleaned_response:
            state.history.append({"role": "assistant", "content": cleaned_response})
        state.history = state.history[-MAX_CALL_HISTORY:]

        state.pending_turn_id = None
        state.pending_started_at = None
        state.ready_audio_id = audio_id
        state.ready_error = None
        state.updated_at = now


async def _finish_turn_error(call_sid: str, turn_id: str, message: str) -> None:
    now = time.time()
    async with _STATE_LOCK:
        await _prune_state(now)
        state = _CALL_SESSIONS.get(call_sid)
        if state is None or state.pending_turn_id != turn_id:
            return
        state.pending_turn_id = None
        state.pending_started_at = None
        state.ready_audio_id = None
        state.ready_error = message.strip() or "I hit an error while processing that. Please ask again."
        state.updated_at = now


async def _poll_call_state(call_sid: str) -> tuple[str, str | None]:
    now = time.time()
    async with _STATE_LOCK:
        await _prune_state(now)
        state = _CALL_SESSIONS.get(call_sid)
        if state is None:
            return "missing", None

        state.updated_at = now

        if state.ready_error:
            error_message = state.ready_error
            state.ready_error = None
            return "error", error_message

        if state.ready_audio_id:
            audio_id = state.ready_audio_id
            state.ready_audio_id = None
            return "audio", audio_id

        if state.pending_turn_id:
            started = state.pending_started_at or now
            if now - started > TURN_TIMEOUT_SECONDS:
                state.pending_turn_id = None
                state.pending_started_at = None
                return "error", "That took too long to process. Please ask again after the beep."
            return "pending", None

        return "idle", None


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


async def _get_or_create_broker_greeting_audio_url(request: Request) -> str | None:
    settings = request.app.state.settings
    greeting_text = str(settings.twilio_broker_greeting_text or "").strip()
    if not greeting_text or not settings.eleven_labs_api_key:
        return None

    global _BROKER_GREETING_AUDIO_ID
    async with _BROKER_GREETING_LOCK:
        if _BROKER_GREETING_AUDIO_ID:
            cached = await _get_audio_blob(_BROKER_GREETING_AUDIO_ID)
            if cached is not None:
                return str(request.url_for("twilio_voice_audio", audio_id=_BROKER_GREETING_AUDIO_ID))
            _BROKER_GREETING_AUDIO_ID = None

        try:
            audio_bytes, mime_type = await synthesize_speech_with_elevenlabs(text=greeting_text, settings=settings)
        except Exception:
            return None

        audio_id = await _store_audio_blob(audio_bytes, mime_type)
        _BROKER_GREETING_AUDIO_ID = audio_id
        return str(request.url_for("twilio_voice_audio", audio_id=audio_id))


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
        for attempt in range(3):
            for candidate in candidates:
                response = await client.get(candidate, auth=auth)
                if response.is_success and response.content:
                    content_type = response.headers.get("content-type", "audio/wav")
                    return response.content, content_type
                last_status = str(response.status_code)
            await asyncio.sleep(0.4 + (attempt * 0.4))

    raise VoiceAgentRuntimeError(f"Could not download Twilio recording ({last_status})")


async def _process_turn_in_background(
    *,
    call_sid: str,
    turn_id: str,
    recording_bytes: bytes,
    content_type: str,
    request: Request,
) -> None:
    settings = request.app.state.settings
    history = await _get_call_history(call_sid)
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
        await _finish_turn_error(call_sid, turn_id, "Voice agent configuration is incomplete. Please try again later.")
        return
    except VoiceAgentRuntimeError:
        await _finish_turn_error(call_sid, turn_id, "I hit an error while processing that. Please ask again.")
        return
    except Exception:
        await _finish_turn_error(call_sid, turn_id, "I ran into an unexpected error. Please ask again.")
        return

    audio_base64 = str(result.get("audio_base64") or "")
    audio_mime_type = str(result.get("audio_mime_type") or "audio/mpeg")
    try:
        audio_bytes = base64.b64decode(audio_base64, validate=True)
    except Exception:
        await _finish_turn_error(call_sid, turn_id, "I generated a reply but audio playback failed. Please ask again.")
        return

    audio_id = await _store_audio_blob(audio_bytes, audio_mime_type)
    transcript = str(result.get("transcript") or "")
    response = str(result.get("response") or "")
    await _finish_turn_success(call_sid, turn_id, transcript, response, audio_id)


@router.post("/voice/incoming")
async def twilio_voice_incoming(request: Request, call_sid: str = Form(default="", alias="CallSid")):
    if call_sid:
        await _store_call_history(call_sid, [])
    record_url = str(request.url_for("twilio_voice_recording"))
    greeting_audio_url: str | None = None
    try:
        greeting_audio_url = await asyncio.wait_for(_get_or_create_broker_greeting_audio_url(request), timeout=8.0)
    except asyncio.TimeoutError:
        greeting_audio_url = None
    return _incoming_prompt_twiml(record_url, greeting_audio_url)


@router.post("/voice/recording")
async def twilio_voice_recording(
    request: Request,
    call_sid: str = Form(default="", alias="CallSid"),
    recording_url: str = Form(default="", alias="RecordingUrl"),
):
    call_sid = call_sid.strip()
    record_url = str(request.url_for("twilio_voice_recording"))
    if not call_sid:
        return _record_prompt_twiml(record_url)
    if not recording_url.strip():
        return _record_prompt_twiml(record_url)

    try:
        recording_bytes, content_type = await _download_recording_bytes(recording_url, request)
    except Exception:
        return _record_prompt_twiml(record_url)

    turn_id = uuid4().hex
    started = await _start_pending_turn(call_sid, turn_id)
    if not started:
        poll_url = f"{str(request.url_for('twilio_voice_poll'))}?call_sid={quote(call_sid, safe='')}"
        return _processing_twiml(poll_url)

    task = asyncio.create_task(
        _process_turn_in_background(
            call_sid=call_sid,
            turn_id=turn_id,
            recording_bytes=recording_bytes,
            content_type=content_type,
            request=request,
        )
    )
    _schedule_background(task)

    poll_url = f"{str(request.url_for('twilio_voice_poll'))}?call_sid={quote(call_sid, safe='')}"
    return _processing_twiml(poll_url)


@router.post("/voice/poll")
async def twilio_voice_poll(
    request: Request,
    call_sid_form: str = Form(default="", alias="CallSid"),
    call_sid: str = Query(default=""),
):
    resolved_call_sid = call_sid_form.strip() or call_sid.strip()
    record_url = str(request.url_for("twilio_voice_recording"))
    if not resolved_call_sid:
        return _record_prompt_twiml(record_url)

    status, payload = await _poll_call_state(resolved_call_sid)
    if status == "audio" and payload:
        audio_url = str(request.url_for("twilio_voice_audio", audio_id=payload))
        return _continue_conversation_twiml(audio_url=audio_url, record_url=record_url)

    if status == "error":
        return _record_prompt_twiml(record_url)

    if status == "pending":
        poll_url = f"{str(request.url_for('twilio_voice_poll'))}?call_sid={quote(resolved_call_sid, safe='')}"
        return _processing_twiml(poll_url)

    return _record_prompt_twiml(record_url)


@router.get("/voice/audio/{audio_id}")
async def twilio_voice_audio(audio_id: str):
    blob = await _get_audio_blob(audio_id)
    if blob is None:
        return Response(status_code=404)
    return Response(content=blob.payload, media_type=blob.mime_type)
