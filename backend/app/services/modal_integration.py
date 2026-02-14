from __future__ import annotations

from typing import Any, Dict

import httpx

from app.config import Settings


async def spin_modal_sandbox(settings: Settings, prompt: str, session_id: str) -> Dict[str, Any]:
    if not settings.modal_token_id or not settings.modal_token_secret:
        return {
            "status": "stub",
            "message": "Modal credentials missing. Returning local simulation handle.",
            "session_id": session_id,
            "prompt": prompt,
            "link": "https://modal.com/docs/guide/sandbox",
        }

    headers = {
        "Modal-Key-Id": settings.modal_token_id,
        "Modal-Key-Secret": settings.modal_token_secret,
        "Content-Type": "application/json",
    }
    body = {
        "name": "tickermaster-sandbox",
        "metadata": {"session_id": session_id, "prompt": prompt},
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            # Placeholder endpoint. Replace with your deployed Modal webhook/runner URL.
            resp = await client.post("https://api.modal.com/v1/sandbox/start", headers=headers, json=body)
            resp.raise_for_status()
            return {"status": "started", "response": resp.json()}
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "session_id": session_id,
            "hint": "Wire this to your own Modal endpoint exposed by a deployed app.",
        }


async def modal_cron_health(settings: Settings) -> Dict[str, Any]:
    if not settings.modal_token_id or not settings.modal_token_secret:
        return {
            "status": "stub",
            "message": "Modal credentials missing; cron must be configured externally.",
            "polling_interval_seconds": settings.tracker_poll_interval_seconds,
        }
    return {
        "status": "configured",
        "message": "Modal credentials loaded. Configure cron in your Modal deployment for minute-level polling.",
        "polling_interval_seconds": settings.tracker_poll_interval_seconds,
    }
