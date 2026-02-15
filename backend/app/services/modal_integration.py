from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from typing import Any, Dict

from app.config import Settings

MODAL_DOCS_LINK = "https://modal.com/docs/guide/sandboxes"


def _modal_sdk_available() -> bool:
    return importlib.util.find_spec("modal") is not None


def _launch_modal_sandbox_sync(
    settings: Settings,
    prompt: str,
    session_id: str,
    metadata: dict[str, Any],
) -> Dict[str, Any]:
    import modal  # type: ignore[import-not-found]

    # Ensure Modal SDK picks up credentials even when backend runs outside `modal run`.
    os.environ.setdefault("MODAL_TOKEN_ID", settings.modal_token_id)
    os.environ.setdefault("MODAL_TOKEN_SECRET", settings.modal_token_secret)

    app = modal.App.lookup(settings.modal_simulation_app_name, create_if_missing=True)

    secrets_payload: dict[str, str | None] = {}
    if settings.openrouter_api_key:
        secrets_payload["OPENROUTER_API_KEY"] = settings.openrouter_api_key

    secrets = [modal.Secret.from_dict(secrets_payload)] if secrets_payload else None
    image = modal.Image.debian_slim(python_version=settings.modal_sandbox_python_version)

    sandbox = modal.Sandbox.create(
        app=app,
        name=f"tickermaster-{session_id[:10]}",
        image=image,
        timeout=max(30, settings.modal_sandbox_timeout_seconds),
        idle_timeout=max(10, settings.modal_sandbox_idle_timeout_seconds),
        env={
            "SESSION_ID": session_id,
            "SIMULATION_PROMPT": prompt,
        },
        secrets=secrets,
    )

    boot_event = {
        "type": "sandbox_boot",
        "session_id": session_id,
        "prompt_preview": prompt[:220],
        "metadata": metadata,
    }
    bootstrap_script = "\n".join(
        [
            "import json",
            "import time",
            f"event = {json.dumps(boot_event)}",
            "print(json.dumps(event), flush=True)",
            "for beat in range(3):",
            "    print(json.dumps({'type': 'sandbox_heartbeat', 'beat': beat + 1}), flush=True)",
            "    time.sleep(1)",
        ]
    )

    process = sandbox.exec(
        "python",
        "-c",
        bootstrap_script,
        timeout=min(30, max(10, settings.modal_sandbox_idle_timeout_seconds)),
        bufsize=1,
    )

    stdout_preview: list[str] = []
    for idx, raw in enumerate(process.stdout):
        line = raw.strip()
        if line:
            stdout_preview.append(line)
        if idx >= 3:
            break

    process.poll()
    sandbox_id = str(sandbox.object_id)
    app_id = str(getattr(app, "app_id", "") or "").strip()
    app_dashboard_url = f"https://modal.com/apps/{app_id}" if app_id else ""
    sandbox_dashboard_url = f"https://modal.com/id/{sandbox_id}"
    return {
        "status": "started",
        "session_id": session_id,
        "sandbox_id": sandbox_id,
        "app_id": app_id,
        "app_name": settings.modal_simulation_app_name,
        "prompt_preview": prompt[:220],
        "stdout_preview": stdout_preview,
        "dashboard_url": app_dashboard_url or sandbox_dashboard_url,
        "app_dashboard_url": app_dashboard_url or None,
        "sandbox_dashboard_url": sandbox_dashboard_url,
        "metadata": metadata,
    }


async def spin_modal_sandbox(
    settings: Settings,
    prompt: str,
    session_id: str,
    metadata: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    clean_prompt = prompt.strip()
    if not clean_prompt:
        clean_prompt = "Run a multi-agent market simulation with realistic slippage and delayed news diffusion."

    if not settings.modal_token_id or not settings.modal_token_secret:
        return {
            "status": "stub",
            "message": "Modal credentials missing. Running local simulation only.",
            "session_id": session_id,
            "prompt_preview": clean_prompt[:220],
            "metadata": metadata,
            "link": MODAL_DOCS_LINK,
        }

    if not _modal_sdk_available():
        return {
            "status": "failed",
            "error": "Modal SDK is not installed in the backend environment.",
            "session_id": session_id,
            "hint": "Add `modal` to backend dependencies, then restart the API.",
            "link": MODAL_DOCS_LINK,
        }

    try:
        return await asyncio.to_thread(
            _launch_modal_sandbox_sync,
            settings,
            clean_prompt,
            session_id,
            metadata,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "session_id": session_id,
            "hint": "Check Modal credentials and app permissions, then retry.",
            "link": MODAL_DOCS_LINK,
        }


async def modal_cron_health(settings: Settings) -> Dict[str, Any]:
    if not settings.modal_token_id or not settings.modal_token_secret:
        return {
            "status": "stub",
            "message": "Modal credentials missing; cron must be configured externally.",
            "polling_interval_seconds": settings.tracker_poll_interval_seconds,
            "app_name": settings.modal_simulation_app_name,
            "sdk_available": _modal_sdk_available(),
            "inference_function_name": settings.modal_inference_function_name,
            "inference_timeout_seconds": settings.modal_inference_timeout_seconds,
        }
    if not _modal_sdk_available():
        return {
            "status": "missing_dependency",
            "message": "Modal credentials are loaded, but the Modal SDK is missing in the running backend environment.",
            "polling_interval_seconds": settings.tracker_poll_interval_seconds,
            "app_name": settings.modal_simulation_app_name,
            "sdk_available": False,
            "inference_function_name": settings.modal_inference_function_name,
            "inference_timeout_seconds": settings.modal_inference_timeout_seconds,
            "install_hint": "Install backend deps (`pip install -r backend/requirements.txt`) and restart backend.",
        }
    return {
        "status": "configured",
        "message": "Modal credentials loaded. Sandbox runtime is available from backend.",
        "polling_interval_seconds": settings.tracker_poll_interval_seconds,
        "app_name": settings.modal_simulation_app_name,
        "sdk_available": _modal_sdk_available(),
        "sandbox_timeout_seconds": settings.modal_sandbox_timeout_seconds,
        "sandbox_idle_timeout_seconds": settings.modal_sandbox_idle_timeout_seconds,
        "inference_function_name": settings.modal_inference_function_name,
        "inference_timeout_seconds": settings.modal_inference_timeout_seconds,
    }
