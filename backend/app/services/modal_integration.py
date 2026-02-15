from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import re
from typing import Any, Dict

from app.config import Settings

MODAL_DOCS_LINK = "https://modal.com/docs/guide/sandboxes"

_CODE_FENCE_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _modal_sdk_available() -> bool:
    return importlib.util.find_spec("modal") is not None


def _extract_python_code(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    match = _CODE_FENCE_PATTERN.search(text)
    if match:
        return (match.group(1) or "").strip()
    return text


def _default_generated_code() -> str:
    # Keep this tiny and deterministic as a safe fallback when codegen is unavailable.
    return "\n".join(
        [
            "import json",
            "import os",
            "",
            "def main():",
            "    session_id = os.environ.get('SESSION_ID', '')",
            "    prompt = os.environ.get('SIMULATION_PROMPT', '')",
            "    out = {",
            "        'ok': True,",
            "        'session_id': session_id,",
            "        'summary': 'Sandbox is live. (Template strategy code used.)',",
            "        'prompt_preview': prompt[:180],",
            "    }",
            "    print(json.dumps(out), flush=True)",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
    )


def _generate_strategy_code(settings: Settings, prompt: str) -> Dict[str, Any]:
    """
    Generate Python strategy code from a natural-language prompt.

    This runs on the backend (outside the sandbox) so sandbox execution doesn't need LLM credentials.
    """
    if not settings.openrouter_api_key:
        return {"code": _default_generated_code(), "source": "template", "model": None}

    system = (
        "You write Python 3.11 code to run inside a sandbox. Output ONLY python code (no markdown). "
        "Requirements: finish quickly (<2s), no network calls, no file writes, and print exactly one JSON line "
        "to stdout with keys: ok(boolean), session_id(string), summary(string). "
        "Read SESSION_ID and SIMULATION_PROMPT from environment variables."
    )
    user = f"SIMULATION_PROMPT: {prompt}"
    payload = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tickermaster.local",
        "X-Title": "TickerMaster",
    }

    try:
        import httpx

        with httpx.Client(timeout=20.0) as client:
            resp = client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            content = str(resp.json()["choices"][0]["message"]["content"] or "")
    except Exception as exc:
        return {
            "code": _default_generated_code(),
            "source": "template",
            "model": settings.openrouter_model,
            "error": str(exc),
        }

    code = _extract_python_code(content)
    if not code:
        return {"code": _default_generated_code(), "source": "template", "model": settings.openrouter_model}

    # Validate syntax before shipping to the sandbox.
    try:
        compile(code, "<generated_strategy>", "exec")
    except Exception:
        return {"code": _default_generated_code(), "source": "template", "model": settings.openrouter_model}

    return {"code": code, "source": "openrouter", "model": settings.openrouter_model}


def _preview_text(text: str, *, max_lines: int = 28, max_chars: int = 2800) -> str:
    lines = (text or "").splitlines()
    preview = "\n".join(lines[: max(1, int(max_lines))]).strip()
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "\n# …(truncated)…"
    return preview


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

    # 1) Backend code generation (AI-generated code path).
    codegen = _generate_strategy_code(settings, prompt)
    generated_code = str(codegen.get("code") or "")
    generated_code_b64 = base64.b64encode(generated_code.encode("utf-8")).decode("ascii")

    boot_event = {
        "type": "sandbox_boot",
        "session_id": session_id,
        "prompt_preview": prompt[:220],
        "metadata": metadata,
    }
    runner = "\n".join(
        [
            "import base64",
            "import json",
            "import traceback",
            f"event = {json.dumps(boot_event)}",
            "print(json.dumps(event), flush=True)",
            f"code_b64 = {json.dumps(generated_code_b64)}",
            "code = base64.b64decode(code_b64.encode('ascii')).decode('utf-8')",
            "try:",
            "    globals_dict = {'__name__': '__main__', '__file__': '<generated_strategy>'}",
            "    exec(compile(code, '<generated_strategy>', 'exec'), globals_dict, globals_dict)",
            "    print(json.dumps({'type': 'sandbox_exec_complete', 'ok': True}), flush=True)",
            "except Exception as exc:",
            "    print(json.dumps({'type': 'sandbox_exec_complete', 'ok': False, 'error': str(exc)}), flush=True)",
            "    traceback.print_exc()",
        ]
    )

    process = sandbox.exec(
        "python",
        "-c",
        runner,
        timeout=min(30, max(10, settings.modal_sandbox_idle_timeout_seconds)),
        bufsize=1,
    )

    stdout_preview: list[str] = []
    for idx, raw in enumerate(process.stdout):
        line = raw.strip()
        if line:
            stdout_preview.append(line)
        if idx >= 11:
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
        "generated_code_preview": _preview_text(generated_code),
        "codegen_source": codegen.get("source"),
        "codegen_model": codegen.get("model"),
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
            "message": "Cloud sandbox runtime is temporarily unavailable.",
            "polling_interval_seconds": settings.tracker_poll_interval_seconds,
            "app_name": settings.modal_simulation_app_name,
            "sdk_available": False,
            "inference_function_name": settings.modal_inference_function_name,
            "inference_timeout_seconds": settings.modal_inference_timeout_seconds,
            "install_hint": "Cloud sandbox runtime is unavailable. Local simulation mode remains available.",
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
