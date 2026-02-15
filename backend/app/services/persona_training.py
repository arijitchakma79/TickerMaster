from __future__ import annotations

import asyncio
import importlib.util
import json
from typing import Any, Dict

import httpx

from app.config import Settings
from app.schemas import AgentConfig


def _modal_sdk_available() -> bool:
    return importlib.util.find_spec("modal") is not None


def _safe_json_extract(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _clamp01(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return float(max(0.0, min(1.0, out)))


def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return int(max(lo, min(hi, out)))


def _fallback_agent_config(name: str) -> AgentConfig:
    clean = (name or "").strip() or "Unnamed Persona"
    upper = clean.upper()

    if "VANGUARD" in upper or "BLACKROCK" in upper:
        personality = "fundamental_value"
        prompt = "Trade like a long-horizon allocator. Prefer liquid names, reduce turnover, and size conservatively."
        return AgentConfig(
            name=clean,
            personality=personality,
            strategy_prompt=prompt,
            aggressiveness=0.35,
            risk_limit=0.45,
            trade_size=50,
        )

    if "JANE" in upper or "CITADEL" in upper:
        personality = "quant_momentum"
        prompt = "Trade like a systematic desk. React to momentum and news skew, keep strict risk limits."
        return AgentConfig(
            name=clean,
            personality=personality,
            strategy_prompt=prompt,
            aggressiveness=0.6,
            risk_limit=0.55,
            trade_size=80,
        )

    personality = "retail_reactive"
    prompt = "Trade like a reactive trader. Follow headlines, size down on uncertainty, avoid overtrading."
    return AgentConfig(
        name=clean,
        personality=personality,
        strategy_prompt=prompt,
        aggressiveness=0.5,
        risk_limit=0.5,
        trade_size=60,
    )


def _normalize_persona_params(persona_name: str, raw: Any) -> AgentConfig:
    payload = _safe_json_extract(raw)
    clean_name = (persona_name or "").strip() or str(payload.get("name") or "").strip() or "Unnamed Persona"

    personality = str(payload.get("personality") or "").strip() or "fundamental_value"
    if personality not in {"quant_momentum", "fundamental_value", "retail_reactive"}:
        personality = "fundamental_value"

    strategy_prompt = str(payload.get("strategy_prompt") or "").strip()
    if not strategy_prompt:
        strategy_prompt = _fallback_agent_config(clean_name).strategy_prompt

    aggressiveness = _clamp01(payload.get("aggressiveness"), 0.5)
    risk_limit = _clamp01(payload.get("risk_limit"), 0.5)
    trade_size = _clamp_int(payload.get("trade_size"), 60, 1, 1000)

    model = str(payload.get("model") or "").strip() or "meta-llama/llama-3.1-8b-instruct"

    return AgentConfig(
        name=clean_name,
        personality=personality,  # type: ignore[arg-type]
        model=model,
        strategy_prompt=strategy_prompt[:1200],
        aggressiveness=aggressiveness,
        risk_limit=risk_limit,
        trade_size=trade_size,
        active=True,
    )


def _persona_openrouter_request(persona_name: str, public_profile: Dict[str, Any], model: str) -> Dict[str, Any]:
    system = (
        "You are generating a trading persona configuration. "
        "Return strict JSON only with keys: personality(one of quant_momentum|fundamental_value|retail_reactive), "
        "strategy_prompt(string), aggressiveness(0-1), risk_limit(0-1), trade_size(int 1-1000). "
        "No markdown."
    )

    user = {
        "persona_name": persona_name,
        "public_profile": public_profile,
        "instructions": "Infer reasonable behavior parameters from the profile. Keep strategy_prompt concise and actionable.",
    }

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
        ],
        "temperature": 0.2,
    }


def _invoke_modal_persona_params_sync(settings: Settings, request_payload: Dict[str, Any]) -> Any:
    import os

    import modal  # type: ignore[import-not-found]

    os.environ.setdefault("MODAL_TOKEN_ID", settings.modal_token_id)
    os.environ.setdefault("MODAL_TOKEN_SECRET", settings.modal_token_secret)

    fn = modal.Function.from_name(
        settings.modal_simulation_app_name,
        "persona_param_inference",
    )
    return fn.remote(request_payload)


async def infer_persona_params(
    settings: Settings,
    persona_name: str,
    public_profile: Dict[str, Any] | None = None,
    *,
    prefer_modal: bool = True,
) -> AgentConfig:
    """
    Infers an AgentConfig from a "public profile" blob.

    Notes:
    - This is not model-weight training. It produces persona parameters (prompt + risk knobs).
    - If Modal is configured, we call the Modal function `persona_param_inference` which uses OPENROUTER_API_KEY
      from a Modal secret to do the actual LLM inference.
    - If Modal is unavailable, we fall back to direct OpenRouter (if configured), else a deterministic default.
    """

    profile = public_profile or {}
    model = settings.openrouter_model or "meta-llama/llama-3.1-8b-instruct"
    openrouter_request = _persona_openrouter_request(persona_name, profile, model=model)

    if (
        prefer_modal
        and settings.modal_token_id
        and settings.modal_token_secret
        and _modal_sdk_available()
    ):
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(
                    _invoke_modal_persona_params_sync,
                    settings,
                    {
                        "persona_name": persona_name,
                        "public_profile": profile,
                        "openrouter_request": openrouter_request,
                    },
                ),
                timeout=max(5, settings.modal_inference_timeout_seconds),
            )
            return _normalize_persona_params(persona_name, raw)
        except Exception:
            pass

    if settings.openrouter_api_key:
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tickermaster.local",
            "X-Title": "TickerMaster",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=openrouter_request,
                    headers=headers,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return _normalize_persona_params(persona_name, content)
        except Exception:
            pass

    return _fallback_agent_config(persona_name)

