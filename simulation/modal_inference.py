from __future__ import annotations

import json
import os
from typing import Any

import modal

app = modal.App("tickermaster-simulation")

inference_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "httpx==0.28.1",
)


def _safe_json_extract(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def _normalize_decision(raw: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        payload = _safe_json_extract(raw)

    side = str(payload.get("side", "hold")).lower().strip()
    if side not in {"buy", "sell", "hold"}:
        side = "hold"

    try:
        quantity = int(payload.get("quantity", 0))
    except (TypeError, ValueError):
        quantity = 0
    quantity = max(0, min(5000, quantity))

    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(payload.get("rationale", "No rationale returned."))
    return {
        "side": side,
        "quantity": quantity,
        "confidence": confidence,
        "rationale": rationale,
    }


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


def _normalize_persona_params(raw: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        payload = _safe_json_extract(raw)

    personality = str(payload.get("personality") or "fundamental_value").strip()
    if personality not in {"quant_momentum", "fundamental_value", "retail_reactive"}:
        personality = "fundamental_value"

    strategy_prompt = str(payload.get("strategy_prompt") or "").strip()
    if not strategy_prompt:
        strategy_prompt = "Trade with disciplined risk controls. Prefer liquid names and avoid overtrading."

    aggressiveness = _clamp01(payload.get("aggressiveness"), 0.5)
    risk_limit = _clamp01(payload.get("risk_limit"), 0.5)
    trade_size = _clamp_int(payload.get("trade_size"), 60, 1, 1000)
    model = str(payload.get("model") or "").strip()

    out: dict[str, Any] = {
        "personality": personality,
        "strategy_prompt": strategy_prompt[:1200],
        "aggressiveness": aggressiveness,
        "risk_limit": risk_limit,
        "trade_size": trade_size,
    }
    if model:
        out["model"] = model
    return out


@app.function(
    image=inference_image,
    secrets=[modal.Secret.from_name("tickermaster-secrets")],
    timeout=45,
)
def agent_inference(request_payload: dict[str, Any]) -> dict[str, Any]:
    import httpx

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not openrouter_api_key:
        return {
            "side": "hold",
            "quantity": 0,
            "confidence": 0.35,
            "rationale": "OPENROUTER_API_KEY is missing in Modal secret.",
        }

    openrouter_request = request_payload.get("openrouter_request", {})
    if not isinstance(openrouter_request, dict):
        return {
            "side": "hold",
            "quantity": 0,
            "confidence": 0.35,
            "rationale": "Invalid request payload passed to Modal inference function.",
        }

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tickermaster.local",
        "X-Title": "TickerMaster",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://openrouter.ai/api/v1/chat/completions", json=openrouter_request, headers=headers)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return _normalize_decision(raw)
    except Exception as exc:
        return {
            "side": "hold",
            "quantity": 0,
            "confidence": 0.35,
            "rationale": f"Modal OpenRouter inference failed: {exc}",
        }


@app.function(
    image=inference_image,
    secrets=[modal.Secret.from_name("tickermaster-secrets")],
    timeout=45,
)
def persona_param_inference(request_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Returns a persona configuration payload derived from a public profile blob.

    Expected input: {"openrouter_request": {...}, "persona_name": "...", "public_profile": {...}}
    Output keys: personality, strategy_prompt, aggressiveness, risk_limit, trade_size (and optional model).
    """

    import httpx

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not openrouter_api_key:
        return _normalize_persona_params(
            {
                "personality": "fundamental_value",
                "strategy_prompt": "OPENROUTER_API_KEY is missing in Modal secret.",
                "aggressiveness": 0.35,
                "risk_limit": 0.5,
                "trade_size": 40,
            }
        )

    openrouter_request = request_payload.get("openrouter_request", {})
    if not isinstance(openrouter_request, dict):
        return _normalize_persona_params(
            {
                "personality": "fundamental_value",
                "strategy_prompt": "Invalid request payload passed to persona_param_inference.",
                "aggressiveness": 0.4,
                "risk_limit": 0.5,
                "trade_size": 40,
            }
        )

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tickermaster.local",
        "X-Title": "TickerMaster",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://openrouter.ai/api/v1/chat/completions", json=openrouter_request, headers=headers)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return _normalize_persona_params(raw)
    except Exception as exc:
        return _normalize_persona_params(
            {
                "personality": "fundamental_value",
                "strategy_prompt": f"Persona inference failed: {exc}",
                "aggressiveness": 0.4,
                "risk_limit": 0.5,
                "trade_size": 40,
            }
        )


@app.local_entrypoint()
def main():
    sample = {
        "agent_name": "Sample Agent",
        "personality": "quant_momentum",
        "market_state": {"ticker": "AAPL", "price": 200.0, "tick": 12},
        "user_constraints": {"risk_limit": 0.5, "aggressiveness": 0.6, "max_trade_size": 120},
        "openrouter_request": {
            "model": "meta-llama/llama-3.1-8b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return strict JSON only with keys: side (buy/sell/hold), quantity (int), "
                        "confidence (0-1), rationale (string)."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "agent": "Sample Agent",
                            "state": {"ticker": "AAPL", "price": 200.0, "tick": 12},
                            "constraints": {"risk_limit": 0.5, "aggressiveness": 0.6, "max_trade_size": 120},
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            "temperature": 0.3,
        },
    }
    print(agent_inference.local(sample))
