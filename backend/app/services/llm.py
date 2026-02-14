from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

import httpx

from app.config import Settings


def _default_commentary(context: Dict[str, Any] | None = None) -> str:
    context = context or {}
    ticker = context.get("ticker", "the market")
    sentiment = context.get("sentiment", 0)
    direction = "risk-on" if sentiment > 0.2 else "risk-off" if sentiment < -0.2 else "mixed"
    return (
        f"Live desk: {ticker} sentiment is {direction}. Watch spread widening around macro headlines, "
        "and size entries in tranches while volatility remains elevated."
    )


async def generate_openai_commentary(prompt: str, context: Dict[str, Any] | None, settings: Settings) -> Dict[str, str]:
    if not settings.openai_api_key:
        return {
            "response": _default_commentary(context),
            "model": "fallback-template",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    system_message = (
        "You are TickerMaster's live market desk commentator. Provide concise, high-signal commentary for educational trading simulation. "
        "Avoid financial advice language and keep the tone analytical."
    )

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": json.dumps({"prompt": prompt, "context": context or {}}, ensure_ascii=True),
            },
        ],
        "temperature": 0.4,
    }

    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            model = data.get("model", "gpt-4o-mini")
            return {
                "response": text,
                "model": model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
    except Exception:
        return {
            "response": _default_commentary(context),
            "model": "fallback-template",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def _safe_json_extract(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


async def generate_agent_decision(
    settings: Settings,
    agent_name: str,
    model: str,
    personality: str,
    market_state: Dict[str, Any],
    user_constraints: Dict[str, Any],
) -> Dict[str, Any]:
    if not settings.openrouter_api_key:
        return {
            "side": "hold",
            "quantity": 0,
            "confidence": 0.41,
            "rationale": "OpenRouter API key missing. Using deterministic baseline policy.",
        }

    system_prompt = (
        "You are a trading agent in a simulated market arena. "
        "Return strict JSON only with keys: side (buy/sell/hold), quantity (int), confidence (0-1), rationale (string). "
        f"Agent personality: {personality}. Respect user constraints exactly."
    )

    user_prompt = {
        "agent": agent_name,
        "state": market_state,
        "constraints": user_constraints,
    }

    payload = {
        "model": model or settings.openrouter_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
        ],
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tickermaster.local",
        "X-Title": "TickerMaster",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            output = resp.json()["choices"][0]["message"]["content"]
            parsed = _safe_json_extract(output)
            side = parsed.get("side", "hold")
            if side not in {"buy", "sell", "hold"}:
                side = "hold"
            quantity = int(max(0, min(5000, int(parsed.get("quantity", 0)))))
            confidence = float(max(0, min(1, float(parsed.get("confidence", 0.5)))))
            rationale = str(parsed.get("rationale", "No rationale returned."))
            return {
                "side": side,
                "quantity": quantity,
                "confidence": confidence,
                "rationale": rationale,
            }
    except Exception:
        return {
            "side": "hold",
            "quantity": 0,
            "confidence": 0.35,
            "rationale": "OpenRouter call failed. Fallback to no-trade to preserve simulation integrity.",
        }
