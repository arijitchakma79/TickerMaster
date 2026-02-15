from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import httpx

from app.config import Settings

_MODAL_INFERENCE_BACKOFF_UNTIL: datetime | None = None


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

    ticker = (context or {}).get("ticker", "the stock")
    system_message = (
        f"You are TickerMaster's research assistant for {ticker}. "
        "Your primary job is to DIRECTLY ANSWER the user's question first. "
        "If they ask about a person (CEO, CFO, founder), company facts, or general information, answer that question directly and concisely. "
        "You have access to current market research context which you can use to supplement your answer when relevant. "
        "After answering the question, you may briefly mention relevant market sentiment or news if it adds value. "
        "Keep responses concise and helpful. Avoid generic market commentary unless specifically asked."
    )

    # Format context more clearly for the model
    context_summary = ""
    if context:
        if context.get("aggregate_sentiment") is not None:
            sentiment = context["aggregate_sentiment"]
            sentiment_label = "bullish" if sentiment > 0.2 else "bearish" if sentiment < -0.2 else "neutral"
            context_summary += f"Current sentiment: {sentiment_label} ({sentiment:.2f}). "
        if context.get("recommendation"):
            context_summary += f"Recommendation: {context['recommendation']}. "
        if context.get("narratives"):
            context_summary += f"Key narratives: {'; '.join(context['narratives'][:3])}. "

    user_content = f"Question: {prompt}"
    if context_summary:
        user_content += f"\n\nMarket context for reference (use only if relevant to the question): {context_summary}"

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
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


def _modal_sdk_available() -> bool:
    return importlib.util.find_spec("modal") is not None


def _normalize_agent_decision_output(raw: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
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
    quantity = int(max(0, min(5000, quantity)))

    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = float(max(0, min(1, confidence)))

    rationale = str(payload.get("rationale", "No rationale returned."))
    target_ticker_raw = payload.get("target_ticker", payload.get("ticker", ""))
    target_ticker = ""
    if isinstance(target_ticker_raw, str):
        normalized = target_ticker_raw.strip().upper().replace(".", "-")
        if normalized:
            target_ticker = normalized
    return {
        "side": side,
        "quantity": quantity,
        "confidence": confidence,
        "rationale": rationale,
        "target_ticker": target_ticker,
    }


def _invoke_modal_agent_decision_sync(settings: Settings, request_payload: Dict[str, Any]) -> Any:
    import modal  # type: ignore[import-not-found]

    os.environ.setdefault("MODAL_TOKEN_ID", settings.modal_token_id)
    os.environ.setdefault("MODAL_TOKEN_SECRET", settings.modal_token_secret)

    fn = modal.Function.from_name(
        settings.modal_simulation_app_name,
        settings.modal_inference_function_name,
    )
    return fn.remote(request_payload)


async def _generate_agent_decision_via_modal(
    settings: Settings,
    request_payload: Dict[str, Any],
) -> Dict[str, Any] | None:
    global _MODAL_INFERENCE_BACKOFF_UNTIL

    now = datetime.now(timezone.utc)
    if _MODAL_INFERENCE_BACKOFF_UNTIL and now < _MODAL_INFERENCE_BACKOFF_UNTIL:
        return None

    if not settings.modal_token_id or not settings.modal_token_secret:
        return None
    if not _modal_sdk_available():
        return None

    timeout = max(5, settings.modal_inference_timeout_seconds)

    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_invoke_modal_agent_decision_sync, settings, request_payload),
            timeout=timeout,
        )
        return _normalize_agent_decision_output(raw)
    except Exception:
        # Back off repeated failing lookups/remotes to avoid stalling simulation ticks.
        _MODAL_INFERENCE_BACKOFF_UNTIL = now + timedelta(seconds=90)
        return None


async def generate_agent_decision(
    settings: Settings,
    agent_name: str,
    model: str,
    personality: str,
    market_state: Dict[str, Any],
    user_constraints: Dict[str, Any],
    use_modal_inference: bool = False,
) -> Dict[str, Any]:
    system_prompt = (
        "You are a trading agent in a simulated market arena. "
        "Return strict JSON only with keys: side (buy/sell/hold), quantity (int), confidence (0-1), "
        "target_ticker (string from allowed_tickers), rationale (string). "
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

    if use_modal_inference:
        modal_out = await _generate_agent_decision_via_modal(
            settings,
            {
                "agent_name": agent_name,
                "personality": personality,
                "market_state": market_state,
                "user_constraints": user_constraints,
                "openrouter_request": payload,
            },
        )
        if modal_out:
            return modal_out

    if not settings.openrouter_api_key:
        return {
            "side": "hold",
            "quantity": 0,
            "confidence": 0.41,
            "rationale": (
                "OpenRouter API key missing. Configure OPENROUTER_API_KEY, or enable Modal inference "
                "with a deployed function."
            ),
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
            return _normalize_agent_decision_output(output)
    except Exception:
        return {
            "side": "hold",
            "quantity": 0,
            "confidence": 0.35,
            "rationale": "OpenRouter call failed. Fallback to no-trade to preserve simulation integrity.",
        }


def _fallback_tracker_intent(prompt: str) -> Dict[str, Any]:
    text = prompt.strip()
    upper = text.upper()
    symbol_match = re.search(r"\b[A-Z]{1,5}\b", upper)
    symbol = symbol_match.group(0) if symbol_match else "AAPL"
    threshold_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    threshold = float(threshold_match.group(1)) if threshold_match else 2.0
    lower = text.lower()
    intent = "create_agent"
    if "pause" in lower or "stop" in lower:
        intent = "pause_agent"
    elif "delete" in lower or "remove" in lower:
        intent = "delete_agent"
    elif "status" in lower or "what are you seeing" in lower:
        intent = "status"
    return {
        "intent": intent,
        "symbol": symbol,
        "name": f"{symbol} Associate",
        "auto_simulate": "simulate" in lower,
        "triggers": {
            "price_change_pct": threshold,
            "volume_spike_ratio": 1.8,
            "sentiment_bearish_threshold": -0.25,
            "sentiment_bullish_threshold": 0.25,
        },
        "response": f"Prepared {intent} plan for {symbol} with {threshold:.2f}% price trigger and sentiment checks.",
    }


async def parse_tracker_instruction(settings: Settings, prompt: str) -> Dict[str, Any]:
    if not settings.openai_api_key:
        return _fallback_tracker_intent(prompt)

    system = (
        "You are a hedge-fund associate assistant. Convert user instruction into strict JSON with keys: "
        "intent(create_agent|update_agent|delete_agent|pause_agent|resume_agent|status|research_note), "
        "symbol, name, auto_simulate(boolean), triggers(object), response(string). "
        "triggers should include any of: price_change_pct, volume_spike_ratio, sentiment_bearish_threshold, "
        "sentiment_bullish_threshold, x_bearish_threshold, rsi_low, rsi_high. "
        "No markdown."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = _safe_json_extract(content)
            if not parsed:
                return _fallback_tracker_intent(prompt)
            parsed.setdefault("triggers", {})
            parsed.setdefault("auto_simulate", False)
            parsed.setdefault("response", "Instruction parsed.")
            return parsed
    except Exception:
        return _fallback_tracker_intent(prompt)


async def tracker_agent_chat_response(
    settings: Settings,
    agent: Dict[str, Any],
    market_state: Dict[str, Any],
    research_state: Dict[str, Any],
    user_message: str,
) -> Dict[str, str]:
    if not settings.openai_api_key:
        symbol = str(agent.get("symbol", ""))
        sentiment = float(research_state.get("aggregate_sentiment", 0.0))
        return {
            "response": (
                f"{agent.get('name', 'Agent')} update for {symbol}: price {market_state.get('price')} and sentiment {sentiment:.2f}. "
                "I am monitoring catalyst risk, social tone, and volume regime for trigger changes."
            ),
            "model": "fallback-template",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    system = (
        "You are a buy-side research associate agent briefing a hedge fund manager. "
        "Be concise, specific, and action-oriented. Mention current market state, social sentiment, and next steps. "
        "No investment guarantee language."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "agent": agent,
                        "market_state": market_state,
                        "research_state": research_state,
                        "manager_message": user_message,
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        "temperature": 0.25,
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return {
                "response": content,
                "model": "gpt-4o-mini",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
    except Exception:
        return {
            "response": "Briefing unavailable due to model request failure. Continue monitoring price, flow, and sentiment drift.",
            "model": "fallback-template",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
