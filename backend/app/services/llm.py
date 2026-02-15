from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx

from app.config import Settings
from app.services.mcp_tool_router import plan_tracker_tools_via_mcp

_MODAL_INFERENCE_BACKOFF_UNTIL: datetime | None = None
_TRACKER_ALLOWED_TOOLS = {
    "price",
    "volume",
    "sentiment",
    "news",
    "prediction_markets",
    "deep_research",
    "simulation",
}
_TRACKER_ALLOWED_SOURCES = {"perplexity", "x", "reddit", "prediction_markets", "deep"}


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
    interval_match = re.search(r"every\s+(\d+)\s*(second|sec|minute|min|hour|hr)s?\b", text, flags=re.IGNORECASE)
    poll_interval_seconds = 120
    if interval_match:
        magnitude = int(interval_match.group(1))
        unit = interval_match.group(2).lower()
        if unit.startswith("sec"):
            poll_interval_seconds = max(30, magnitude)
        elif unit.startswith("hour") or unit == "hr":
            poll_interval_seconds = max(30, magnitude * 3600)
        else:
            poll_interval_seconds = max(30, magnitude * 60)

    lower = text.lower()
    intent = "create_agent"
    if "pause" in lower or "stop" in lower:
        intent = "pause_agent"
    elif "delete" in lower or "remove" in lower:
        intent = "delete_agent"
    elif "status" in lower or "what are you seeing" in lower:
        intent = "status"

    tools = ["price", "volume", "sentiment", "news"]
    if "prediction" in lower or "kalshi" in lower or "polymarket" in lower:
        tools.append("prediction_markets")
    if "deep research" in lower or "browserbase" in lower:
        tools.append("deep_research")
    if "simulate" in lower or "sandbox" in lower or "scenario" in lower or "backtest" in lower:
        tools.append("simulation")

    research_sources = []
    if "perplexity" in lower:
        research_sources.append("perplexity")
    if "reddit" in lower:
        research_sources.append("reddit")
    if " x " in f" {lower} " or "twitter" in lower:
        research_sources.append("x")
    if "prediction" in lower:
        research_sources.append("prediction_markets")
    if not research_sources:
        research_sources = ["perplexity", "x", "reddit"]

    notify_channels = ["twilio"] if any(token in lower for token in {"sms", "text", "twilio"}) else ["twilio", "poke"]
    phone_match = re.search(r"(\+?\d[\d\-\s\(\)]{8,}\d)", text)
    notify_phone = phone_match.group(1).strip() if phone_match else ""

    wants_reports = any(
        token in lower
        for token in {
            "report every",
            "summary every",
            "update every",
            "send report",
            "scheduled report",
            "notification every",
            "notify every",
            "send me notification",
            "send me notifications",
        }
    )
    wants_alerts = any(token in lower for token in {"alert", "trigger", "notify me when", "if it", "if price"})
    if wants_reports and wants_alerts:
        report_mode = "hybrid"
    elif wants_reports:
        report_mode = "periodic"
    elif wants_alerts:
        report_mode = "triggers_only"
    else:
        report_mode = "hybrid"

    schedule_mode = "custom" if interval_match else "realtime"
    if "hourly" in lower:
        schedule_mode = "hourly"
        poll_interval_seconds = 3600
    if "daily" in lower:
        schedule_mode = "daily"
        poll_interval_seconds = 86400

    baseline_mode = "prev_close"
    if "from open" in lower or "session open" in lower:
        baseline_mode = "session_open"
    elif "from last check" in lower:
        baseline_mode = "last_check"
    elif "from last alert" in lower:
        baseline_mode = "last_alert"

    message_style = "auto"
    if any(token in lower for token in {"brief", "short", "quick", "concise"}):
        message_style = "short"
    elif any(token in lower for token in {"long", "detailed", "thesis", "deep dive", "full analysis"}):
        message_style = "long"

    return {
        "intent": intent,
        "symbol": symbol,
        "name": f"{symbol} Associate",
        "auto_simulate": "simulate" in lower or "sandbox" in lower or "backtest" in lower,
        "triggers": {
            "price_change_pct": threshold,
            "volume_spike_ratio": 1.8,
            "sentiment_bearish_threshold": -0.25,
            "sentiment_bullish_threshold": 0.25,
            "tools": list(dict.fromkeys(tools)),
            "research_sources": list(dict.fromkeys(research_sources)),
            "poll_interval_seconds": poll_interval_seconds,
            "report_interval_seconds": poll_interval_seconds,
            "report_mode": report_mode,
            "schedule_mode": schedule_mode,
            "daily_run_time": "09:30",
            "timezone": "America/New_York",
            "custom_time_enabled": False,
            "baseline_mode": baseline_mode,
            "tool_mode": "auto",
            "notify_channels": notify_channels,
            "notify_phone": notify_phone[:40] if notify_phone else "",
            "simulate_on_alert": "simulate" in lower or "sandbox" in lower,
            "notification_style": message_style,
        },
        "response": (
            f"Prepared {intent} plan for {symbol} with {threshold:.2f}% price trigger, "
            f"{poll_interval_seconds}s cadence, report_mode={report_mode}, and tools: {', '.join(dict.fromkeys(tools))}."
        ),
    }


async def parse_tracker_instruction(settings: Settings, prompt: str) -> Dict[str, Any]:
    if not settings.openai_api_key:
        return _fallback_tracker_intent(prompt)

    system = (
        "You are a hedge-fund associate assistant. Convert user instruction into strict JSON with keys: "
        "intent(create_agent|update_agent|delete_agent|pause_agent|resume_agent|status|research_note), "
        "symbol, name, auto_simulate(boolean), triggers(object), response(string). "
        "triggers should include any of: price_change_pct, volume_spike_ratio, sentiment_bearish_threshold, "
        "sentiment_bullish_threshold, x_bearish_threshold, rsi_low, rsi_high, "
        "poll_interval_seconds, report_interval_seconds, report_mode(triggers_only|periodic|hybrid), "
        "schedule_mode(realtime|hourly|daily|custom), daily_run_time(HH:MM), timezone(IANA), "
        "custom_time_enabled(boolean), notification_style(auto|short|long), "
        "start_at(ISO8601 datetime in UTC), "
        "baseline_mode(prev_close|last_check|last_alert|session_open), tool_mode(auto|manual), "
        "simulate_on_alert, notify_channels(array), notify_phone, tools(array), research_sources(array). "
        "tools values: price, volume, sentiment, news, prediction_markets, deep_research, simulation. "
        "research_sources values: perplexity, x, reddit, prediction_markets, deep. "
        "notify_channels values: twilio, poke. "
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
            if isinstance(parsed.get("triggers"), dict):
                parsed["triggers"].setdefault("poll_interval_seconds", 120)
                parsed["triggers"].setdefault("report_interval_seconds", parsed["triggers"].get("poll_interval_seconds", 120))
                parsed["triggers"].setdefault("report_mode", "hybrid")
                parsed["triggers"].setdefault("schedule_mode", "realtime")
                parsed["triggers"].setdefault("daily_run_time", "09:30")
                parsed["triggers"].setdefault("timezone", "America/New_York")
                parsed["triggers"].setdefault("custom_time_enabled", False)
                parsed["triggers"].setdefault("baseline_mode", "prev_close")
                parsed["triggers"].setdefault("tool_mode", "auto")
                parsed["triggers"].setdefault("tools", ["price", "volume", "sentiment", "news"])
                parsed["triggers"].setdefault("research_sources", ["perplexity", "x", "reddit"])
                parsed["triggers"].setdefault("notify_channels", ["twilio", "poke"])
                parsed["triggers"].setdefault("simulate_on_alert", bool(parsed.get("auto_simulate")))
                parsed["triggers"].setdefault("notification_style", "auto")
            parsed.setdefault("response", "Instruction parsed.")
            return parsed
    except Exception:
        return _fallback_tracker_intent(prompt)


def _fallback_tracker_runtime_plan(
    *,
    manager_prompt: str,
    available_tools: List[str],
    available_sources: List[str],
    event_hint: str,
) -> Dict[str, Any]:
    lower = f" {manager_prompt.lower()} "
    tool_set = set(token for token in available_tools if token in _TRACKER_ALLOWED_TOOLS)
    source_set = set(token for token in available_sources if token in _TRACKER_ALLOWED_SOURCES)

    # Base operating profile for a beginner analyst assistant.
    signals_price = any(token in lower for token in {" price ", " drop ", " rally ", " up ", " down ", "%", "breakout", "drawdown"})
    signals_volume = any(token in lower for token in {" volume ", "flow", "volume spike", "liquidity"})
    signals_sentiment = any(token in lower for token in {" sentiment ", "tone", "bullish", "bearish", "social"})
    signals_news = any(
        token in lower
        for token in {" news ", "headline", "catalyst", "filing", "investigate", "search", "what happened", "why moved"}
    )
    tools: List[str] = []
    if "price" in tool_set and (signals_price or not signals_volume):
        tools.append("price")
    if "volume" in tool_set and (signals_volume or signals_price):
        tools.append("volume")
    if signals_sentiment and "sentiment" in tool_set:
        tools.append("sentiment")
    if signals_news and "news" in tool_set:
        tools.append("news")
    sources: List[str] = []

    source_mentions: Dict[str, tuple[str, ...]] = {
        "reddit": (" reddit ", " r/", " at reddit"),
        "perplexity": (" perplexity ",),
        "x": (" x ", " twitter ", " from x", " on x", " at x", " from twitter", " on twitter", " at twitter"),
        "prediction_markets": (" prediction market", " prediction markets", " polymarket", " kalshi", " trading market"),
        "deep": (" deep research", " deep-research", " browserbase", " at deep research"),
    }
    directional_patterns: Dict[str, tuple[str, ...]] = {
        "reddit": (" from reddit", " on reddit", " at reddit", " via reddit", " reddit sentiment"),
        "perplexity": (" from perplexity", " at perplexity", " via perplexity", " perplexity search"),
        "x": (" from x", " on x", " at x", " via x", " from twitter", " on twitter", " at twitter", " twitter sentiment"),
        "prediction_markets": (" from prediction market", " at prediction market", " from prediction markets", " from polymarket", " from kalshi", " from trading market"),
        "deep": (" from deep research", " at deep research", " via deep research", " from browserbase"),
    }
    mentioned_sources = [
        source
        for source, needles in source_mentions.items()
        if source in source_set and any(needle in lower for needle in needles)
    ]
    directional_sources = [
        source
        for source, patterns in directional_patterns.items()
        if source in source_set and any(pattern in lower for pattern in patterns)
    ]
    exclusive = any(token in lower for token in {" only ", " just ", " strictly ", " exclusively "})
    broad_source_markers = {
        " all sources",
        " cross-source",
        " cross source",
        " combine sources",
        " blend sources",
        " multi-source",
    }
    source_specific_request = bool(mentioned_sources) and not any(token in lower for token in broad_source_markers)

    if exclusive and mentioned_sources:
        sources = list(dict.fromkeys(mentioned_sources))
    elif directional_sources and len(mentioned_sources) <= 1:
        sources = list(dict.fromkeys(directional_sources))
    elif directional_sources:
        sources = list(dict.fromkeys(directional_sources + mentioned_sources))
    elif source_specific_request:
        sources = list(dict.fromkeys(mentioned_sources))
    elif mentioned_sources:
        sources = list(dict.fromkeys([token for token in ["perplexity", "x", "reddit"] if token in source_set] + mentioned_sources))
    else:
        sources = [token for token in ["perplexity", "x", "reddit"] if token in source_set]

    if any(token in lower for token in {"bad thing", "bad news", "negative", "bearish news", "investigate", "search"}):
        if "news" in tool_set:
            tools.append("news")
        if "sentiment" in tool_set:
            tools.append("sentiment")
        if "perplexity" in source_set and not source_specific_request:
            sources.append("perplexity")

    source_tokens = set(sources)
    if source_tokens.intersection({"reddit", "x"}):
        if "sentiment" in tool_set:
            tools.append("sentiment")
    if "perplexity" in source_tokens:
        if "sentiment" in tool_set:
            tools.append("sentiment")
        if "news" in tool_set:
            tools.append("news")
    if "prediction_markets" in source_tokens and "prediction_markets" in tool_set:
        tools.append("prediction_markets")
    if "deep" in source_tokens and "deep_research" in tool_set:
        tools.append("deep_research")

    simulate_on_alert = any(token in lower for token in {"simulate", "simulation", "backtest", "scenario", "sandbox"})
    if simulate_on_alert and "simulation" in tool_set:
        tools.append("simulation")

    notification_style = "auto"
    if any(token in lower for token in {"brief", "short", "quick", "concise"}):
        notification_style = "short"
    elif any(token in lower for token in {"long", "detailed", "full analysis", "thesis", "deep dive"}):
        notification_style = "long"
    elif event_hint == "report":
        notification_style = "long"
    elif event_hint == "alert":
        notification_style = "short"

    deduped_tools = [token for token in dict.fromkeys(tools) if token in tool_set]
    if not deduped_tools:
        deduped_tools = [token for token in ["price", "volume"] if token in tool_set] or [token for token in available_tools if token in tool_set][:2]
    deduped_sources = [token for token in dict.fromkeys(sources) if token in source_set]

    return {
        "tools": deduped_tools,
        "research_sources": deduped_sources,
        "simulate_on_alert": bool(simulate_on_alert),
        "notification_style": notification_style,
        "rationale": "fallback-heuristics",
    }


def _normalize_runtime_plan(
    candidate: Dict[str, Any] | None,
    *,
    available_tools: List[str],
    available_sources: List[str],
    fallback_style_hint: str = "auto",
    rationale_default: str = "planner",
) -> Dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None

    tool_set = {token for token in available_tools if token in _TRACKER_ALLOWED_TOOLS}
    source_set = {token for token in available_sources if token in _TRACKER_ALLOWED_SOURCES}

    raw_tools = candidate.get("tools") if isinstance(candidate.get("tools"), list) else []
    resolved_tools = [str(item).strip().lower() for item in raw_tools if str(item).strip()]
    resolved_tools = [item for item in dict.fromkeys(resolved_tools) if item in tool_set]
    if not resolved_tools:
        return None

    raw_sources = candidate.get("research_sources") if isinstance(candidate.get("research_sources"), list) else []
    resolved_sources = [str(item).strip().lower() for item in raw_sources if str(item).strip()]
    resolved_sources = [item for item in dict.fromkeys(resolved_sources) if item in source_set]

    style_raw = str(candidate.get("notification_style") or fallback_style_hint or "auto").strip().lower()
    style = style_raw if style_raw in {"auto", "short", "long"} else "auto"

    return {
        "tools": resolved_tools,
        "research_sources": resolved_sources,
        "simulate_on_alert": bool(candidate.get("simulate_on_alert", False)),
        "notification_style": style,
        "rationale": str(candidate.get("rationale") or rationale_default),
    }


async def decide_tracker_runtime_plan(
    settings: Settings,
    *,
    manager_prompt: str,
    available_tools: List[str] | None = None,
    available_sources: List[str] | None = None,
    market_state: Dict[str, Any] | None = None,
    event_hint: str = "cycle",
) -> Dict[str, Any]:
    tools = [str(item).strip().lower() for item in (available_tools or list(_TRACKER_ALLOWED_TOOLS)) if str(item).strip()]
    tools = [item for item in dict.fromkeys(tools) if item in _TRACKER_ALLOWED_TOOLS]
    sources = [str(item).strip().lower() for item in (available_sources or list(_TRACKER_ALLOWED_SOURCES)) if str(item).strip()]
    sources = [item for item in dict.fromkeys(sources) if item in _TRACKER_ALLOWED_SOURCES]

    mcp_plan = await plan_tracker_tools_via_mcp(
        settings,
        manager_prompt=manager_prompt,
        available_tools=tools,
        available_sources=sources,
        market_state=market_state or {},
        event_hint=event_hint,
    )
    normalized_mcp_plan = _normalize_runtime_plan(
        mcp_plan,
        available_tools=tools,
        available_sources=sources,
        fallback_style_hint=("long" if event_hint == "report" else "short" if event_hint == "alert" else "auto"),
        rationale_default="mcp",
    )
    if normalized_mcp_plan:
        return normalized_mcp_plan

    if not settings.openai_api_key:
        return _fallback_tracker_runtime_plan(
            manager_prompt=manager_prompt,
            available_tools=tools,
            available_sources=sources,
            event_hint=event_hint,
        )

    system = (
        "You are selecting tracker tools for a beginner hedge-fund assistant. "
        "Return strict JSON with keys: tools(array), research_sources(array), "
        "simulate_on_alert(boolean), notification_style(auto|short|long), rationale(string). "
        "Only include tools from: price, volume, sentiment, news, prediction_markets, deep_research, simulation. "
        "Only include sources from: perplexity, x, reddit, prediction_markets, deep. "
        "Keep it practical: include only what this instruction needs."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "manager_prompt": manager_prompt,
                        "available_tools": tools,
                        "available_sources": sources,
                        "market_state": market_state or {},
                        "event_hint": event_hint,
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            content = str(resp.json()["choices"][0]["message"]["content"] or "")
            parsed = _safe_json_extract(content)
            normalized_llm_plan = _normalize_runtime_plan(
                parsed if isinstance(parsed, dict) else None,
                available_tools=tools,
                available_sources=sources,
                fallback_style_hint=("long" if event_hint == "report" else "short" if event_hint == "alert" else "auto"),
                rationale_default="llm",
            )
            if not normalized_llm_plan:
                raise ValueError("invalid_plan")
            return normalized_llm_plan
    except Exception:
        return _fallback_tracker_runtime_plan(
            manager_prompt=manager_prompt,
            available_tools=tools,
            available_sources=sources,
            event_hint=event_hint,
        )


async def tracker_agent_chat_response(
    settings: Settings,
    agent: Dict[str, Any],
    market_state: Dict[str, Any],
    research_state: Dict[str, Any],
    user_message: str,
    memory_context: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    memory_context = memory_context or {}
    if not settings.openai_api_key:
        symbol = str(agent.get("symbol", ""))
        sentiment = float(research_state.get("aggregate_sentiment", 0.0))
        latest_instruction = str(memory_context.get("latest_instruction") or "").strip()
        latest_instruction_line = f" Last instruction: {latest_instruction[:160]}." if latest_instruction else ""
        return {
            "response": (
                f"Broker update on {symbol}: price {market_state.get('price')} and sentiment {sentiment:.2f}. "
                "I am tracking catalyst risk, social tone drift, and volume regime changes for you."
                f"{latest_instruction_line}"
            ),
            "model": "fallback-template",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    system = (
        "You are a personal broker-style assistant for a beginner investor. "
        "Help configure and operate stock tracker agents and explain updates clearly. "
        "Write naturally, concise but complete, with practical next actions and key risks. "
        "Use only provided context. If data is missing, state exactly what is missing."
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
                        "memory_context": memory_context,
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
            data = resp.json()
            content = str(data["choices"][0]["message"]["content"]).strip()
            return {
                "response": content,
                "model": str(data.get("model") or "gpt-4o-mini"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
    except Exception:
        return {
            "response": "Briefing unavailable due to model request failure. Continue monitoring price, flow, and sentiment drift.",
            "model": "fallback-template",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


async def tracker_context_query_response(
    settings: Settings,
    *,
    question: str,
    context: Dict[str, Any],
) -> Dict[str, str]:
    question = question.strip()
    if not question:
        return {
            "response": "No question provided.",
            "model": "context-fallback",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    if not settings.openai_api_key:
        runs = context.get("runs") if isinstance(context.get("runs"), list) else []
        history = context.get("history") if isinstance(context.get("history"), list) else []
        thesis = context.get("thesis") if isinstance(context.get("thesis"), dict) else {}
        latest_run = runs[0] if runs else {}
        latest_hist = history[0] if history else {}
        return {
            "response": (
                f"Context fallback answer for: {question}\n"
                f"Latest run type: {latest_run.get('run_type', 'n/a')} at {latest_run.get('created_at', 'n/a')}.\n"
                f"Latest instruction: {latest_hist.get('raw_prompt', 'n/a')}.\n"
                f"Current thesis stance_score: {thesis.get('stance_score', 'n/a')} confidence: {thesis.get('confidence', 'n/a')}."
            ),
            "model": "context-fallback",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    system = (
        "You are a hedge-fund analyst copilot. Answer strictly from provided tracker context "
        "(agent runs, thesis memory, history, CSV logs). Be factual and time-aware. "
        "If evidence is missing, say what is missing."
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "context": context,
                    },
                    ensure_ascii=True,
                ),
            },
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            text = str(data["choices"][0]["message"]["content"]).strip()
            return {
                "response": text,
                "model": str(data.get("model") or "gpt-4o-mini"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
    except Exception:
        return {
            "response": "Context query failed at model layer. Use recent runs/history tables for manual inspection.",
            "model": "context-fallback",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
