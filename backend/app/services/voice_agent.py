from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any, Sequence
from urllib.parse import quote

import httpx

from app.config import Settings

VOICE_SYSTEM_PROMPT = (
    "You are a broker-style voice assistant for TickerMaster. You have access to real market data, "
    "research, sentiment, user profile, watchlist, tracker agents, and simulations via tool calls. "
    "Always use the appropriate tool to answer; do not invent data. For stock symbols use the exact ticker "
    "(e.g. AAPL, NVDA). If the user says a company name, call search_tickers first. Keep answers concise "
    "and natural for voice: 1-3 short sentences unless the user asks for detail. Do not give financial advice; "
    "frame everything as educational or informational. If a tool fails, say so simply and suggest retrying or rephrasing."
)

TIMEFRAME_ENUM = ["24h", "7d", "30d", "90d", "180d", "1y", "2y", "5y", "10y", "max"]
MAX_TOOL_ROUNDS = 8
MAX_HISTORY_MESSAGES = 16


class VoiceAgentConfigError(RuntimeError):
    pass


class VoiceAgentRuntimeError(RuntimeError):
    pass


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


VOICE_AGENT_TOOLS: list[dict[str, Any]] = [
    _tool(
        "get_stock_quote",
        "Get current quote (price, change percent, PE, volume, market cap).",
        {"symbol": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL."}},
        required=["symbol"],
    ),
    _tool(
        "get_ticker_full_bundle",
        "Get quote, research, macro and deep research bundle for a ticker.",
        {
            "symbol": {"type": "string"},
            "timeframe": {
                "type": "string",
                "enum": TIMEFRAME_ENUM,
                "description": "Optional timeframe. Defaults to 7d.",
            },
        },
        required=["symbol"],
    ),
    _tool(
        "get_candles",
        "Get OHLC candlestick data.",
        {
            "ticker": {"type": "string"},
            "period": {"type": "string", "description": "Optional period, e.g. 1mo, 3mo, 6mo, 1y."},
            "interval": {"type": "string", "description": "Optional interval, e.g. 1d, 1wk."},
        },
        required=["ticker"],
    ),
    _tool(
        "search_tickers",
        "Search ticker symbols by company name or keyword.",
        {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        required=["query"],
    ),
    _tool(
        "get_ai_research",
        "Get AI research summary, source breakdown, citations, recommendation.",
        {
            "symbol": {"type": "string"},
            "timeframe": {"type": "string", "enum": TIMEFRAME_ENUM},
        },
        required=["symbol"],
    ),
    _tool(
        "get_sentiment",
        "Get composite sentiment and per-source breakdown.",
        {
            "symbol": {"type": "string"},
            "timeframe": {"type": "string", "enum": TIMEFRAME_ENUM},
        },
        required=["symbol"],
    ),
    _tool(
        "get_x_sentiment",
        "Get X/Twitter sentiment for a ticker.",
        {"symbol": {"type": "string"}},
        required=["symbol"],
    ),
    _tool(
        "get_deep_research",
        "Run deep research for a ticker.",
        {"symbol": {"type": "string"}},
        required=["symbol"],
    ),
    _tool(
        "analyze_research",
        "Run research with custom timeframe and optional prediction markets.",
        {
            "ticker": {"type": "string"},
            "timeframe": {"type": "string", "enum": TIMEFRAME_ENUM},
            "include_prediction_markets": {"type": "boolean"},
        },
        required=["ticker"],
    ),
    _tool(
        "get_prediction_markets",
        "Get prediction market data for a topic.",
        {"query": {"type": "string", "description": "Optional topic, default is fed."}},
    ),
    _tool("get_user_profile", "Get current user profile and watchlist.", {}),
    _tool(
        "get_user_trades",
        "Get user's simulation/trade history.",
        {"limit": {"type": "integer", "minimum": 1, "maximum": 1000}},
    ),
    _tool(
        "update_user_preferences",
        "Update user display name, watchlist, and other preferences.",
        {
            "display_name": {"type": "string"},
            "watchlist": {"type": "array", "items": {"type": "string"}},
            "poke_enabled": {"type": "boolean"},
            "tutorial_completed": {"type": "boolean"},
        },
    ),
    _tool("list_tracker_agents", "List user's tracker agents.", {}),
    _tool(
        "get_tracker_agent",
        "Get one tracker agent detail.",
        {"agent_id": {"type": "string"}},
        required=["agent_id"],
    ),
    _tool(
        "create_tracker_agent",
        "Create tracker agent for a symbol.",
        {
            "symbol": {"type": "string"},
            "name": {"type": "string"},
            "triggers": {"type": "object"},
            "auto_simulate": {"type": "boolean"},
        },
        required=["symbol", "name"],
    ),
    _tool(
        "interact_tracker_agent",
        "Send a message to tracker agent.",
        {
            "agent_id": {"type": "string"},
            "message": {"type": "string"},
        },
        required=["agent_id", "message"],
    ),
    _tool(
        "list_tracker_alerts",
        "List tracker alerts.",
        {
            "agent_id": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    ),
    _tool(
        "start_simulation",
        "Start trading simulation.",
        {
            "ticker": {"type": "string"},
            "duration_seconds": {"type": "integer"},
            "initial_price": {"type": "number"},
            "starting_cash": {"type": "number"},
            "volatility": {"type": "number"},
            "user_id": {"type": "string"},
        },
    ),
    _tool(
        "stop_simulation",
        "Stop a simulation by session id.",
        {"session_id": {"type": "string"}},
        required=["session_id"],
    ),
    _tool(
        "get_simulation_state",
        "Get simulation state by session id.",
        {"session_id": {"type": "string"}},
        required=["session_id"],
    ),
    _tool("list_simulation_sessions", "List active simulation sessions.", {}),
    _tool(
        "pause_simulation",
        "Pause a simulation by session id.",
        {"session_id": {"type": "string"}},
        required=["session_id"],
    ),
    _tool(
        "resume_simulation",
        "Resume a simulation by session id.",
        {"session_id": {"type": "string"}},
        required=["session_id"],
    ),
    _tool(
        "get_agents_activity",
        "Get recent agent activity.",
        {
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            "module": {"type": "string", "enum": ["tracker", "research", "simulation"]},
        },
    ),
    _tool(
        "get_live_commentary",
        "Generate AI market commentary.",
        {
            "prompt": {"type": "string"},
            "context": {"type": "object"},
        },
        required=["prompt"],
    ),
]


def _trim_trailing_slashes(value: str) -> str:
    return value.rstrip("/") if value else value


def _safe_parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _normalize_symbol(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().upper().replace(".", "-")


def _normalize_watchlist(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        symbol = _normalize_symbol(item)
        if symbol:
            out.append(symbol)
    return out


def _to_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _tool_failure(name: str, detail: str) -> str:
    cleaned = detail.strip() if detail else "unknown error"
    return f"Tool {name} failed: {cleaned}"


def _serialize_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=True)
    except Exception:
        return json.dumps({"error": "tool_result_serialization_failed"}, ensure_ascii=True)


def _sanitize_history(history: Sequence[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not history:
        return []
    sanitized: list[dict[str, str]] = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        text = content.strip()
        if not text:
            continue
        sanitized.append({"role": role, "content": text})
    return sanitized


def _output_format_to_mime(output_format: str) -> str:
    prefix = (output_format or "").lower()
    if prefix.startswith("pcm"):
        return "audio/wav"
    if prefix.startswith("ulaw") or prefix.startswith("mulaw"):
        return "audio/basic"
    if prefix.startswith("ogg"):
        return "audio/ogg"
    return "audio/mpeg"


async def _call_backend_json(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    response = await client.request(method, path, params=params, json=json_body, headers=headers)
    if response.is_error:
        raise httpx.HTTPStatusError(
            f"Backend request failed with {response.status_code}",
            request=response.request,
            response=response,
        )
    if not response.content:
        return {}
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return {"raw": response.text}


async def _execute_tool(
    name: str,
    args: dict[str, Any],
    backend_client: httpx.AsyncClient,
    user_headers: dict[str, str],
) -> Any:
    headers = user_headers or None
    try:
        if name == "get_stock_quote":
            symbol = _normalize_symbol(args.get("symbol"))
            if not symbol:
                return _tool_failure(name, "missing symbol")
            return await _call_backend_json(backend_client, "GET", f"/api/ticker/{quote(symbol, safe='')}/quote")

        if name == "get_ticker_full_bundle":
            symbol = _normalize_symbol(args.get("symbol"))
            if not symbol:
                return _tool_failure(name, "missing symbol")
            timeframe = args.get("timeframe") or "7d"
            return await _call_backend_json(
                backend_client,
                "GET",
                f"/api/ticker/{quote(symbol, safe='')}",
                params={"timeframe": timeframe},
            )

        if name == "get_candles":
            ticker = _normalize_symbol(args.get("ticker"))
            if not ticker:
                return _tool_failure(name, "missing ticker")
            period = args.get("period") or "1mo"
            interval = args.get("interval") or "1d"
            return await _call_backend_json(
                backend_client,
                "GET",
                f"/research/candles/{quote(ticker, safe='')}",
                params={"period": period, "interval": interval},
            )

        if name == "search_tickers":
            query = str(args.get("query") or "").strip()
            if not query:
                return _tool_failure(name, "missing query")
            limit = _to_int(args.get("limit"), 8)
            if limit is None:
                limit = 8
            limit = max(1, min(limit, 20))
            return await _call_backend_json(
                backend_client,
                "GET",
                "/research/search/tickers",
                params={"query": query, "limit": limit},
            )

        if name == "get_ai_research":
            symbol = _normalize_symbol(args.get("symbol"))
            if not symbol:
                return _tool_failure(name, "missing symbol")
            timeframe = args.get("timeframe") or "7d"
            return await _call_backend_json(
                backend_client,
                "GET",
                f"/api/ticker/{quote(symbol, safe='')}/ai-research",
                params={"timeframe": timeframe},
            )

        if name == "get_sentiment":
            symbol = _normalize_symbol(args.get("symbol"))
            if not symbol:
                return _tool_failure(name, "missing symbol")
            timeframe = args.get("timeframe") or "7d"
            return await _call_backend_json(
                backend_client,
                "GET",
                f"/api/ticker/{quote(symbol, safe='')}/sentiment",
                params={"timeframe": timeframe},
            )

        if name == "get_x_sentiment":
            symbol = _normalize_symbol(args.get("symbol"))
            if not symbol:
                return _tool_failure(name, "missing symbol")
            return await _call_backend_json(
                backend_client,
                "GET",
                f"/api/ticker/{quote(symbol, safe='')}/x-sentiment",
            )

        if name == "get_deep_research":
            symbol = _normalize_symbol(args.get("symbol"))
            if not symbol:
                return _tool_failure(name, "missing symbol")
            try:
                return await _call_backend_json(
                    backend_client,
                    "POST",
                    f"/api/research/deep/{quote(symbol, safe='')}",
                )
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    return await _call_backend_json(
                        backend_client,
                        "POST",
                        f"/research/deep/{quote(symbol, safe='')}",
                    )
                raise

        if name == "analyze_research":
            ticker = _normalize_symbol(args.get("ticker"))
            if not ticker:
                return _tool_failure(name, "missing ticker")
            body: dict[str, Any] = {
                "ticker": ticker,
                "timeframe": args.get("timeframe") or "7d",
                "include_prediction_markets": bool(args.get("include_prediction_markets", True)),
            }
            return await _call_backend_json(backend_client, "POST", "/research/analyze", json_body=body)

        if name == "get_prediction_markets":
            query = str(args.get("query") or "fed").strip() or "fed"
            return await _call_backend_json(
                backend_client,
                "GET",
                "/api/prediction-markets",
                params={"query": query},
            )

        if name == "get_user_profile":
            return await _call_backend_json(
                backend_client,
                "GET",
                "/api/user/profile",
                headers=headers,
            )

        if name == "get_user_trades":
            limit = _to_int(args.get("limit"), 200)
            if limit is None:
                limit = 200
            limit = max(1, min(limit, 1000))
            return await _call_backend_json(
                backend_client,
                "GET",
                "/api/user/trades",
                params={"limit": limit},
                headers=headers,
            )

        if name == "update_user_preferences":
            body: dict[str, Any] = {}
            if isinstance(args.get("display_name"), str):
                body["display_name"] = args["display_name"].strip()
            if isinstance(args.get("poke_enabled"), bool):
                body["poke_enabled"] = args["poke_enabled"]
            if isinstance(args.get("tutorial_completed"), bool):
                body["tutorial_completed"] = args["tutorial_completed"]
            watchlist = _normalize_watchlist(args.get("watchlist"))
            if watchlist:
                body["watchlist"] = watchlist
            return await _call_backend_json(
                backend_client,
                "PATCH",
                "/api/user/preferences",
                json_body=body,
                headers=headers,
            )

        if name == "list_tracker_agents":
            return await _call_backend_json(
                backend_client,
                "GET",
                "/api/tracker/agents",
                headers=headers,
            )

        if name == "get_tracker_agent":
            agent_id = str(args.get("agent_id") or "").strip()
            if not agent_id:
                return _tool_failure(name, "missing agent_id")
            encoded = quote(agent_id, safe="")
            try:
                return await _call_backend_json(
                    backend_client,
                    "GET",
                    f"/api/tracker/agents/{encoded}/detail",
                    headers=headers,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    return await _call_backend_json(
                        backend_client,
                        "GET",
                        f"/api/tracker/agents/{encoded}",
                        headers=headers,
                    )
                raise

        if name == "create_tracker_agent":
            symbol = _normalize_symbol(args.get("symbol"))
            agent_name = str(args.get("name") or "").strip()
            if not symbol or not agent_name:
                return _tool_failure(name, "missing symbol or name")
            body: dict[str, Any] = {"symbol": symbol, "name": agent_name}
            if isinstance(args.get("triggers"), dict):
                body["triggers"] = args["triggers"]
            if isinstance(args.get("auto_simulate"), bool):
                body["auto_simulate"] = args["auto_simulate"]
            return await _call_backend_json(
                backend_client,
                "POST",
                "/api/tracker/agents",
                json_body=body,
                headers=headers,
            )

        if name == "interact_tracker_agent":
            agent_id = str(args.get("agent_id") or "").strip()
            message = str(args.get("message") or "").strip()
            if not agent_id or not message:
                return _tool_failure(name, "missing agent_id or message")
            return await _call_backend_json(
                backend_client,
                "POST",
                f"/api/tracker/agents/{quote(agent_id, safe='')}/interact",
                json_body={"message": message},
                headers=headers,
            )

        if name == "list_tracker_alerts":
            params: dict[str, Any] = {}
            agent_id = str(args.get("agent_id") or "").strip()
            if agent_id:
                params["agent_id"] = agent_id
            limit = _to_int(args.get("limit"), None)
            if limit is not None:
                params["limit"] = max(1, min(limit, 100))
            return await _call_backend_json(
                backend_client,
                "GET",
                "/api/tracker/alerts",
                params=params or None,
                headers=headers,
            )

        if name == "start_simulation":
            body: dict[str, Any] = {}
            ticker = _normalize_symbol(args.get("ticker"))
            if ticker:
                body["ticker"] = ticker
            duration_seconds = _to_int(args.get("duration_seconds"), None)
            if duration_seconds is not None:
                body["duration_seconds"] = duration_seconds
            initial_price = _to_float(args.get("initial_price"), None)
            if initial_price is not None:
                body["initial_price"] = initial_price
            starting_cash = _to_float(args.get("starting_cash"), None)
            if starting_cash is not None:
                body["starting_cash"] = starting_cash
            volatility = _to_float(args.get("volatility"), None)
            if volatility is not None:
                body["volatility"] = volatility
            if isinstance(args.get("user_id"), str) and args.get("user_id"):
                body["user_id"] = args["user_id"]
            return await _call_backend_json(
                backend_client,
                "POST",
                "/simulation/start",
                json_body=body,
                headers=headers,
            )

        if name == "stop_simulation":
            session_id = str(args.get("session_id") or "").strip()
            if not session_id:
                return _tool_failure(name, "missing session_id")
            return await _call_backend_json(
                backend_client,
                "POST",
                f"/simulation/stop/{quote(session_id, safe='')}",
                headers=headers,
            )

        if name == "get_simulation_state":
            session_id = str(args.get("session_id") or "").strip()
            if not session_id:
                return _tool_failure(name, "missing session_id")
            return await _call_backend_json(
                backend_client,
                "GET",
                f"/simulation/state/{quote(session_id, safe='')}",
                headers=headers,
            )

        if name == "list_simulation_sessions":
            return await _call_backend_json(backend_client, "GET", "/simulation/sessions", headers=headers)

        if name == "pause_simulation":
            session_id = str(args.get("session_id") or "").strip()
            if not session_id:
                return _tool_failure(name, "missing session_id")
            return await _call_backend_json(
                backend_client,
                "POST",
                f"/simulation/pause/{quote(session_id, safe='')}",
                headers=headers,
            )

        if name == "resume_simulation":
            session_id = str(args.get("session_id") or "").strip()
            if not session_id:
                return _tool_failure(name, "missing session_id")
            return await _call_backend_json(
                backend_client,
                "POST",
                f"/simulation/resume/{quote(session_id, safe='')}",
                headers=headers,
            )

        if name == "get_agents_activity":
            params: dict[str, Any] = {}
            limit = _to_int(args.get("limit"), None)
            if limit is not None:
                params["limit"] = max(1, min(limit, 200))
            module = args.get("module")
            if isinstance(module, str) and module in {"tracker", "research", "simulation"}:
                params["module"] = module
            return await _call_backend_json(
                backend_client,
                "GET",
                "/api/agents/activity",
                params=params or None,
                headers=headers,
            )

        if name == "get_live_commentary":
            prompt = str(args.get("prompt") or "").strip() or "Give a short market commentary."
            body: dict[str, Any] = {"prompt": prompt}
            context = args.get("context")
            if isinstance(context, dict):
                body["context"] = context
            return await _call_backend_json(backend_client, "POST", "/chat/commentary", json_body=body, headers=headers)

        return _tool_failure(name, "unknown tool")
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "http_error"
        return _tool_failure(name, str(status))
    except Exception as exc:
        return _tool_failure(name, str(exc))


async def _load_optional_user_context(
    backend_client: httpx.AsyncClient,
    user_headers: dict[str, str],
) -> str | None:
    if not user_headers:
        return None

    profile: Any | None = None
    activity: Any | None = None

    try:
        profile = await _call_backend_json(
            backend_client,
            "GET",
            "/api/user/profile",
            headers=user_headers,
        )
    except Exception:
        profile = None

    try:
        activity = await _call_backend_json(
            backend_client,
            "GET",
            "/api/agents/activity",
            params={"limit": 5},
            headers=user_headers,
        )
    except Exception:
        activity = None

    parts: list[str] = []
    if profile is not None:
        parts.append(f"User profile: {json.dumps(profile, ensure_ascii=True)}")
    if activity is not None:
        parts.append(f"Recent activity: {json.dumps(activity, ensure_ascii=True)}")
    if not parts:
        return None
    return " ".join(parts)


async def _post_openai_chat(
    settings: Settings,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise VoiceAgentConfigError("OPENAI_API_KEY is not configured")

    body = {
        "model": settings.openai_voice_model,
        "messages": messages,
        "tools": VOICE_AGENT_TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers)
    if response.is_error:
        raise VoiceAgentRuntimeError(f"OpenAI chat completion failed: {response.status_code}")
    return response.json()


async def _run_openai_tool_loop(
    *,
    transcript: str,
    history: Sequence[dict[str, Any]] | None,
    settings: Settings,
    backend_client: httpx.AsyncClient,
    user_headers: dict[str, str],
    user_context: str | None,
) -> tuple[str, str]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": VOICE_SYSTEM_PROMPT}]
    if user_context:
        messages.append({"role": "system", "content": user_context})
    messages.extend(_sanitize_history(history))
    messages.append({"role": "user", "content": transcript})

    model_name = settings.openai_voice_model
    for round_index in range(MAX_TOOL_ROUNDS):
        completion = await _post_openai_chat(settings, messages)
        model_name = str(completion.get("model") or model_name)
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VoiceAgentRuntimeError("OpenAI returned no choices")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        if not isinstance(message, dict):
            raise VoiceAgentRuntimeError("OpenAI returned invalid message payload")

        assistant_content = _content_to_text(message.get("content")).strip()
        tool_calls = message.get("tool_calls")

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
        }

        if isinstance(tool_calls, list) and tool_calls:
            assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

            for call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function_payload = tool_call.get("function")
                if not isinstance(function_payload, dict):
                    continue

                tool_name = str(function_payload.get("name") or "").strip()
                if not tool_name:
                    continue
                raw_args = function_payload.get("arguments")
                arguments = _safe_parse_json(raw_args if isinstance(raw_args, str) else None)

                tool_result = await _execute_tool(tool_name, arguments, backend_client, user_headers)
                call_id = str(tool_call.get("id") or f"tool_call_{round_index}_{call_index}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": _serialize_tool_result(tool_result),
                    }
                )
            continue

        if assistant_content:
            return assistant_content, model_name

    return (
        "I could not complete that request right now. Please try rephrasing or ask me to retry.",
        model_name,
    )


async def transcribe_audio_with_elevenlabs(
    *,
    audio_bytes: bytes,
    filename: str,
    content_type: str | None,
    settings: Settings,
) -> str:
    if not settings.eleven_labs_api_key:
        raise VoiceAgentConfigError("ELEVEN_LABS_API_KEY is not configured")
    if not audio_bytes:
        raise VoiceAgentRuntimeError("Audio payload is empty")

    form_data: dict[str, Any] = {"model_id": settings.eleven_labs_stt_model}
    if settings.eleven_labs_stt_language_code:
        form_data["language_code"] = settings.eleven_labs_stt_language_code

    files = {
        "file": (
            filename or "utterance.webm",
            audio_bytes,
            content_type or "application/octet-stream",
        )
    }
    headers = {"xi-api-key": settings.eleven_labs_api_key}

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            data=form_data,
            files=files,
            headers=headers,
        )
    if response.is_error:
        raise VoiceAgentRuntimeError(f"ElevenLabs STT failed: {response.status_code}")

    payload = response.json()
    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

        transcripts = payload.get("transcripts")
        if isinstance(transcripts, list):
            chunks: list[str] = []
            for item in transcripts:
                if not isinstance(item, dict):
                    continue
                channel_text = item.get("text")
                if isinstance(channel_text, str) and channel_text.strip():
                    chunks.append(channel_text.strip())
            if chunks:
                return " ".join(chunks)
    raise VoiceAgentRuntimeError("ElevenLabs STT returned an empty transcript")


async def synthesize_speech_with_elevenlabs(
    *,
    text: str,
    settings: Settings,
) -> tuple[bytes, str]:
    if not settings.eleven_labs_api_key:
        raise VoiceAgentConfigError("ELEVEN_LABS_API_KEY is not configured")
    if not text.strip():
        raise VoiceAgentRuntimeError("Cannot synthesize empty response")

    output_format = settings.eleven_labs_tts_output_format
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{quote(settings.eleven_labs_voice_id, safe='')}/stream"
    headers = {
        "xi-api-key": settings.eleven_labs_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": settings.eleven_labs_tts_model,
    }

    chunks: list[bytes] = []
    async with httpx.AsyncClient(timeout=90.0) as client:
        async with client.stream(
            "POST",
            url,
            params={"output_format": output_format},
            json=payload,
            headers=headers,
        ) as response:
            if response.is_error:
                raise VoiceAgentRuntimeError(f"ElevenLabs TTS failed: {response.status_code}")
            async for chunk in response.aiter_bytes():
                if chunk:
                    chunks.append(chunk)

    audio_bytes = b"".join(chunks)
    if not audio_bytes:
        raise VoiceAgentRuntimeError("ElevenLabs TTS returned empty audio")
    return audio_bytes, _output_format_to_mime(output_format)


async def run_voice_agent_turn(
    *,
    audio_bytes: bytes,
    filename: str,
    content_type: str | None,
    history: Sequence[dict[str, Any]] | None,
    request_user_id: str | None,
    settings: Settings,
) -> dict[str, Any]:
    transcript = await transcribe_audio_with_elevenlabs(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
        settings=settings,
    )

    user_id = (request_user_id or settings.x_user_id or "").strip()
    user_headers = {"x-user-id": user_id} if user_id else {}

    backend_base_url = _trim_trailing_slashes(settings.backend_base_url)
    if not backend_base_url:
        raise VoiceAgentConfigError("BACKEND_BASE_URL is not configured")

    async with httpx.AsyncClient(base_url=backend_base_url, timeout=45.0) as backend_client:
        user_context = await _load_optional_user_context(backend_client, user_headers)
        response_text, model = await _run_openai_tool_loop(
            transcript=transcript,
            history=history,
            settings=settings,
            backend_client=backend_client,
            user_headers=user_headers,
            user_context=user_context,
        )

    audio_out, mime_type = await synthesize_speech_with_elevenlabs(text=response_text, settings=settings)
    return {
        "transcript": transcript,
        "response": response_text,
        "audio_base64": base64.b64encode(audio_out).decode("ascii"),
        "audio_mime_type": mime_type,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
