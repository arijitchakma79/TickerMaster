from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "tickermaster-tracker-tool-router"
SERVER_VERSION = "1.0.0"
TOOL_NAME = "route_tracker_tools"

ALLOWED_TOOLS = {
    "price",
    "volume",
    "sentiment",
    "news",
    "prediction_markets",
    "deep_research",
    "simulation",
}
ALLOWED_SOURCES = {"perplexity", "x", "reddit", "prediction_markets", "deep"}


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        token = str(item or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _route_plan(arguments: Dict[str, Any]) -> Dict[str, Any]:
    manager_prompt = str(arguments.get("manager_prompt") or "")
    available_tools = [str(item).strip().lower() for item in (arguments.get("available_tools") or [])]
    available_sources = [str(item).strip().lower() for item in (arguments.get("available_sources") or [])]
    event_hint = str(arguments.get("event_hint") or "cycle").strip().lower()

    tool_set = {token for token in available_tools if token in ALLOWED_TOOLS}
    if not tool_set:
        tool_set = set(ALLOWED_TOOLS)
    source_set = {token for token in available_sources if token in ALLOWED_SOURCES}
    if not source_set:
        source_set = set(ALLOWED_SOURCES)

    lower = f" {manager_prompt.lower()} "

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
        for source, needles in directional_patterns.items()
        if source in source_set and any(needle in lower for needle in needles)
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
        sources = list(mentioned_sources)
    elif directional_sources and len(mentioned_sources) <= 1:
        sources = list(directional_sources)
    elif directional_sources:
        sources = list(_dedupe(directional_sources + mentioned_sources))
    elif source_specific_request:
        sources = list(mentioned_sources)
    elif mentioned_sources:
        sources = [token for token in ["perplexity", "x", "reddit"] if token in source_set] + mentioned_sources
    else:
        sources = [token for token in ["perplexity", "x", "reddit"] if token in source_set]

    if signals_sentiment and "sentiment" in tool_set:
        tools.append("sentiment")
    if signals_news and "news" in tool_set:
        tools.append("news")

    if any(token in lower for token in {"bad thing", "bad news", "negative", "bearish", "investigate", "search"}):
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

    simulate_on_alert = any(token in lower for token in {"simulate", "simulation", "scenario", "backtest", "sandbox"})
    if simulate_on_alert and "simulation" in tool_set:
        tools.append("simulation")

    notification_style = "auto"
    if any(token in lower for token in {"brief", "short", "quick", "concise"}):
        notification_style = "short"
    elif any(token in lower for token in {"long", "detailed", "thesis", "full analysis", "deep dive"}):
        notification_style = "long"
    elif event_hint == "report":
        notification_style = "long"
    elif event_hint == "alert":
        notification_style = "short"

    deduped_tools = [token for token in _dedupe(tools) if token in tool_set]
    if not deduped_tools:
        deduped_tools = [token for token in ["price", "volume"] if token in tool_set] or list(tool_set)[:2]
    deduped_sources = [token for token in _dedupe(sources) if token in source_set]

    return {
        "tools": deduped_tools,
        "research_sources": deduped_sources,
        "simulate_on_alert": bool(simulate_on_alert),
        "notification_style": notification_style,
        "rationale": "mcp-tool-router",
    }


def _result(payload_id: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": payload_id, "result": data}


def _error(payload_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": payload_id, "error": {"code": code, "message": message}}


def _handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    payload_id = request.get("id")
    method = str(request.get("method") or "")
    params = request.get("params") if isinstance(request.get("params"), dict) else {}

    if method == "initialize":
        return _result(
            payload_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )

    if method == "tools/list":
        return _result(
            payload_id,
            {
                "tools": [
                    {
                        "name": TOOL_NAME,
                        "description": "Route tracker prompts to tools/sources for monitoring, research, and simulation.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "manager_prompt": {"type": "string"},
                                "available_tools": {"type": "array", "items": {"type": "string"}},
                                "available_sources": {"type": "array", "items": {"type": "string"}},
                                "event_hint": {"type": "string"},
                                "market_state": {"type": "object"},
                            },
                            "required": ["manager_prompt"],
                        },
                    }
                ]
            },
        )

    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name != TOOL_NAME:
            return _error(payload_id, -32602, f"Unknown tool: {name}")
        plan = _route_plan(arguments)
        return _result(
            payload_id,
            {
                "content": [{"type": "text", "text": "Tracker tool plan generated."}],
                "structuredContent": plan,
            },
        )

    return _error(payload_id, -32601, f"Method not found: {method}")


def main() -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception:
            response = _error(None, -32700, "Invalid JSON")
            print(json.dumps(response, ensure_ascii=True), flush=True)
            continue
        if not isinstance(request, dict):
            response = _error(None, -32600, "Invalid request payload")
            print(json.dumps(response, ensure_ascii=True), flush=True)
            continue
        response = _handle_request(request)
        print(json.dumps(response, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
