from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

from app.config import Settings

_REQUEST_TIMEOUT_DEFAULT = 6
_INITIALIZE_METHOD = "initialize"
_TOOLS_LIST_METHOD = "tools/list"
_TOOLS_CALL_METHOD = "tools/call"
_TOOL_NAME = "route_tracker_tools"
_TOOL_REDDIT = "get_reddit_sentiment"
_TOOL_X = "get_x_sentiment"
_TOOL_PREDICTION = "get_prediction_markets"
_TOOL_PERPLEXITY = "get_perplexity_brief"
_TOOL_DEEP = "get_deep_research_brief"

_SOURCE_WEIGHTS = {
    "Perplexity Sonar": 0.45,
    "Reddit API": 0.30,
    "X API": 0.25,
}


def _parse_command(command: str) -> List[str]:
    token = str(command or "").strip()
    if not token:
        script_path = Path(__file__).resolve().parents[1] / "mcp" / "tracker_tools_server.py"
        return [sys.executable, str(script_path)]
    return shlex.split(token, posix=(os.name != "nt"))


def _extract_structured_content(result: Dict[str, Any]) -> Dict[str, Any] | None:
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0] if isinstance(content[0], dict) else {}
        text = str(first.get("text") or "").strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                maybe = json.loads(text)
            except Exception:
                return None
            if isinstance(maybe, dict):
                return maybe
    return None


def _recommendation(score: float) -> str:
    if score > 0.65:
        return "strong_buy"
    if score > 0.25:
        return "buy"
    if score < -0.65:
        return "strong_sell"
    if score < -0.25:
        return "sell"
    return "hold"


def _weighted_sentiment(breakdown: Dict[str, float]) -> float:
    entries = [
        (label, float(score))
        for label, score in breakdown.items()
        if label in _SOURCE_WEIGHTS
    ]
    if not entries:
        return 0.0
    total_weight = sum(_SOURCE_WEIGHTS[label] for label, _ in entries)
    if total_weight <= 0:
        return 0.0
    weighted = sum(score * (_SOURCE_WEIGHTS[label] / total_weight) for label, score in entries)
    return float(weighted)


class MCPToolRouterClient:
    def __init__(self, command: str, timeout_seconds: int = _REQUEST_TIMEOUT_DEFAULT) -> None:
        self.command = command
        self.timeout_seconds = max(2, int(timeout_seconds or _REQUEST_TIMEOUT_DEFAULT))
        self._proc: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _ensure_process(self) -> None:
        if self._proc and self._proc.returncode is None:
            return
        args = _parse_command(self.command)
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._initialized = False

    async def _request(self, method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        await self._ensure_process()
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("mcp_process_unavailable")

        self._request_id += 1
        req_id = self._request_id
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
        self._proc.stdin.write((json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

        raw = await asyncio.wait_for(self._proc.stdout.readline(), timeout=float(self.timeout_seconds))
        if not raw:
            raise RuntimeError("mcp_no_response")
        response = json.loads(raw.decode("utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("mcp_invalid_response")
        if response.get("id") != req_id:
            raise RuntimeError("mcp_request_mismatch")
        if "error" in response:
            raise RuntimeError(str((response.get("error") or {}).get("message") or "mcp_error"))
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("mcp_missing_result")
        return result

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        await self._request(
            _INITIALIZE_METHOD,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "tickermaster-backend", "version": "0.1.0"},
            },
        )
        await self._request(_TOOLS_LIST_METHOD, {})
        self._initialized = True

    async def plan(
        self,
        *,
        manager_prompt: str,
        available_tools: List[str],
        available_sources: List[str],
        market_state: Dict[str, Any],
        event_hint: str,
    ) -> Dict[str, Any] | None:
        async with self._lock:
            try:
                await self._ensure_initialized()
                result = await self._request(
                    _TOOLS_CALL_METHOD,
                    {
                        "name": _TOOL_NAME,
                        "arguments": {
                            "manager_prompt": manager_prompt,
                            "available_tools": available_tools,
                            "available_sources": available_sources,
                            "market_state": market_state,
                            "event_hint": event_hint,
                        },
                    },
                )
                structured = _extract_structured_content(result)
                if isinstance(structured, dict):
                    return structured
            except Exception:
                self._initialized = False
                return None
            return None

    async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        async with self._lock:
            try:
                await self._ensure_initialized()
                result = await self._request(
                    _TOOLS_CALL_METHOD,
                    {
                        "name": str(name or "").strip(),
                        "arguments": arguments or {},
                    },
                )
                structured = _extract_structured_content(result)
                if isinstance(structured, dict):
                    return structured
            except Exception:
                self._initialized = False
                return None
            return None

    async def close(self) -> None:
        async with self._lock:
            proc = self._proc
            self._proc = None
            self._initialized = False
            if proc is None:
                return
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.5)
                except Exception:
                    proc.kill()
                    with contextlib.suppress(Exception):
                        await proc.wait()


_CLIENT: MCPToolRouterClient | None = None
_CLIENT_LOCK = asyncio.Lock()


async def _get_client(settings: Settings) -> MCPToolRouterClient:
    global _CLIENT
    async with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = MCPToolRouterClient(
                command=str(settings.mcp_tracker_server_command or ""),
                timeout_seconds=int(settings.mcp_tracker_timeout_seconds or _REQUEST_TIMEOUT_DEFAULT),
            )
        return _CLIENT


async def plan_tracker_tools_via_mcp(
    settings: Settings,
    *,
    manager_prompt: str,
    available_tools: List[str],
    available_sources: List[str],
    market_state: Dict[str, Any] | None = None,
    event_hint: str = "cycle",
) -> Dict[str, Any] | None:
    if not bool(settings.mcp_tracker_router_enabled):
        return None
    client = await _get_client(settings)
    return await client.plan(
        manager_prompt=manager_prompt,
        available_tools=available_tools,
        available_sources=available_sources,
        market_state=market_state or {},
        event_hint=event_hint,
    )


async def call_tracker_tool_via_mcp(
    settings: Settings,
    *,
    tool_name: str,
    arguments: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if not bool(settings.mcp_tracker_router_enabled):
        return None
    client = await _get_client(settings)
    return await client.call_tool(name=tool_name, arguments=arguments or {})


async def collect_tracker_research_via_mcp(
    settings: Settings,
    *,
    ticker: str,
    manager_prompt: str,
    tools: List[str] | None = None,
    sources: List[str] | None = None,
    market_state: Dict[str, Any] | None = None,
    timeframe: str = "7d",
    event_hint: str = "cycle",
) -> Dict[str, Any] | None:
    if not bool(settings.mcp_tracker_router_enabled):
        return None

    symbol = str(ticker or "").upper().strip()
    if not symbol:
        return None

    normalized_tools = [str(item).strip().lower() for item in (tools or []) if str(item).strip()]
    normalized_sources = [str(item).strip().lower() for item in (sources or []) if str(item).strip()]
    tool_set = set(normalized_tools)
    source_set = set(normalized_sources)

    needs_reddit = "reddit" in source_set
    needs_x = "x" in source_set
    needs_prediction = "prediction_markets" in source_set or "prediction_markets" in tool_set
    needs_perplexity = (
        "perplexity" in source_set
        or "news" in tool_set
        or "sentiment" in tool_set
    )
    needs_deep = "deep" in source_set or "deep_research" in tool_set

    breakdown: Dict[str, float] = {}
    summaries: Dict[str, str] = {}
    prediction_markets: List[Dict[str, Any]] = []
    debug_payload: Dict[str, Any] = {
        "event_hint": event_hint,
        "ticker": symbol,
        "timeframe": timeframe,
        "tools": normalized_tools,
        "sources": normalized_sources,
        "calls": {},
        "manager_prompt": manager_prompt,
        "market_state": market_state or {},
    }

    if needs_reddit:
        reddit = await call_tracker_tool_via_mcp(
            settings,
            tool_name=_TOOL_REDDIT,
            arguments={"ticker": symbol, "limit": 18},
        )
        if isinstance(reddit, dict):
            debug_payload["calls"]["reddit"] = reddit
            try:
                breakdown["Reddit API"] = float(reddit.get("score", 0.0) or 0.0)
            except Exception:
                breakdown["Reddit API"] = 0.0
            summaries["Reddit API"] = str(reddit.get("summary") or "")

    if needs_x:
        x_payload = await call_tracker_tool_via_mcp(
            settings,
            tool_name=_TOOL_X,
            arguments={"ticker": symbol, "limit": 25},
        )
        if isinstance(x_payload, dict):
            debug_payload["calls"]["x"] = x_payload
            try:
                breakdown["X API"] = float(x_payload.get("score", 0.0) or 0.0)
            except Exception:
                breakdown["X API"] = 0.0
            summaries["X API"] = str(x_payload.get("summary") or "")

    if needs_prediction:
        prediction = await call_tracker_tool_via_mcp(
            settings,
            tool_name=_TOOL_PREDICTION,
            arguments={"ticker": symbol, "limit": 8},
        )
        if isinstance(prediction, dict):
            debug_payload["calls"]["prediction_markets"] = prediction
            rows = prediction.get("markets") if isinstance(prediction.get("markets"), list) else []
            prediction_markets = [row for row in rows if isinstance(row, dict)]
            summaries["Prediction Markets"] = str(prediction.get("summary") or "")

    perplexity_context = (
        f"ticker={symbol} timeframe={timeframe} "
        f"market_state={json.dumps(market_state or {}, ensure_ascii=True)}"
    )[:1800]
    if needs_perplexity:
        perplexity = await call_tracker_tool_via_mcp(
            settings,
            tool_name=_TOOL_PERPLEXITY,
            arguments={
                "ticker": symbol,
                "context": perplexity_context,
                "prompt": manager_prompt or "",
            },
        )
        if isinstance(perplexity, dict):
            debug_payload["calls"]["perplexity"] = perplexity
            try:
                breakdown["Perplexity Sonar"] = float(perplexity.get("score", 0.0) or 0.0)
            except Exception:
                breakdown["Perplexity Sonar"] = 0.0
            summaries["Perplexity Sonar"] = str(perplexity.get("summary") or "")

    if needs_deep:
        deep_payload = await call_tracker_tool_via_mcp(
            settings,
            tool_name=_TOOL_DEEP,
            arguments={
                "ticker": symbol,
                "analysis_goal": manager_prompt or "",
                "context": perplexity_context,
            },
        )
        if isinstance(deep_payload, dict):
            debug_payload["calls"]["deep"] = deep_payload
            summaries["Deep Research"] = str(deep_payload.get("summary") or "")

    if not breakdown and not prediction_markets and not summaries:
        return None

    aggregate = _weighted_sentiment(breakdown)
    response = {
        "aggregate_sentiment": round(float(aggregate), 3),
        "recommendation": _recommendation(aggregate),
        "breakdown": {label: round(float(score), 3) for label, score in breakdown.items()},
        "breakdown_summaries": {label: str(summary or "")[:500] for label, summary in summaries.items()},
        "prediction_markets": prediction_markets,
        "mcp_debug": debug_payload,
    }
    if "perplexity" in debug_payload.get("calls", {}):
        response["investigation"] = str(
            (debug_payload.get("calls", {}).get("perplexity") or {}).get("summary") or ""
        )[:1600]
    return response


async def shutdown_tracker_mcp_router() -> None:
    global _CLIENT
    async with _CLIENT_LOCK:
        client = _CLIENT
        _CLIENT = None
    if client is not None:
        await client.close()
