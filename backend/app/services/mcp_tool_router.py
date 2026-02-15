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


def _parse_command(command: str) -> List[str]:
    token = str(command or "").strip()
    if not token:
        script_path = Path(__file__).resolve().parents[1] / "mcp" / "tracker_tools_server.py"
        return [sys.executable, str(script_path)]
    return shlex.split(token, posix=(os.name != "nt"))


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
                structured = result.get("structuredContent")
                if isinstance(structured, dict):
                    return structured
                content = result.get("content")
                if isinstance(content, list) and content:
                    first = content[0] if isinstance(content[0], dict) else {}
                    text = str(first.get("text") or "").strip()
                    if text.startswith("{") and text.endswith("}"):
                        maybe = json.loads(text)
                        if isinstance(maybe, dict):
                            return maybe
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


async def shutdown_tracker_mcp_router() -> None:
    global _CLIENT
    async with _CLIENT_LOCK:
        client = _CLIENT
        _CLIENT = None
    if client is not None:
        await client.close()
