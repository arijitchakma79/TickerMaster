from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, Dict, Set

from fastapi import WebSocket


class WSManager:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._channel_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channels: set[str] | None = None) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
            for channel in channels or {"global"}:
                self._channel_connections[channel].add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
            for subscribers in self._channel_connections.values():
                subscribers.discard(websocket)

    async def broadcast(self, event: Dict[str, Any], channel: str = "global") -> None:
        payload = json.dumps(event)
        async with self._lock:
            targets = list(self._channel_connections.get(channel, set()) | self._channel_connections.get("global", set()))

        stale: list[WebSocket] = []
        for socket in targets:
            try:
                await socket.send_text(payload)
            except Exception:
                stale.append(socket)

        for socket in stale:
            await self.disconnect(socket)

    async def broadcast_many(self, events: list[Dict[str, Any]], channel: str = "global") -> None:
        for event in events:
            await self.broadcast(event, channel=channel)
