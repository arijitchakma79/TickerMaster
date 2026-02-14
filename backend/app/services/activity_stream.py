from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque

from app.ws_manager import WSManager

_manager: WSManager | None = None
_recent: Deque[dict[str, Any]] = deque(maxlen=300)


def set_ws_manager(manager: WSManager) -> None:
    global _manager
    _manager = manager


def get_recent_activity_local(limit: int = 50, module: str | None = None) -> list[dict[str, Any]]:
    items = list(_recent)
    if module:
        items = [item for item in items if item.get("module") == module]
    return list(reversed(items[-limit:]))


async def broadcast_activity(event: dict[str, Any]) -> None:
    payload = {
        "channel": "agents",
        "type": "agent_activity",
        "created_at": event.get("created_at") or datetime.now(timezone.utc).isoformat(),
        **event,
    }
    _recent.append(payload)
    if _manager is not None:
        await _manager.broadcast(payload, channel="agents")
