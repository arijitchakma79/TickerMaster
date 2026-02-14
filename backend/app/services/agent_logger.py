from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.activity_stream import broadcast_activity, get_recent_activity_local
from app.services.database import get_supabase


async def log_agent_activity(
    module: str,
    agent_name: str,
    action: str,
    user_id: str | None = None,
    details: dict[str, Any] | None = None,
    status: str = "success",
) -> None:
    client = get_supabase()
    if client is None:
        return

    payload = {
        "user_id": user_id,
        "module": module,
        "agent_name": agent_name,
        "action": action,
        "details": details or {},
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        client.table("agent_activity").insert(payload).execute()
    except Exception:
        pass
    await broadcast_activity(payload)


async def get_recent_activity(limit: int = 50, module: str | None = None) -> list[dict[str, Any]]:
    client = get_supabase()
    if client is None:
        return get_recent_activity_local(limit=limit, module=module)

    try:
        query = client.table("agent_activity").select("*").order("created_at", desc=True).limit(limit)
        if module:
            query = query.eq("module", module)
        data = query.execute().data
        return data if isinstance(data, list) else []
    except Exception:
        return get_recent_activity_local(limit=limit, module=module)
