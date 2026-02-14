from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.database import get_supabase


def create_simulation_record(config: dict[str, Any], user_id: str | None = None) -> str | None:
    client = get_supabase()
    if client is None:
        return None
    payload = {
        "user_id": user_id,
        "config": config,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        data = client.table("simulations").insert(payload).execute().data
        if data and isinstance(data, list):
            return str(data[0].get("id"))
    except Exception:
        return None
    return None


def complete_simulation_record(record_id: str, results: dict[str, Any], status: str = "completed") -> None:
    client = get_supabase()
    if client is None:
        return
    try:
        client.table("simulations").update(
            {
                "status": status,
                "results": results,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", record_id).execute()
    except Exception:
        return
