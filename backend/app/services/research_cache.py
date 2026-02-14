from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.database import get_supabase


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_cached_research(symbol: str, data_type: str) -> dict[str, Any] | None:
    client = get_supabase()
    if client is None:
        return None

    try:
        resp = (
            client.table("research_cache")
            .select("id,data,expires_at")
            .eq("symbol", symbol.upper())
            .eq("data_type", data_type)
            .single()
            .execute()
        )
        row = resp.data or {}
        expires_at = row.get("expires_at")
        if not expires_at:
            return None
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if expiry < _utc_now():
            return None
        data = row.get("data")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def set_cached_research(symbol: str, data_type: str, data: dict[str, Any], ttl_minutes: int = 15) -> None:
    client = get_supabase()
    if client is None:
        return

    payload = {
        "symbol": symbol.upper(),
        "data_type": data_type,
        "data": data,
        "expires_at": (_utc_now() + timedelta(minutes=ttl_minutes)).isoformat(),
    }

    try:
        client.table("research_cache").upsert(payload, on_conflict="symbol,data_type").execute()
    except Exception:
        return
