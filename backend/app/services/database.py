from __future__ import annotations

from functools import lru_cache

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover
    Client = object  # type: ignore[assignment]
    create_client = None  # type: ignore[assignment]

from app.config import get_settings


@lru_cache(maxsize=1)
def get_supabase() -> Client | None:
    if create_client is None:
        return None
    settings = get_settings()
    if not settings.supabase_url:
        return None

    key = settings.supabase_service_key or settings.supabase_key
    if not key:
        return None

    try:
        return create_client(settings.supabase_url, key)
    except Exception:
        return None
