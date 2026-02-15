from __future__ import annotations

import hashlib
import time
from uuid import UUID

from fastapi import Request

from app.services.database import get_supabase

_TOKEN_CACHE_TTL_SECONDS = 300.0
_TOKEN_CACHE_MAX_ENTRIES = 2048
_token_user_cache: dict[str, tuple[float, str | None]] = {}


def _is_uuid(value: str | None) -> bool:
    if not value:
        return False
    try:
        UUID(value)
        return True
    except Exception:
        return False


def _cached_token_user(token: str) -> str | None:
    key = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = time.time()
    cached = _token_user_cache.get(key)
    if cached and (now - cached[0]) < _TOKEN_CACHE_TTL_SECONDS:
        return cached[1]

    user_id: str | None = None
    client = get_supabase()
    if client is not None:
        try:
            response = client.auth.get_user(token)
        except TypeError:
            response = client.auth.get_user(jwt=token)
        except Exception:
            response = None

        user_obj = getattr(response, "user", None) if response is not None else None
        if user_obj is None and isinstance(response, dict):
            user_obj = response.get("user")
        candidate = getattr(user_obj, "id", None) if user_obj is not None else None
        if candidate is None and isinstance(user_obj, dict):
            candidate = user_obj.get("id")
        if isinstance(candidate, str) and _is_uuid(candidate):
            user_id = candidate

    if len(_token_user_cache) >= _TOKEN_CACHE_MAX_ENTRIES:
        # Drop oldest entries to keep token cache bounded in long-lived workers.
        oldest_keys = sorted(_token_user_cache.items(), key=lambda item: item[1][0])[:256]
        for stale_key, _ in oldest_keys:
            _token_user_cache.pop(stale_key, None)
    _token_user_cache[key] = (now, user_id)
    return user_id


def get_user_id_from_request(request: Request) -> str | None:
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header:
        token = auth_header.removeprefix("Bearer ").removeprefix("bearer ").strip()
        if token:
            token_user = _cached_token_user(token)
            if _is_uuid(token_user):
                return token_user

    # Prefer explicit header from frontend/session middleware.
    header_user = request.headers.get("x-user-id")
    if _is_uuid(header_user):
        return header_user

    return None
