from __future__ import annotations

from uuid import UUID

from fastapi import Request


def _is_uuid(value: str | None) -> bool:
    if not value:
        return False
    try:
        UUID(value)
        return True
    except Exception:
        return False


def get_user_id_from_request(request: Request) -> str | None:
    # Prefer explicit header from frontend/session middleware.
    header_user = request.headers.get("x-user-id")
    if _is_uuid(header_user):
        return header_user

    # Optional query fallback for local testing.
    query_user = request.query_params.get("user_id")
    if _is_uuid(query_user):
        return query_user

    return None
