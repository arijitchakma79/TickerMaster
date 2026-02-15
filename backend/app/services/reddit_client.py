from __future__ import annotations

import time
from typing import Any

import httpx

from app.config import Settings

_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
_SEARCH_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_FRESH_TTL_SECONDS = 5 * 60
_STALE_TTL_SECONDS = 6 * 60 * 60


def _cache_key(query: str, sort: str, timeframe: str, limit: int) -> str:
    return "|".join([query.strip().lower(), sort.strip().lower(), timeframe.strip().lower(), str(int(limit))])


def _get_cached_posts(key: str, now: float, allow_stale: bool) -> list[dict[str, Any]]:
    hit = _SEARCH_CACHE.get(key)
    if not hit:
        return []
    fetched_at, posts = hit
    age = max(0.0, now - fetched_at)
    if age <= _FRESH_TTL_SECONDS:
        return posts
    if allow_stale and age <= _STALE_TTL_SECONDS:
        return posts
    return []


async def _reddit_app_token(settings: Settings, client: httpx.AsyncClient) -> str:
    client_id = (settings.reddit_client_id or "").strip()
    client_secret = (settings.reddit_client_secret or "").strip()
    if not client_id or not client_secret:
        return ""

    token_key = f"{client_id}:{client_secret}"
    now = time.time()
    cached = _TOKEN_CACHE.get(token_key)
    if cached and cached[1] > now + 30:
        return cached[0]

    headers = {"User-Agent": settings.reddit_user_agent or "TickerMaster/1.0"}
    auth = (client_id, client_secret)
    resp = await client.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=auth,
        data={"grant_type": "client_credentials"},
        headers=headers,
    )
    resp.raise_for_status()
    payload_raw = resp.json()
    payload = payload_raw if isinstance(payload_raw, dict) else {}
    token = str(payload.get("access_token") or "").strip()
    expires_in = int(payload.get("expires_in") or 3600)
    if token:
        _TOKEN_CACHE[token_key] = (token, now + max(60, expires_in))
    return token


async def reddit_search_posts(
    settings: Settings,
    *,
    query: str,
    sort: str = "new",
    timeframe: str = "week",
    limit: int = 10,
    timeout_seconds: float = 12.0,
) -> list[dict[str, Any]]:
    key = _cache_key(query, sort, timeframe, limit)
    now = time.time()
    cached = _get_cached_posts(key, now, allow_stale=False)
    if cached:
        return cached

    headers = {"User-Agent": settings.reddit_user_agent or "TickerMaster/1.0"}
    params = {"q": query, "restrict_sr": "false", "sort": sort, "t": timeframe, "limit": limit}

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, headers=headers) as client:
            token = ""
            try:
                token = await _reddit_app_token(settings, client)
            except Exception:
                token = ""

            if token:
                oauth_headers = dict(headers)
                oauth_headers["Authorization"] = f"Bearer {token}"
                response = await client.get("https://oauth.reddit.com/search", params=params, headers=oauth_headers)
            else:
                response = await client.get("https://www.reddit.com/search.json", params=params)

            if response.status_code in {403, 429}:
                stale = _get_cached_posts(key, now, allow_stale=True)
                if stale:
                    return stale
                return []

            response.raise_for_status()
            payload_raw = response.json()
            payload = payload_raw if isinstance(payload_raw, dict) else {}
    except Exception:
        stale = _get_cached_posts(key, now, allow_stale=True)
        if stale:
            return stale
        return []

    children = payload.get("data", {}).get("children", []) if isinstance(payload, dict) else []
    posts = [child.get("data", {}) for child in children if isinstance(child, dict) and isinstance(child.get("data"), dict)]
    _SEARCH_CACHE[key] = (time.time(), posts)
    return posts
