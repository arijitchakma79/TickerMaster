from __future__ import annotations

from typing import Iterable, List

from app.services.database import get_supabase


def _clean_symbols(symbols: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        symbol = str(raw or "").upper().strip()
        if not symbol:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def get_watchlist(user_id: str) -> list[str]:
    client = get_supabase()
    if client is None:
        return []
    try:
        rows = (
            client.table("watchlist")
            .select("symbol")
            .eq("user_id", user_id)
            .order("added_at", desc=False)
            .execute()
            .data
            or []
        )
    except Exception:
        return []
    return _clean_symbols(str(row.get("symbol") or "") for row in rows if isinstance(row, dict))


def set_watchlist(user_id: str, symbols: List[str]) -> list[str]:
    clean = _clean_symbols(symbols)
    target = set(clean)
    client = get_supabase()
    if client is None:
        return clean
    try:
        if clean:
            upserts = [{"user_id": user_id, "symbol": symbol} for symbol in clean]
            client.table("watchlist").upsert(upserts, on_conflict="user_id,symbol").execute()
        rows = (
            client.table("watchlist")
            .select("symbol")
            .eq("user_id", user_id)
            .execute()
            .data
            or []
        )
        stale = [
            str(row.get("symbol") or "").upper().strip()
            for row in rows
            if isinstance(row, dict)
            and str(row.get("symbol") or "").upper().strip()
            and str(row.get("symbol") or "").upper().strip() not in target
        ]
        for symbol in stale:
            client.table("watchlist").delete().eq("user_id", user_id).eq("symbol", symbol).execute()
    except Exception:
        return get_watchlist(user_id)
    return clean


def get_favorites(user_id: str) -> list[str]:
    client = get_supabase()
    if client is None:
        return []
    try:
        rows = (
            client.table("favorite_stocks")
            .select("symbol")
            .eq("user_id", user_id)
            .order("added_at", desc=False)
            .execute()
            .data
            or []
        )
    except Exception:
        return []
    return _clean_symbols(str(row.get("symbol") or "") for row in rows if isinstance(row, dict))


def set_favorites(user_id: str, symbols: List[str]) -> list[str]:
    clean = _clean_symbols(symbols)
    target = set(clean)
    client = get_supabase()
    if client is None:
        return clean
    try:
        if clean:
            upserts = [{"user_id": user_id, "symbol": symbol} for symbol in clean]
            client.table("favorite_stocks").upsert(upserts, on_conflict="user_id,symbol").execute()
        rows = (
            client.table("favorite_stocks")
            .select("symbol")
            .eq("user_id", user_id)
            .execute()
            .data
            or []
        )
        stale = [
            str(row.get("symbol") or "").upper().strip()
            for row in rows
            if isinstance(row, dict)
            and str(row.get("symbol") or "").upper().strip()
            and str(row.get("symbol") or "").upper().strip() not in target
        ]
        for symbol in stale:
            client.table("favorite_stocks").delete().eq("user_id", user_id).eq("symbol", symbol).execute()
    except Exception:
        return get_favorites(user_id)
    return clean
