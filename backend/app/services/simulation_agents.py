from __future__ import annotations

from typing import Any

from app.services.database import get_supabase
from app.services.research_cache import get_cached_research, set_cached_research

_LOCAL_SIM_AGENTS: dict[str, list[dict[str, Any]]] = {}
_FALLBACK_DATA_TYPE = "simulation_agents:v1"
_FALLBACK_TTL_MINUTES = 365 * 24 * 60


def _sanitize_entry(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    config = raw.get("config")
    if not isinstance(config, dict):
        return None
    name = str(config.get("name") or "").strip()
    if not name:
        return None
    clean_config = dict(config)
    clean_config["name"] = name

    out: dict[str, Any] = {"config": clean_config}

    icon_emoji = raw.get("iconEmoji")
    if isinstance(icon_emoji, str) and icon_emoji.strip():
        out["iconEmoji"] = icon_emoji.strip()[:8]

    editor = raw.get("editor")
    if isinstance(editor, dict):
        clean_editor = {
            "risk": int(editor.get("risk", 50)),
            "tempo": int(editor.get("tempo", 50)),
            "style": int(editor.get("style", 50)),
            "news": int(editor.get("news", 50)),
        }
        out["editor"] = {
            "risk": max(0, min(100, clean_editor["risk"])),
            "tempo": max(0, min(100, clean_editor["tempo"])),
            "style": max(0, min(100, clean_editor["style"])),
            "news": max(0, min(100, clean_editor["news"])),
        }
    return out


def _normalize_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for raw in entries:
        item = _sanitize_entry(raw)
        if not item:
            continue
        name = str(item["config"]["name"])
        if name in seen:
            continue
        seen.add(name)
        out.append(item)
    return out


def _read_fallback_store(user_id: str) -> list[dict[str, Any]]:
    cached = get_cached_research(user_id, _FALLBACK_DATA_TYPE)
    if not isinstance(cached, dict):
        return []
    rows = cached.get("entries")
    if not isinstance(rows, list):
        return []
    return _normalize_entries([row for row in rows if isinstance(row, dict)])


def _write_fallback_store(user_id: str, entries: list[dict[str, Any]]) -> None:
    payload = {"user_id": user_id, "entries": _normalize_entries(entries)}
    set_cached_research(
        user_id,
        _FALLBACK_DATA_TYPE,
        payload,
        ttl_minutes=_FALLBACK_TTL_MINUTES,
    )


def list_simulation_agents(user_id: str) -> list[dict[str, Any]]:
    fallback = _read_fallback_store(user_id)
    client = get_supabase()
    if client is None:
        if fallback:
            _LOCAL_SIM_AGENTS[user_id] = list(fallback)
            return fallback
        return list(_LOCAL_SIM_AGENTS.get(user_id, []))
    try:
        rows = (
            client.table("simulation_agents")
            .select("agent_name,config,icon_emoji,editor")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
            .data
            or []
        )
    except Exception:
        if fallback:
            _LOCAL_SIM_AGENTS[user_id] = list(fallback)
            return fallback
        return list(_LOCAL_SIM_AGENTS.get(user_id, []))

    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        config = row.get("config")
        if not isinstance(config, dict):
            continue
        agent_name = str(row.get("agent_name") or config.get("name") or "").strip()
        if not agent_name:
            continue
        clean: dict[str, Any] = {"config": {**config, "name": agent_name}}
        icon_emoji = row.get("icon_emoji")
        if isinstance(icon_emoji, str) and icon_emoji.strip():
            clean["iconEmoji"] = icon_emoji.strip()[:8]
        editor = row.get("editor")
        if isinstance(editor, dict):
            clean["editor"] = editor
        normalized = _sanitize_entry(clean)
        if normalized:
            out.append(normalized)
    normalized_rows = _normalize_entries(out)
    if not normalized_rows and fallback:
        _LOCAL_SIM_AGENTS[user_id] = list(fallback)
        return fallback
    _LOCAL_SIM_AGENTS[user_id] = list(normalized_rows)
    if normalized_rows:
        _write_fallback_store(user_id, normalized_rows)
    return normalized_rows


def set_simulation_agents(user_id: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = _normalize_entries(entries)
    _LOCAL_SIM_AGENTS[user_id] = list(clean)
    _write_fallback_store(user_id, clean)

    client = get_supabase()
    if client is None:
        return clean
    try:
        if clean:
            payload = [
                {
                    "user_id": user_id,
                    "agent_name": str(item["config"]["name"]),
                    "config": item["config"],
                    "icon_emoji": item.get("iconEmoji"),
                    "editor": item.get("editor"),
                }
                for item in clean
            ]
            client.table("simulation_agents").upsert(payload, on_conflict="user_id,agent_name").execute()

        rows = (
            client.table("simulation_agents")
            .select("agent_name")
            .eq("user_id", user_id)
            .execute()
            .data
            or []
        )
        keep_names = {str(item["config"]["name"]) for item in clean}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("agent_name") or "").strip()
            if not name or name in keep_names:
                continue
            client.table("simulation_agents").delete().eq("user_id", user_id).eq("agent_name", name).execute()
    except Exception:
        return clean
    return list_simulation_agents(user_id)


def delete_simulation_agent(user_id: str, agent_name: str) -> list[dict[str, Any]]:
    target = str(agent_name or "").strip()
    if not target:
        return list_simulation_agents(user_id)

    current_local = list(_LOCAL_SIM_AGENTS.get(user_id, []))
    _LOCAL_SIM_AGENTS[user_id] = [
        entry
        for entry in current_local
        if str((entry.get("config") or {}).get("name") or "").strip() != target
    ]
    _write_fallback_store(user_id, _LOCAL_SIM_AGENTS[user_id])

    client = get_supabase()
    if client is None:
        return list(_LOCAL_SIM_AGENTS.get(user_id, []))
    try:
        client.table("simulation_agents").delete().eq("user_id", user_id).eq("agent_name", target).execute()
    except Exception:
        pass
    return list_simulation_agents(user_id)
