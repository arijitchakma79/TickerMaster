from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from app.config import get_settings
from app.services.database import get_supabase

logger = logging.getLogger(__name__)


def _ensure_exports_bucket(client: Any, bucket: str) -> bool:
    try:
        existing = client.storage.list_buckets()
        for item in existing or []:
            item_id = str(getattr(item, "id", "") or "")
            if item_id == bucket:
                return True
    except Exception as exc:
        logger.warning("Unable to list Supabase storage buckets before ensure: %s", exc)

    try:
        client.storage.create_bucket(
            bucket,
            options={
                "public": False,
                "file_size_limit": 20_000_000,
                "allowed_mime_types": ["text/csv", "application/json", "text/plain"],
            },
        )
    except Exception as exc:
        logger.warning("Unable to create Supabase bucket '%s': %s", bucket, exc)

    try:
        existing = client.storage.list_buckets()
        for item in existing or []:
            item_id = str(getattr(item, "id", "") or "")
            if item_id == bucket:
                return True
    except Exception as exc:
        logger.warning("Unable to verify Supabase bucket '%s': %s", bucket, exc)
    return False


def ensure_tracker_exports_bucket() -> bool:
    client = get_supabase()
    if client is None:
        return False
    settings = get_settings()
    bucket = settings.supabase_tracker_exports_bucket or "tracker-exports"
    return _ensure_exports_bucket(client, bucket)


def ensure_tracker_memory_bucket() -> bool:
    client = get_supabase()
    if client is None:
        return False
    settings = get_settings()
    bucket = settings.supabase_tracker_memory_bucket or "tracker-memory"
    return _ensure_exports_bucket(client, bucket)


def ensure_tracker_storage_buckets() -> bool:
    exports_ok = ensure_tracker_exports_bucket()
    memory_ok = ensure_tracker_memory_bucket()
    return exports_ok and memory_ok


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value if value is not None else {}, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"


def append_agent_response_csv(
    *,
    user_id: str,
    agent_id: str,
    symbol: str,
    agent_name: str,
    manager_instruction: str,
    response_text: str,
    generated_at: str,
    intent: str,
    parsed_intent: dict[str, Any] | None = None,
    trigger_snapshot: dict[str, Any] | None = None,
    tool_outputs: dict[str, Any] | None = None,
) -> dict[str, str]:
    client = get_supabase()
    if client is None:
        raise RuntimeError("Supabase client unavailable.")

    settings = get_settings()
    bucket = settings.supabase_tracker_exports_bucket or "tracker-exports"
    bucket_ready = _ensure_exports_bucket(client, bucket)
    if not bucket_ready:
        logger.warning("Proceeding with CSV upload without confirmed bucket readiness for '%s'.", bucket)

    path = f"{user_id}/agents/{agent_id}/responses.csv"
    header = [
        "generated_at",
        "user_id",
        "agent_id",
        "symbol",
        "agent_name",
        "intent",
        "manager_instruction",
        "agent_response",
        "parsed_intent_json",
        "trigger_snapshot_json",
        "tool_outputs_json",
    ]
    row = [
        generated_at,
        user_id,
        agent_id,
        symbol,
        agent_name,
        intent,
        manager_instruction,
        response_text,
        _to_json(parsed_intent or {}),
        _to_json(trigger_snapshot or {}),
        _to_json(tool_outputs or {}),
    ]

    content = io.StringIO()
    existing_text = ""
    try:
        existing_text = client.storage.from_(bucket).download(path).decode("utf-8")
    except Exception:
        existing_text = ""

    if existing_text.strip():
        content.write(existing_text.rstrip("\n"))
        content.write("\n")
        writer = csv.writer(content)
        writer.writerow(row)
    else:
        writer = csv.writer(content)
        writer.writerow(header)
        writer.writerow(row)

    payload = content.getvalue().encode("utf-8")
    client.storage.from_(bucket).upload(
        path,
        payload,
        {"content-type": "text/csv", "upsert": True},
    )
    return {"bucket": bucket, "path": path}


def read_agent_response_csv_tail(
    *,
    user_id: str,
    agent_id: str,
    limit: int = 80,
) -> dict[str, Any]:
    client = get_supabase()
    if client is None:
        return {"bucket": None, "path": None, "rows": []}

    settings = get_settings()
    bucket = settings.supabase_tracker_exports_bucket or "tracker-exports"
    path = f"{user_id}/agents/{agent_id}/responses.csv"
    try:
        raw = client.storage.from_(bucket).download(path).decode("utf-8")
    except Exception:
        return {"bucket": bucket, "path": path, "rows": []}

    try:
        reader = csv.DictReader(io.StringIO(raw))
        rows = [dict(row) for row in reader]
    except Exception:
        rows = []
    limit = max(1, min(1000, int(limit)))
    return {"bucket": bucket, "path": path, "rows": rows[-limit:]}


def append_alert_context_csv(
    *,
    user_id: str,
    agent_id: str,
    symbol: str,
    event_type: str,
    generated_at: str,
    alert_id: str | None = None,
    context_summary: str | None = None,
    simulation_requested: bool = False,
    context_payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    client = get_supabase()
    if client is None:
        raise RuntimeError("Supabase client unavailable.")

    settings = get_settings()
    bucket = settings.supabase_tracker_memory_bucket or "tracker-memory"
    bucket_ready = _ensure_exports_bucket(client, bucket)
    if not bucket_ready:
        logger.warning("Proceeding with memory CSV upload without confirmed bucket readiness for '%s'.", bucket)

    path = f"{user_id}/agents/{agent_id}/alert_context.csv"
    header = [
        "generated_at",
        "user_id",
        "agent_id",
        "symbol",
        "event_type",
        "alert_id",
        "simulation_requested",
        "context_summary",
        "context_payload_json",
    ]
    row = [
        generated_at,
        user_id,
        agent_id,
        symbol,
        event_type,
        alert_id or "",
        "true" if simulation_requested else "false",
        context_summary or "",
        _to_json(context_payload or {}),
    ]

    content = io.StringIO()
    existing_text = ""
    try:
        existing_text = client.storage.from_(bucket).download(path).decode("utf-8")
    except Exception:
        existing_text = ""

    if existing_text.strip():
        content.write(existing_text.rstrip("\n"))
        content.write("\n")
        writer = csv.writer(content)
        writer.writerow(row)
    else:
        writer = csv.writer(content)
        writer.writerow(header)
        writer.writerow(row)

    payload = content.getvalue().encode("utf-8")
    client.storage.from_(bucket).upload(
        path,
        payload,
        {"content-type": "text/csv", "upsert": True},
    )
    return {"bucket": bucket, "path": path}


def read_alert_context_csv_tail(
    *,
    user_id: str,
    agent_id: str,
    limit: int = 120,
) -> dict[str, Any]:
    client = get_supabase()
    if client is None:
        return {"bucket": None, "path": None, "rows": []}

    settings = get_settings()
    bucket = settings.supabase_tracker_memory_bucket or "tracker-memory"
    path = f"{user_id}/agents/{agent_id}/alert_context.csv"
    try:
        raw = client.storage.from_(bucket).download(path).decode("utf-8")
    except Exception:
        return {"bucket": bucket, "path": path, "rows": []}

    try:
        reader = csv.DictReader(io.StringIO(raw))
        rows = [dict(row) for row in reader]
    except Exception:
        rows = []
    limit = max(1, min(1000, int(limit)))
    return {"bucket": bucket, "path": path, "rows": rows[-limit:]}


def _download_text(client: Any, bucket: str, path: str) -> str:
    try:
        raw = client.storage.from_(bucket).download(path)
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8")
        return str(raw or "")
    except Exception:
        return ""


def _upload_text(client: Any, bucket: str, path: str, payload: str, content_type: str) -> None:
    client.storage.from_(bucket).upload(
        path,
        payload.encode("utf-8"),
        {"content-type": content_type, "upsert": True},
    )


def append_agent_memory_documents(
    *,
    user_id: str,
    agent_id: str,
    symbol: str,
    generated_at: str,
    event_type: str,
    manager_instruction: str,
    agent_response: str,
    context_payload: dict[str, Any] | None = None,
    mcp_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = get_supabase()
    if client is None:
        raise RuntimeError("Supabase client unavailable.")

    settings = get_settings()
    bucket = settings.supabase_tracker_memory_bucket or "tracker-memory"
    bucket_ready = _ensure_exports_bucket(client, bucket)
    if not bucket_ready:
        logger.warning("Proceeding with memory document upload without confirmed bucket readiness for '%s'.", bucket)

    base_path = f"{user_id}/agents/{agent_id}"
    jsonl_path = f"{base_path}/memory.jsonl"
    txt_path = f"{base_path}/memory.txt"
    payload = {
        "generated_at": generated_at,
        "user_id": user_id,
        "agent_id": agent_id,
        "symbol": symbol,
        "event_type": event_type,
        "manager_instruction": manager_instruction,
        "agent_response": agent_response,
        "context": context_payload or {},
        "mcp_debug": mcp_debug or {},
    }

    existing_jsonl = _download_text(client, bucket, jsonl_path).rstrip("\n")
    next_jsonl = f"{existing_jsonl}\n{json.dumps(payload, ensure_ascii=False)}" if existing_jsonl else json.dumps(payload, ensure_ascii=False)
    _upload_text(client, bucket, jsonl_path, next_jsonl, "application/json")

    instruction = str(manager_instruction or "").strip()
    response = str(agent_response or "").strip()
    existing_txt = _download_text(client, bucket, txt_path).rstrip("\n")
    entry_lines = [
        f"[{generated_at}] {event_type} {symbol}",
        f"Manager: {instruction}" if instruction else "Manager: (none)",
        f"Agent: {response}" if response else "Agent: (none)",
    ]
    txt_entry = "\n".join(entry_lines)
    next_txt = f"{existing_txt}\n\n{txt_entry}" if existing_txt else txt_entry
    _upload_text(client, bucket, txt_path, next_txt, "text/plain")

    return {
        "bucket": bucket,
        "jsonl_path": jsonl_path,
        "txt_path": txt_path,
    }


def read_agent_memory_json_tail(
    *,
    user_id: str,
    agent_id: str,
    limit: int = 120,
) -> dict[str, Any]:
    client = get_supabase()
    if client is None:
        return {"bucket": None, "path": None, "rows": []}

    settings = get_settings()
    bucket = settings.supabase_tracker_memory_bucket or "tracker-memory"
    path = f"{user_id}/agents/{agent_id}/memory.jsonl"
    raw = _download_text(client, bucket, path)
    if not raw.strip():
        return {"bucket": bucket, "path": path, "rows": []}

    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            parsed = json.loads(token)
        except Exception:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    limit = max(1, min(2000, int(limit)))
    return {"bucket": bucket, "path": path, "rows": rows[-limit:]}


def read_agent_memory_text_tail(
    *,
    user_id: str,
    agent_id: str,
    max_chars: int = 16000,
) -> dict[str, Any]:
    client = get_supabase()
    if client is None:
        return {"bucket": None, "path": None, "text": ""}

    settings = get_settings()
    bucket = settings.supabase_tracker_memory_bucket or "tracker-memory"
    path = f"{user_id}/agents/{agent_id}/memory.txt"
    raw = _download_text(client, bucket, path)
    if not raw:
        return {"bucket": bucket, "path": path, "text": ""}
    size = max(500, min(200000, int(max_chars)))
    trimmed = raw[-size:]
    return {"bucket": bucket, "path": path, "text": trimmed}
