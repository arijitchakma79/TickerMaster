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
                "allowed_mime_types": ["text/csv"],
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
