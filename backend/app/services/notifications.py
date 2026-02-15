from __future__ import annotations

import re
from typing import Any, Dict

import httpx

from app.config import Settings

POKE_ENDPOINT = "https://poke.com/api/v1/inbound-sms/webhook"
_PHONE_RE = re.compile(r"[^\d+]")


def _normalize_phone(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    cleaned = _PHONE_RE.sub("", raw)
    if cleaned.startswith("00"):
        cleaned = f"+{cleaned[2:]}"
    if cleaned and not cleaned.startswith("+"):
        # Default to US E.164 format when country code is omitted.
        if len(cleaned) == 10:
            cleaned = f"+1{cleaned}"
        elif len(cleaned) == 11 and cleaned.startswith("1"):
            cleaned = f"+{cleaned}"
    return cleaned if cleaned.startswith("+") and len(cleaned) >= 8 else ""


async def send_poke_message(settings: Settings, message: str) -> bool:
    if not settings.poke_api_key:
        return False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                POKE_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {settings.poke_api_key}",
                    "Content-Type": "application/json",
                },
                json={"message": message},
            )
            return response.status_code < 300
    except Exception:
        return False


async def send_twilio_message(settings: Settings, to_number: str, message: str) -> Dict[str, Any]:
    account_sid = settings.twilio_account_sid.strip()
    auth_token = settings.twilio_auth_token.strip()
    from_number = _normalize_phone(settings.twilio_from_number)
    target = _normalize_phone(to_number)

    if not (account_sid and auth_token and from_number and target):
        return {
            "attempted": False,
            "delivered": False,
            "error": "Twilio configuration or recipient phone is missing.",
        }

    endpoint = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    try:
        async with httpx.AsyncClient(timeout=12.0, auth=(account_sid, auth_token)) as client:
            response = await client.post(
                endpoint,
                data={
                    "From": from_number,
                    "To": target,
                    "Body": message[:1500],
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            if response.status_code >= 300:
                return {
                    "attempted": True,
                    "delivered": False,
                    "status_code": response.status_code,
                    "error": str(payload.get("message") or response.text or "Twilio request failed."),
                }
            return {
                "attempted": True,
                "delivered": True,
                "sid": payload.get("sid"),
                "status": payload.get("status"),
                "to": payload.get("to") or target,
                "from": payload.get("from") or from_number,
            }
    except Exception as exc:
        return {
            "attempted": True,
            "delivered": False,
            "error": str(exc),
        }


async def prepare_poke_recipe_handoff(
    settings: Settings,
    title: str,
    body: str,
    link: str,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
        "title": title,
        "body": body,
        "link": link,
        "metadata": metadata or {},
    }

    delivered = await send_poke_message(
        settings,
        f"{title}\n{body}\n{link}",
    )

    return {
        "status": "sent" if delivered else ("ready" if settings.poke_recipe_enabled else "disabled"),
        "delivered": delivered,
        "mode": "poke_http" if delivered else "mcp_recipe",
        "kitchen_url": settings.poke_kitchen_url,
        "recipe_slug": settings.poke_recipe_slug or None,
        "next_step": "Run `npx poke` from project root, then configure this alert payload as a Recipe ingredient in Kitchen.",
        "payload": payload,
    }


async def dispatch_alert_notification(
    settings: Settings,
    title: str,
    body: str,
    link: str,
    preferred_channels: list[str] | None = None,
    *,
    to_number: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    channels = [str(item).strip().lower() for item in (preferred_channels or ["twilio", "poke"]) if str(item).strip()]
    if not channels:
        channels = ["twilio", "poke"]

    notification: Dict[str, Any] = {
        "channels": channels,
        "twilio": {"attempted": False, "delivered": False},
        "poke": {"attempted": False, "delivered": False},
        "delivered": False,
    }

    target_number = _normalize_phone(to_number) or _normalize_phone(settings.twilio_default_to_number)
    if "twilio" in channels:
        notification["twilio"] = await send_twilio_message(
            settings,
            target_number,
            f"{title}\n{body}\n{link}".strip(),
        )

    if "poke" in channels:
        notification["poke"]["attempted"] = True
        poke = await prepare_poke_recipe_handoff(
            settings=settings,
            title=title,
            body=body,
            link=link,
            metadata=metadata or {},
        )
        notification["poke"] = {
            "attempted": True,
            "delivered": bool(poke.get("delivered")),
            **poke,
        }

    notification["delivered"] = bool(
        (notification.get("twilio") or {}).get("delivered")
        or (notification.get("poke") or {}).get("delivered")
    )
    return notification
