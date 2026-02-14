from __future__ import annotations

from typing import Any, Dict

import httpx

from app.config import Settings

POKE_ENDPOINT = "https://poke.com/api/v1/inbound-sms/webhook"


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
