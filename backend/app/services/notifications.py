from __future__ import annotations

from typing import Any, Dict

from app.config import Settings


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
    return {
        "status": "ready" if settings.poke_recipe_enabled else "disabled",
        "mode": "mcp_recipe",
        "kitchen_url": settings.poke_kitchen_url,
        "recipe_slug": settings.poke_recipe_slug or None,
        "next_step": "Run `npx poke` from project root, then configure this alert payload as a Recipe ingredient in Kitchen.",
        "payload": payload,
    }
