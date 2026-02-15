from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request):
    settings = request.app.state.settings
    return {
        "ok": True,
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/integrations")
async def integrations(request: Request):
    settings = request.app.state.settings
    return {
        "openai": bool(settings.openai_api_key),
        "elevenlabs": bool(settings.eleven_labs_api_key),
        "openrouter": bool(settings.openrouter_api_key),
        "perplexity": bool(settings.perplexity_api_key),
        "x": bool(settings.x_api_bearer_token),
        "reddit": bool(settings.reddit_client_id or settings.reddit_user_agent),
        "kalshi": bool(settings.kalshi_api_key),
        "polymarket": True,
        "modal": bool(settings.modal_token_id),
        "twilio": bool(settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_phone_number),
        "poke_recipe": bool(settings.poke_recipe_enabled),
        "cerebras": bool(settings.cerebras_api_key),
        "nvidia_nim": bool(settings.nvidia_nim_api_key),
    }
