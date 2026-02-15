from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.services.database import get_supabase

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request):
    settings = request.app.state.settings
    db_ok = False
    client = get_supabase()
    if client is not None:
        try:
            client.table("profiles").select("id").limit(1).execute()
            db_ok = True
        except Exception:
            db_ok = False
    return {
        "ok": True,
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        "dependencies": {"database": "ok" if db_ok else "degraded"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/integrations")
async def integrations(request: Request):
    settings = request.app.state.settings
    return {
        "openai": bool(settings.openai_api_key),
        "openrouter": bool(settings.openrouter_api_key),
        "perplexity": bool(settings.perplexity_api_key),
        "finnhub": bool(settings.finnhub_api_key),
        "alpaca": bool(settings.alpaca_api_key and settings.alpaca_api_secret),
        "twelvedata": bool(settings.twelvedata_api_key),
        "x": bool(settings.x_api_bearer_token),
        "reddit": bool(settings.reddit_client_id or settings.reddit_user_agent),
        "kalshi": bool(settings.kalshi_api_key),
        "polymarket": True,
        "modal": bool(settings.modal_token_id),
        "poke_recipe": bool(settings.poke_recipe_enabled),
        "cerebras": bool(settings.cerebras_api_key),
        "nvidia_nim": bool(settings.nvidia_nim_api_key),
    }
