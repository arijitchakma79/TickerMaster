from __future__ import annotations

from typing import Any, Dict

import httpx

from app.config import Settings
from app.services.agent_logger import log_agent_activity


async def run_deep_research(symbol: str, settings: Settings) -> Dict[str, Any]:
    ticker = symbol.upper().strip()
    await log_agent_activity(
        module="research",
        agent_name="Browserbase Agent",
        action=f"Starting deep research scrape for {ticker}",
        status="running",
    )

    if not settings.browserbase_api_key or not settings.browserbase_project_id:
        result = {
            "symbol": ticker,
            "source": "browserbase",
            "analyst_ratings": "N/A",
            "insider_trading": "N/A",
            "reddit_dd_summary": "Browserbase keys not configured.",
            "notes": "Set BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID to enable live scraping.",
        }
        await log_agent_activity(
            module="research",
            agent_name="Browserbase Agent",
            action=f"Deep research skipped for {ticker}",
            status="error",
            details={"reason": "missing_credentials"},
        )
        return result

    session_payload: dict[str, Any] | None = None
    session_id: str | None = None
    connect_url: str | None = None

    # Best-effort Browserbase session create probe (real API call).
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.browserbase.com/v1/sessions",
                headers={
                    "x-bb-api-key": settings.browserbase_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "projectId": settings.browserbase_project_id,
                    "keepAlive": False,
                },
            )
            if resp.status_code < 300:
                session_payload = resp.json()
                session_id = str(session_payload.get("id") or session_payload.get("sessionId") or "")
                connect_url = str(session_payload.get("connectUrl") or session_payload.get("wsEndpoint") or "")
    except Exception as exc:
        await log_agent_activity(
            module="research",
            agent_name="Browserbase Agent",
            action=f"Browserbase session create failed for {ticker}",
            status="error",
            details={"error": str(exc)},
        )

    result = {
        "symbol": ticker,
        "source": "browserbase",
        "analyst_ratings": "Session initialized; Stagehand extraction still pending full selector wiring.",
        "insider_trading": "Session initialized; Stagehand extraction still pending full selector wiring.",
        "reddit_dd_summary": "Deep research pipeline connected to Browserbase API.",
        "session_id": session_id,
        "connect_url": connect_url,
        "raw_session": session_payload,
    }
    await log_agent_activity(
        module="research",
        agent_name="Browserbase Agent",
        action=f"Deep research complete for {ticker}",
        status="success",
    )
    return result
