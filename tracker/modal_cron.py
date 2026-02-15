from __future__ import annotations

import os
from typing import Any

import modal

app = modal.App("tickermaster-tracker")

tracker_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "httpx==0.28.1",
)


@app.function(
    schedule=modal.Period(minutes=2),
    image=tracker_image,
    secrets=[modal.Secret.from_name("tickermaster-secrets")],
    timeout=120,
)
def poll_all_agents() -> dict[str, Any]:
    import httpx

    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
    cron_token = os.environ.get("TRACKER_CRON_TOKEN", "").strip()
    headers = {"Content-Type": "application/json"}
    if cron_token:
        headers["x-cron-token"] = cron_token

    try:
        with httpx.Client(timeout=30.0, headers=headers) as client:
            response = client.post(f"{backend_url}/tracker/poll")
            response.raise_for_status()
            payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "backend_url": backend_url}

    alerts = payload.get("alerts_triggered", []) if isinstance(payload, dict) else []
    tickers = payload.get("tickers", []) if isinstance(payload, dict) else []
    return {
        "ok": True,
        "backend_url": backend_url,
        "polled_tickers": len(tickers) if isinstance(tickers, list) else 0,
        "alerts": len(alerts) if isinstance(alerts, list) else 0,
    }


@app.local_entrypoint()
def main():
    print(poll_all_agents.local())
