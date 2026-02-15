from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings


def main() -> int:
    settings = get_settings()

    required = {
        "SUPABASE_URL": settings.supabase_url,
        "SUPABASE_SERVICE_KEY": settings.supabase_service_key,
        "FINNHUB_API_KEY": settings.finnhub_api_key,
        "PERPLEXITY_API_KEY": settings.perplexity_api_key,
    }
    optional = {
        "ALPACA_API_KEY": settings.alpaca_api_key,
        "ALPACA_API_SECRET": settings.alpaca_api_secret,
        "OPENAI_API_KEY": settings.openai_api_key,
        "OPENROUTER_API_KEY": settings.openrouter_api_key,
        "BROWSERBASE_API_KEY": settings.browserbase_api_key,
        "BROWSERBASE_PROJECT_ID": settings.browserbase_project_id,
        "MODAL_TOKEN_ID": settings.modal_token_id,
        "MODAL_TOKEN_SECRET": settings.modal_token_secret,
    }

    missing_required = [name for name, value in required.items() if not str(value or "").strip()]

    print("Environment check")
    print("=================")
    for name, value in required.items():
        print(f"[{'ok' if value else 'missing'}] {name} (required)")
    for name, value in optional.items():
        print(f"[{'ok' if value else 'missing'}] {name} (optional)")

    if missing_required:
        print("\nMissing required environment variables:")
        for item in missing_required:
            print(f"- {item}")
        return 1

    print("\nAll required environment variables are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
