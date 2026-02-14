from __future__ import annotations

from typing import Any, Dict, List

import httpx

from app.config import Settings


async def fetch_kalshi_markets(ticker: str, settings: Settings) -> List[Dict[str, Any]]:
    if not settings.kalshi_api_key:
        return [
            {
                "source": "Kalshi",
                "market": f"Will {ticker.upper()} close higher this week?",
                "yes_price": 0.57,
                "no_price": 0.43,
                "link": "https://kalshi.com/markets",
                "note": "Demo value because KALSHI_API_KEY is not set.",
            }
        ]

    headers = {
        "Authorization": f"Bearer {settings.kalshi_api_key}",
        "Content-Type": "application/json",
    }
    params = {"search": ticker.upper(), "limit": 5}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://trading-api.kalshi.com/trade-api/v2/markets", headers=headers, params=params)
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("markets", [])
            return [
                {
                    "source": "Kalshi",
                    "market": item.get("title", item.get("ticker", "Unknown")),
                    "yes_price": item.get("yes_bid") or item.get("yes_ask"),
                    "no_price": item.get("no_bid") or item.get("no_ask"),
                    "link": f"https://kalshi.com/markets/{item.get('ticker', '')}",
                }
                for item in items[:5]
            ]
    except Exception as exc:
        return [
            {
                "source": "Kalshi",
                "market": f"Kalshi fetch failed for {ticker.upper()}",
                "error": str(exc),
                "link": "https://kalshi.com/markets",
            }
        ]


async def fetch_polymarket_markets(ticker: str, settings: Settings) -> List[Dict[str, Any]]:
    headers = {}
    if settings.polymarket_api_key:
        headers["Authorization"] = f"Bearer {settings.polymarket_api_key}"

    params = {"limit": 8, "closed": "false", "search": ticker.upper()}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://gamma-api.polymarket.com/markets", headers=headers, params=params)
            resp.raise_for_status()
            payload = resp.json()
            results = payload if isinstance(payload, list) else payload.get("data", [])
            if not results:
                raise ValueError("No matching Polymarket contracts")
            return [
                {
                    "source": "Polymarket",
                    "market": item.get("question") or item.get("title", "Unknown market"),
                    "probability": item.get("probability") or item.get("outcomePrices"),
                    "volume": item.get("volumeNum") or item.get("volume"),
                    "link": f"https://polymarket.com/event/{item.get('slug', '')}",
                }
                for item in results[:5]
            ]
    except Exception as exc:
        return [
            {
                "source": "Polymarket",
                "market": f"Will {ticker.upper()} outperform this month?",
                "probability": 0.52,
                "link": "https://polymarket.com/markets",
                "note": f"Fallback value due to API error: {exc}",
            }
        ]
