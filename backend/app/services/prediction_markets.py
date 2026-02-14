from __future__ import annotations

from typing import Any, Dict, List

import httpx

from app.config import Settings


async def fetch_kalshi_markets(ticker: str, settings: Settings) -> List[Dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    params = {"limit": 20, "status": "open"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Kalshi migrated reads to api.elections.kalshi.com
            resp = await client.get("https://api.elections.kalshi.com/trade-api/v2/markets", headers=headers, params=params)
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("markets", [])
            filtered = []
            query = ticker.upper()
            finance_keywords = [
                "FED",
                "RATE",
                "INFLATION",
                "RECESSION",
                "GDP",
                "UNEMPLOYMENT",
                "NASDAQ",
                "S&P",
                "TREASURY",
                "CPI",
                "STOCK",
            ]
            for item in items:
                title = str(item.get("title", ""))
                upper_title = title.upper()
                if (
                    query in upper_title
                    or query in str(item.get("ticker", "")).upper()
                    or any(keyword in upper_title for keyword in finance_keywords)
                ):
                    filtered.append(item)
            if not filtered:
                return []
            return [
                {
                    "source": "Kalshi",
                    "market": item.get("title", item.get("ticker", "Unknown")),
                    "yes_price": item.get("yes_bid") or item.get("yes_ask"),
                    "no_price": item.get("no_bid") or item.get("no_ask"),
                    "link": f"https://kalshi.com/markets/{item.get('ticker', '')}",
                }
                for item in filtered[:5]
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
    params = {"limit": 20, "closed": "false", "search": ticker.upper()}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{settings.polymarket_gamma_url.rstrip('/')}/markets", headers=headers, params=params)
            resp.raise_for_status()
            payload = resp.json()
            results = payload if isinstance(payload, list) else payload.get("data", [])
            if not results:
                raise ValueError("No matching Polymarket contracts")
            finance_keywords = [
                "fed",
                "rate",
                "inflation",
                "recession",
                "gdp",
                "s&p",
                "nasdaq",
                "bitcoin",
                "treasury",
                "unemployment",
                "stock",
            ]
            filtered = []
            for item in results:
                question = str(item.get("question", "")).lower()
                if ticker.lower() in question or any(keyword in question for keyword in finance_keywords):
                    filtered.append(item)
            if not filtered:
                return []
            return [
                {
                    "source": "Polymarket",
                    "market": item.get("question") or item.get("title", "Unknown market"),
                    "probability": item.get("probability") or item.get("outcomePrices"),
                    "volume": item.get("volumeNum") or item.get("volume"),
                    "link": f"https://polymarket.com/event/{item.get('slug', '')}",
                }
                for item in filtered[:5]
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
