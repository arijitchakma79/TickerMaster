from __future__ import annotations

import re
from typing import Any, Dict, List

import httpx

from app.config import Settings


_COMPANY_STOPWORDS = {
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "COMPANY",
    "CO",
    "PLC",
    "LP",
    "LTD",
    "LIMITED",
    "GROUP",
    "HOLDINGS",
    "THE",
    "CLASS",
    "COMMON",
}


def _semantic_terms(ticker: str, company_name: str | None = None) -> set[str]:
    normalized = ticker.upper().strip()
    terms = {normalized}
    collapsed = normalized.replace("-", "").replace(".", "")
    if collapsed:
        terms.add(collapsed)

    if company_name:
        for token in company_name.upper().replace("&", " ").split():
            clean = "".join(ch for ch in token if ch.isalnum())
            if len(clean) < 3 or clean in _COMPANY_STOPWORDS:
                continue
            terms.add(clean)
    return {term for term in terms if term}


def _market_text(item: Dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key, "") or "")
        for key in ("title", "question", "subtitle", "description", "ticker", "slug")
    ).upper()


def _semantic_relevance(item: Dict[str, Any], ticker: str, company_name: str | None = None) -> float:
    text = _market_text(item)
    if not text:
        return 0.0

    terms = _semantic_terms(ticker, company_name)
    ticker_token = ticker.upper().strip()
    tokens = set(re.findall(r"[A-Z0-9]+", text))
    score = 0.0
    matched = 0

    for term in terms:
        if term in tokens:
            matched += 1
            if term == ticker_token or term == ticker_token.replace("-", ""):
                score += 4.0
            else:
                score += 1.4

    if company_name:
        company_tokens = [token for token in re.findall(r"[A-Z0-9]+", company_name.upper()) if len(token) >= 3]
        overlap = sum(1 for token in company_tokens if token in tokens)
        if overlap >= 2:
            score += 1.2

    if "/" in text and ticker_token in tokens:
        score += 0.6
    if matched >= 2:
        score += 0.8
    return score


async def fetch_kalshi_markets(ticker: str, settings: Settings, company_name: str | None = None) -> List[Dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    params = {"limit": 20, "status": "open"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Kalshi migrated reads to api.elections.kalshi.com
            resp = await client.get("https://api.elections.kalshi.com/trade-api/v2/markets", headers=headers, params=params)
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("markets", [])
            scored: list[tuple[float, Dict[str, Any]]] = []
            for item in items:
                relevance = _semantic_relevance(item, ticker, company_name)
                if relevance <= 0:
                    continue
                scored.append((relevance, item))

            scored.sort(key=lambda pair: pair[0], reverse=True)
            if not scored:
                return []

            return [
                {
                    "source": "Kalshi",
                    "market": item.get("title", item.get("ticker", "Unknown")),
                    "yes_price": item.get("yes_bid") or item.get("yes_ask"),
                    "no_price": item.get("no_bid") or item.get("no_ask"),
                    "link": f"https://kalshi.com/markets/{item.get('ticker', '')}",
                    "relevance_score": round(score, 3),
                }
                for score, item in scored[:5]
            ]
    except Exception:
        return []


async def fetch_polymarket_markets(ticker: str, settings: Settings, company_name: str | None = None) -> List[Dict[str, Any]]:
    headers = {}
    queries = [ticker.upper()]
    if company_name and company_name.strip():
        queries.append(company_name.strip())

    try:
        merged: dict[str, Dict[str, Any]] = {}
        async with httpx.AsyncClient(timeout=15.0) as client:
            for query in queries:
                params = {"limit": 40, "closed": "false", "search": query}
                resp = await client.get(f"{settings.polymarket_gamma_url.rstrip('/')}/markets", headers=headers, params=params)
                resp.raise_for_status()
                payload = resp.json()
                results = payload if isinstance(payload, list) else payload.get("data", [])
                for item in results or []:
                    key = str(item.get("id") or item.get("slug") or item.get("question") or item.get("title") or "")
                    if not key:
                        continue
                    merged[key] = item

            scored: list[tuple[float, Dict[str, Any]]] = []
            for item in merged.values():
                relevance = _semantic_relevance(item, ticker, company_name)
                if relevance <= 0:
                    continue
                scored.append((relevance, item))

            scored.sort(key=lambda pair: pair[0], reverse=True)
            if not scored:
                return []

            return [
                {
                    "source": "Polymarket",
                    "market": item.get("question") or item.get("title", "Unknown market"),
                    "probability": item.get("probability") or item.get("outcomePrices"),
                    "volume": item.get("volumeNum") or item.get("volume"),
                    "link": f"https://polymarket.com/event/{item.get('slug', '')}",
                    "relevance_score": round(score, 3),
                }
                for score, item in scored[:5]
            ]
    except Exception:
        return []
