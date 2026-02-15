from __future__ import annotations

import logging
import re
from typing import Any, Dict, List
from urllib.parse import quote, urlparse

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


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

_KALSHI_QUERY_LIMIT = 120
_POLYMARKET_QUERY_LIMIT = 50
_THEME_TOKEN_STOPWORDS = {"stock", "market", "markets", "price", "prices", "sales", "earnings", "interest", "rates", "demand", "supply", "revenue"}

_FINANCE_CONTEXT_TERMS = {
    "EARNINGS",
    "EPS",
    "GUIDE",
    "REVENUE",
    "GUIDANCE",
    "SALES",
    "SELL",
    "SOLD",
    "DELIVERIES",
    "DELIVER",
    "DELIVERS",
    "DELIVERY",
    "VEHICLE",
    "VEHICLES",
    "CARS",
    "UNITS",
    "QUARTER",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "STOCK",
    "SHARE",
    "SHARES",
    "VALUATION",
    "PROFIT",
    "LOSS",
    "MARGIN",
    "CHIP",
    "DATACENTER",
    # Note: Removed "MARKET", "PRICE", "AI", "CLOUD" as these are too generic
    # and cause false positives on prediction market platforms
}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            clean = value.strip().replace("%", "").replace(",", "")
            if not clean:
                return None
            value = clean
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_percent(value: Any) -> float | None:
    raw = _safe_float(value)
    if raw is None:
        return None
    if raw < 0:
        return None
    if raw <= 1:
        raw = raw * 100.0
    elif raw > 100 and raw <= 10000:
        # Some feeds return basis points / cents-style values.
        raw = raw / 100.0
    if raw > 100:
        return None
    return round(raw, 2)


def _extract_kalshi_prices(item: Dict[str, Any]) -> tuple[float | None, float | None]:
    yes_bid = _to_percent(item.get("yes_bid") or item.get("yesBid"))
    yes_ask = _to_percent(item.get("yes_ask") or item.get("yesAsk"))
    no_bid = _to_percent(item.get("no_bid") or item.get("noBid"))
    no_ask = _to_percent(item.get("no_ask") or item.get("noAsk"))

    yes_price = _to_percent(item.get("yes_price") or item.get("yesPrice"))
    no_price = _to_percent(item.get("no_price") or item.get("noPrice"))

    if yes_price is None:
        if yes_bid is not None and yes_ask is not None:
            yes_price = round((yes_bid + yes_ask) / 2.0, 2)
        else:
            yes_price = yes_bid if yes_bid is not None else yes_ask
    if no_price is None:
        if no_bid is not None and no_ask is not None:
            no_price = round((no_bid + no_ask) / 2.0, 2)
        else:
            no_price = no_bid if no_bid is not None else no_ask

    return yes_price, no_price


def _extract_polymarket_prices(item: Dict[str, Any]) -> tuple[float | None, float | None]:
    yes_price = _to_percent(item.get("yes_price") or item.get("yesPrice") or item.get("probability"))
    no_price = _to_percent(item.get("no_price") or item.get("noPrice"))

    outcome_prices = item.get("outcomePrices")
    prices_list: list[float] = []
    if isinstance(outcome_prices, list):
        prices_list = [price for price in (_to_percent(value) for value in outcome_prices) if price is not None]
    elif isinstance(outcome_prices, str):
        text = outcome_prices.strip()
        if text:
            parsed: Any = None
            try:
                import json
                parsed = json.loads(text)
            except Exception:
                parsed = [part.strip() for part in text.split(",")]
            if isinstance(parsed, list):
                prices_list = [price for price in (_to_percent(value) for value in parsed) if price is not None]

    outcomes = item.get("outcomes")
    if yes_price is None and no_price is None and prices_list:
        if isinstance(outcomes, list) and len(outcomes) == len(prices_list):
            mapped: Dict[str, float] = {}
            for idx, label in enumerate(outcomes):
                key = str(label or "").strip().lower()
                mapped[key] = prices_list[idx]
            if "yes" in mapped:
                yes_price = mapped["yes"]
                no_price = mapped.get("no")
            elif "no" in mapped:
                no_price = mapped["no"]
                yes_price = mapped.get("yes")

        if yes_price is None and prices_list:
            yes_price = prices_list[0]
        if no_price is None and len(prices_list) > 1:
            no_price = prices_list[1]

    return yes_price, no_price


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

def _company_terms(company_name: str | None) -> set[str]:
    if not company_name:
        return set()
    terms: set[str] = set()
    for token in re.findall(r"[A-Z0-9]+", company_name.upper()):
        if len(token) < 3 or token in _COMPANY_STOPWORDS:
            continue
        terms.add(token)
    return terms


def _company_phrases(company_name: str | None) -> list[str]:
    if not company_name:
        return []
    lowered = company_name.lower().strip()
    if not lowered:
        return []

    # Strip common legal suffixes for phrase matching.
    tokens = [t for t in re.findall(r"[a-z0-9]+", lowered) if t]
    stripped: list[str] = []
    for token in tokens:
        if token.upper() in _COMPANY_STOPWORDS:
            continue
        stripped.append(token)
    if not stripped:
        return []

    phrases: list[str] = []
    full = " ".join(stripped)
    if len(full) >= 5:
        phrases.append(full)
    if len(stripped) >= 2:
        phrases.append(" ".join(stripped[:2]))
    if len(stripped) >= 3:
        phrases.append(" ".join(stripped[:3]))
    # Keep order, unique only.
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
    return out


def _ticker_regexes(ticker: str) -> list[re.Pattern[str]]:
    token = ticker.upper().strip()
    if not token:
        return []
    escaped = re.escape(token)
    collapsed = re.escape(token.replace(".", "").replace("-", ""))
    patterns = [
        re.compile(rf"(^|[^A-Z0-9])\${escaped}([^A-Z0-9]|$)"),
        re.compile(rf"(^|[^A-Z0-9]){escaped}([^A-Z0-9]|$)"),
    ]
    if collapsed and collapsed != escaped:
        patterns.append(re.compile(rf"(^|[^A-Z0-9]){collapsed}([^A-Z0-9]|$)"))
    return patterns


def _contains_other_ticker(text: str, ticker: str) -> bool:
    target = ticker.upper().strip().replace(".", "").replace("-", "")
    for candidate in re.findall(r"\$([A-Z]{1,5})\b", text):
        clean = candidate.replace(".", "").replace("-", "")
        if clean and clean != target:
            return True
    return False


def _market_text(item: Dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key, "") or "")
        for key in ("title", "question", "subtitle", "description", "ticker", "slug")
    ).upper()


def _build_polymarket_link(item: Dict[str, Any]) -> str:
    """Build a stable, user-facing Polymarket URL with safe fallbacks."""
    def _normalize_slug(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        slug = value.strip().strip("/")
        if not slug:
            return None
        # Drop accidental path prefixes from API payload fields.
        slug = re.sub(r"^(event|market)/", "", slug, flags=re.IGNORECASE)
        # Remove query/hash fragments.
        slug = slug.split("?", 1)[0].split("#", 1)[0].strip("/")
        if not slug:
            return None
        # Keep only slug-like values; reject free-form text.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{1,200}", slug):
            return None
        return slug.lower()

    def _build_search_fallback() -> str:
        query = str(item.get("question") or item.get("title") or "").strip()
        if query:
            return f"https://polymarket.com/search?q={quote(query)}"
        return "https://polymarket.com/markets"

    for key in ("url", "link", "market_url", "marketUrl"):
        value = item.get(key)
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.startswith("http://") or candidate.startswith("https://"):
                parsed = urlparse(candidate)
                host = (parsed.hostname or "").lower()
                if host in {"polymarket.com", "www.polymarket.com"}:
                    return candidate

    market_slug = None
    for key in ("market_slug", "marketSlug", "slug"):
        market_slug = _normalize_slug(item.get(key))
        if market_slug:
            break
    if market_slug:
        return f"https://polymarket.com/market/{quote(market_slug, safe='-')}"

    event_slug = None
    for key in ("event_slug", "eventSlug"):
        event_slug = _normalize_slug(item.get(key))
        if event_slug:
            break
    if event_slug:
        return f"https://polymarket.com/event/{quote(event_slug, safe='-')}"

    return _build_search_fallback()


def _semantic_relevance(item: Dict[str, Any], ticker: str, company_name: str | None = None) -> float:
    """Calculate how relevant a prediction market is to a given stock ticker.

    Returns a score > 0 if relevant, 0 if not relevant.
    Requires either:
    - Direct ticker match (e.g., "TSLA" in text), OR
    - Multiple company name tokens AND financial context terms
    """
    text = _market_text(item)
    if not text:
        return 0.0

    terms = _semantic_terms(ticker, company_name)
    ticker_token = ticker.upper().strip()
    tokens = set(re.findall(r"[A-Z0-9]+", text))
    text_lower = text.lower()
    score = 0.0
    matched = 0
    direct_ticker_hit = False

    ticker_regexes = _ticker_regexes(ticker_token)
    direct_ticker_hit = any(pattern.search(text) for pattern in ticker_regexes)
    if direct_ticker_hit:
        score += 4.5

    phrase_hits = 0
    for phrase in _company_phrases(company_name):
        if phrase in text_lower:
            phrase_hits += 1
    if phrase_hits > 0:
        score += min(3.0, 1.5 * phrase_hits)

    for term in terms:
        if term in tokens:
            matched += 1
            if term == ticker_token or term == ticker_token.replace("-", ""):
                score += 2.0
            else:
                # Company name match - lower score, requires additional context
                score += 0.8

    company_tokens = _company_terms(company_name)
    overlap = sum(1 for token in company_tokens if token in tokens)
    if overlap >= 2:
        score += 1.2

    if "/" in text and ticker_token in tokens:
        score += 0.6
    if matched >= 2:
        score += 0.8

    if matched == 0 and not direct_ticker_hit and phrase_hits == 0:
        return 0.0

    # Require an explicit stock identity signal to avoid repeating generic macro contracts.
    if not direct_ticker_hit and phrase_hits == 0:
        return 0.0

    context_hits = sum(1 for term in _FINANCE_CONTEXT_TERMS if term in tokens)
    if context_hits > 0:
        score += min(1.2, 0.3 * context_hits)
    elif not direct_ticker_hit and overlap < 2:
        # Require direct ticker hit OR multiple company name matches for relevance
        # This avoids false positives from generic company name words (e.g., "Apple" in "Apple Music")
        return 0.0

    if _contains_other_ticker(text, ticker):
        score -= 2.0

    if score <= 0:
        return 0.0
    return score


async def fetch_kalshi_markets(ticker: str, settings: Settings, company_name: str | None = None) -> List[Dict[str, Any]]:
    """Fetch prediction markets from Kalshi that may be relevant to the given stock ticker.

    Note: Kalshi primarily offers political and sports markets. Stock-specific markets are rare.
    """
    headers = {"Content-Type": "application/json"}

    # Kalshi elections API is the current primary endpoint
    kalshi_url = "https://api.elections.kalshi.com/trade-api/v2/markets"

    queries: list[str] = [ticker.upper()]
    if company_name and company_name.strip():
        queries.extend(_company_phrases(company_name))
    queries.extend(_stock_theme_queries(ticker, company_name))
    seen_queries: set[str] = set()
    queries = [q for q in queries if q and not (q.lower() in seen_queries or seen_queries.add(q.lower()))]
    all_items: List[Dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for query in queries:
                params = {"limit": _KALSHI_QUERY_LIMIT, "status": "open", "search": query}
                try:
                    resp = await client.get(kalshi_url, headers=headers, params=params)
                    resp.raise_for_status()
                    payload = resp.json()
                    items = payload.get("markets", [])
                    logger.debug(f"Kalshi query '{query}' returned {len(items)} markets")
                    all_items.extend(item for item in items if isinstance(item, dict))
                except httpx.HTTPStatusError as e:
                    logger.warning(f"Kalshi API error for query '{query}': {e.response.status_code}")
                except Exception as e:
                    logger.warning(f"Kalshi API exception for query '{query}': {type(e).__name__}: {e}")

            if not all_items:
                logger.info(f"No Kalshi markets found for {ticker}")
                return []

            dedup: dict[str, Dict[str, Any]] = {}
            for item in all_items:
                key = str(item.get("ticker") or item.get("id") or item.get("title") or "")
                if key:
                    dedup[key] = item
            all_items = list(dedup.values())

            scored: list[tuple[float, Dict[str, Any]]] = []
            for item in all_items:
                relevance = _semantic_relevance(item, ticker, company_name)
                if relevance <= 0:
                    continue
                scored.append((relevance, item))

            if scored:
                logger.debug(f"Kalshi: {len(scored)} markets matched relevance filter for {ticker} (from {len(all_items)} total)")
            else:
                themes = _stock_theme_queries(ticker, company_name)
                themed: list[tuple[float, Dict[str, Any]]] = []
                for item in all_items:
                    relevance = _thematic_relevance(item, themes)
                    if relevance > 0:
                        themed.append((relevance, item))
                themed.sort(key=lambda pair: pair[0], reverse=True)
                scored = themed[:5]
                if scored:
                    logger.info(f"Kalshi: using {len(scored)} sector-proxy markets for {ticker}")
                else:
                    logger.info(f"Kalshi: No stock-related markets found for {ticker}. Available markets are primarily political/sports events.")

            scored.sort(key=lambda pair: pair[0], reverse=True)
            if not scored:
                return []

            out: list[Dict[str, Any]] = []
            for score, item in scored[:5]:
                yes_price, no_price = _extract_kalshi_prices(item)
                if yes_price is None or no_price is None:
                    continue
                out.append(
                    {
                        "source": "Kalshi",
                        "market": item.get("title", item.get("ticker", "Unknown")),
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "link": f"https://kalshi.com/markets/{item.get('ticker', '')}",
                        "relevance_score": round(score, 3),
                        "context": "stock-specific" if _semantic_relevance(item, ticker, company_name) > 0 else "sector-proxy",
                    }
                )
            logger.info(f"Kalshi: returning {len(out)} markets for {ticker}")
            return out
    except Exception as e:
        logger.error(f"Kalshi fetch failed for {ticker}: {type(e).__name__}: {e}")
        return []


async def fetch_polymarket_markets(ticker: str, settings: Settings, company_name: str | None = None) -> List[Dict[str, Any]]:
    headers = {}
    queries: list[str] = [ticker.upper()]
    if company_name and company_name.strip():
        company = company_name.strip()
        queries.append(company)
        company_terms = [term.title() for term in sorted(_company_terms(company)) if term.upper() != ticker.upper()]
        queries.extend(company_terms[:3])
        queries.extend(_company_phrases(company))
        # Expand stock-specific searches to capture contracts phrased in natural language.
        queries.append(f"{company} earnings")
        queries.append(f"{company} revenue")
        queries.append(f"{company} stock")
        queries.append(f"{company} price")
        queries.append(f"{ticker} stock price")
        if any(token in company.lower() for token in ["tesla", "ford", "gm", "toyota", "rivian", "lucid"]):
            queries.append(f"{company} deliveries")
            queries.append(f"{company} cars sold")
            queries.append(f"{company} vehicle sales")
        if any(token in company.lower() for token in ["nvidia", "amd", "intel", "qualcomm", "broadcom"]):
            queries.append(f"{company} data center revenue")
            queries.append(f"{company} ai chip sales")
    queries.extend(_stock_theme_queries(ticker, company_name))
    # Deduplicate while preserving order
    seen_queries: set[str] = set()
    queries = [q for q in queries if q and not (q.lower() in seen_queries or seen_queries.add(q.lower()))]

    logger.debug(f"Polymarket: searching with queries {queries[:5]}... for {ticker}")

    try:
        merged: dict[str, Dict[str, Any]] = {}
        async with httpx.AsyncClient(timeout=15.0) as client:
            for query in queries:
                try:
                    params = {"limit": _POLYMARKET_QUERY_LIMIT, "closed": "false", "search": query}
                    url = f"{settings.polymarket_gamma_url.rstrip('/')}/markets"
                    resp = await client.get(url, headers=headers, params=params)
                    resp.raise_for_status()
                    payload = resp.json()
                    results = payload if isinstance(payload, list) else payload.get("data", [])
                    logger.debug(f"Polymarket query '{query}' returned {len(results) if results else 0} results")
                    for item in results or []:
                        key = str(item.get("id") or item.get("slug") or item.get("question") or item.get("title") or "")
                        if not key:
                            continue
                        merged[key] = item
                except httpx.HTTPStatusError as e:
                    logger.warning(f"Polymarket API error for query '{query}': {e.response.status_code}")
                except Exception as e:
                    logger.warning(f"Polymarket API exception for query '{query}': {type(e).__name__}: {e}")

            logger.debug(f"Polymarket: {len(merged)} unique markets found for {ticker}")

            scored: list[tuple[float, Dict[str, Any]]] = []
            for item in merged.values():
                relevance = _semantic_relevance(item, ticker, company_name)
                if relevance <= 0:
                    continue
                scored.append((relevance, item))

            if scored:
                logger.debug(f"Polymarket: {len(scored)} markets passed relevance filter for {ticker}")
            else:
                logger.info(f"Polymarket: No stock-related markets found for {ticker}. Current markets are primarily political events.")

            scored.sort(key=lambda pair: pair[0], reverse=True)
            if not scored:
                themes = _stock_theme_queries(ticker, company_name)
                themed: list[tuple[float, Dict[str, Any]]] = []
                broad_items: list[Dict[str, Any]] = []
                for item in merged.values():
                    relevance = _thematic_relevance(item, themes)
                    if relevance > 0:
                        themed.append((relevance, item))
                themed.sort(key=lambda pair: pair[0], reverse=True)
                scored = themed[:5]
                if not scored:
                    # Last fallback: rank from the broader open market set by stock theme.
                    try:
                        resp = await client.get(
                            f"{settings.polymarket_gamma_url.rstrip('/')}/markets",
                            headers=headers,
                            params={"limit": 200, "closed": "false"},
                        )
                        resp.raise_for_status()
                        payload = resp.json()
                        broad = payload if isinstance(payload, list) else payload.get("data", [])
                        broad_items = [item for item in (broad or []) if isinstance(item, dict)]
                        for item in broad or []:
                            if not isinstance(item, dict):
                                continue
                            relevance = _thematic_relevance(item, themes)
                            if relevance > 0:
                                themed.append((relevance, item))
                        themed.sort(key=lambda pair: pair[0], reverse=True)
                        scored = themed[:5]
                    except Exception:
                        scored = []
                if not scored and broad_items:
                    macro_candidates: list[Dict[str, Any]] = []
                    for item in broad_items:
                        yes_price, no_price = _extract_polymarket_prices(item)
                        if yes_price is None or no_price is None:
                            continue
                        if not _is_macro_relevant(item):
                            continue
                        macro_candidates.append(item)
                    if macro_candidates:
                        macro_candidates.sort(
                            key=lambda row: _safe_float(row.get("volumeNum") or row.get("volume")) or 0.0,
                            reverse=True,
                        )
                        seed = sum(ord(ch) for ch in ticker.upper())
                        window = min(5, len(macro_candidates))
                        offset = seed % max(1, len(macro_candidates))
                        picked: list[Dict[str, Any]] = []
                        for idx in range(window):
                            picked.append(macro_candidates[(offset + idx) % len(macro_candidates)])
                        scored = [(0.2, item) for item in picked]
                if not scored:
                    return []

            out: list[Dict[str, Any]] = []
            for score, item in scored[:5]:
                yes_price, no_price = _extract_polymarket_prices(item)
                if yes_price is None or no_price is None:
                    logger.debug(f"Polymarket: skipping market '{item.get('question', '')[:50]}' - no valid prices")
                    continue
                out.append(
                    {
                        "source": "Polymarket",
                        "market": item.get("question") or item.get("title", "Unknown market"),
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "probability": yes_price,
                        "volume": item.get("volumeNum") or item.get("volume"),
                        "link": _build_polymarket_link(item),
                        "relevance_score": round(score, 3),
                        "context": "stock-specific" if _semantic_relevance(item, ticker, company_name) > 0 else "sector-proxy",
                    }
                )
            logger.info(f"Polymarket: returning {len(out)} markets for {ticker}")
            return out
    except Exception as e:
        logger.error(f"Polymarket fetch failed for {ticker}: {type(e).__name__}: {e}")
        return []


# Keywords that indicate macro/economic relevance for fallback markets
_MACRO_POSITIVE_KEYWORDS = {
    "fed", "federal reserve", "rate cut", "rate hike", "inflation", "cpi",
    "gdp", "recession", "unemployment", "jobs report", "treasury", "fomc",
    "tariff", "trade war", "deficit", "debt ceiling",
    "fiscal", "bitcoin", "btc", "crypto",
    "s&p", "nasdaq", "dow", "stock market", "oil", "wti", "brent",
}

# Keywords that indicate the market is NOT relevant (filter out)
_MACRO_NEGATIVE_KEYWORDS = {
    "prime minister", "parliament", "election", "president", "governor",
    "album", "gta", "game", "movie", "song", "artist", "sports", "nfl", "nba",
    "netherlands", "germany", "france", "uk ", "britain", "canada",
    "deport", "immigration", "border", "cabinet", "senate", "house race",
    "celebrity", "oscar", "grammy", "world cup",
}


def _is_macro_relevant(item: Dict[str, Any]) -> bool:
    """Check if a market is relevant to US macro/economic topics."""
    text = _market_text(item).lower()

    def contains_keyword(keyword: str) -> bool:
        key = keyword.lower().strip()
        if not key:
            return False
        if " " in key:
            return key in text
        return re.search(rf"\b{re.escape(key)}\b", text) is not None

    # Exclude if it contains negative keywords
    if any(contains_keyword(kw) for kw in _MACRO_NEGATIVE_KEYWORDS):
        return False

    # Include if it contains positive keywords
    return any(contains_keyword(kw) for kw in _MACRO_POSITIVE_KEYWORDS)


def _stock_theme_queries(ticker: str, company_name: str | None) -> list[str]:
    text = f"{ticker} {company_name or ''}".lower()
    themes: list[str] = []
    if any(token in text for token in ("nvidia", "amd", "intel", "qualcomm", "broadcom", "tsm")):
        themes.extend(["ai chips", "semiconductors", "data center demand"])
    if any(token in text for token in ("tesla", "rivian", "lucid", "nio", "ford", "gm", "toyota")):
        themes.extend(["electric vehicles", "ev sales", "auto deliveries", "battery supply", "oil prices", "lithium"])
    if any(token in text for token in ("apple", "microsoft", "amazon", "google", "alphabet", "meta")):
        themes.extend(["big tech earnings", "cloud revenue", "ai demand", "consumer electronics", "nasdaq"])
    if any(token in text for token in ("coinbase", "microstrategy", "riot", "marathon", "ibit", "bitcoin")):
        themes.extend(["bitcoin price", "crypto market", "btc adoption"])
    if any(token in text for token in ("jpmorgan", "bank", "goldman", "wells fargo", "citigroup")):
        themes.extend(["interest rates", "fed policy", "bank earnings"])

    if not themes:
        themes.extend(["stock market", "earnings", "interest rates"])

    seen: set[str] = set()
    out: list[str] = []
    for term in themes:
        clean = term.strip().lower()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(term)
    return out[:4]


def _thematic_relevance(item: Dict[str, Any], themes: list[str]) -> float:
    text = _market_text(item)
    if not text:
        return 0.0
    text_lower = text.lower()
    hits = 0
    for theme in themes:
        theme_lower = theme.lower()
        if theme_lower in text_lower:
            hits += 1
            continue
        parts = [
            p
            for p in re.findall(r"[a-z0-9]+", theme_lower)
            if len(p) >= 4 and p not in _THEME_TOKEN_STOPWORDS
        ]
        matched_parts = sum(1 for part in parts if part in text_lower)
        if matched_parts >= 2 or (len(parts) == 1 and matched_parts == 1):
            hits += 1
    if hits == 0:
        return 0.0

    score = 0.7 * hits
    if _is_macro_relevant(item):
        score += 0.4
    volume = _safe_float(item.get("volumeNum") or item.get("volume")) or 0.0
    if volume > 0:
        score += min(0.8, volume / 1_000_000.0)
    return score


async def fetch_macro_fallback_markets(settings: Settings, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch general macro/economic prediction markets as fallback when no stock-specific markets exist.

    These markets cover topics like government spending, fiscal policy, and economic indicators
    that can broadly impact stock markets.
    """
    headers = {}
    out: List[Dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Fetch recent active markets from Polymarket
            resp = await client.get(
                f"{settings.polymarket_gamma_url.rstrip('/')}/markets",
                headers=headers,
                params={"limit": 100, "closed": "false"}
            )
            resp.raise_for_status()
            markets = resp.json()

            # Filter for macro-relevant markets
            macro_markets: List[tuple[float, Dict[str, Any]]] = []
            for item in markets:
                if not _is_macro_relevant(item):
                    continue
                # Score by volume (popularity) for ranking
                volume = float(item.get("volumeNum") or item.get("volume") or 0)
                macro_markets.append((volume, item))

            # Sort by volume (most popular first)
            macro_markets.sort(key=lambda x: x[0], reverse=True)

            for _, item in macro_markets[:limit]:
                yes_price, no_price = _extract_polymarket_prices(item)
                if yes_price is None or no_price is None:
                    continue
                out.append({
                    "source": "Polymarket",
                    "market": item.get("question") or item.get("title", "Unknown market"),
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "probability": yes_price,
                    "volume": item.get("volumeNum") or item.get("volume"),
                    "link": _build_polymarket_link(item),
                    "relevance_score": 0.5,  # Lower score to indicate macro-adjacent
                    "context": "macro-adjacent",
                })

            logger.info(f"Macro fallback: returning {len(out)} economic/policy markets")
            return out

    except Exception as e:
        logger.warning(f"Macro fallback fetch failed: {type(e).__name__}: {e}")
        return []
