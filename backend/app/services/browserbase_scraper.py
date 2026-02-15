from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

import httpx

from app.config import Settings
from app.schemas import ResearchRequest
from app.services.agent_logger import log_agent_activity
from app.services.market_data import fetch_advanced_stock_data
from app.services.sentiment import run_research


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _finnhub_get(settings: Settings, path: str, params: dict[str, Any] | None = None) -> Any:
    if not settings.finnhub_api_key:
        return None
    query = dict(params or {})
    query["token"] = settings.finnhub_api_key
    url = f"{settings.finnhub_api_url.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=12.0) as client:
            resp = client.get(url, params=query)
            if resp.status_code in {401, 403, 404, 429}:
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


def _recommendation_timeline(settings: Settings, ticker: str) -> list[dict[str, Any]]:
    payload = _finnhub_get(settings, "/stock/recommendation", {"symbol": ticker})
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for row in payload[:8]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "period": str(row.get("period") or ""),
                "strong_buy": _safe_int(row.get("strongBuy")),
                "buy": _safe_int(row.get("buy")),
                "hold": _safe_int(row.get("hold")),
                "sell": _safe_int(row.get("sell")),
                "strong_sell": _safe_int(row.get("strongSell")),
            }
        )
    return out


def _price_target(settings: Settings, ticker: str) -> dict[str, Any]:
    payload = _finnhub_get(settings, "/stock/price-target", {"symbol": ticker})
    if not isinstance(payload, dict):
        return {}
    return {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "target_high": _safe_float(payload.get("targetHigh")),
        "target_low": _safe_float(payload.get("targetLow")),
        "target_mean": _safe_float(payload.get("targetMean")),
        "target_median": _safe_float(payload.get("targetMedian")),
    }


def _company_news(settings: Settings, ticker: str) -> list[dict[str, Any]]:
    if not settings.finnhub_api_key:
        return []
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=21)
    payload = _finnhub_get(
        settings,
        "/company-news",
        {"symbol": ticker, "from": start.isoformat(), "to": today.isoformat()},
    )
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for row in payload[:10]:
        if not isinstance(row, dict):
            continue
        summary = str(row.get("summary") or "").strip()
        out.append(
            {
                "headline": str(row.get("headline") or "").strip(),
                "source": str(row.get("source") or "news").strip(),
                "url": str(row.get("url") or "").strip(),
                "datetime": row.get("datetime"),
                "summary": summary[:500],
            }
        )
    return [item for item in out if item.get("headline")]


async def _reddit_highlights(ticker: str, user_agent: str) -> Tuple[list[dict[str, Any]], str]:
    headers = {"User-Agent": user_agent or "TickerMaster/1.0"}
    params = {"q": ticker, "restrict_sr": "false", "sort": "new", "t": "month", "limit": 12}
    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            resp = await client.get("https://www.reddit.com/search.json", params=params)
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
    except Exception:
        return [], f"Reddit discussions for {ticker} unavailable from public endpoint."

    highlights: list[dict[str, Any]] = []
    for child in children:
        row = child.get("data", {}) if isinstance(child, dict) else {}
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        permalink = str(row.get("permalink") or "").strip()
        url = f"https://www.reddit.com{permalink}" if permalink else str(row.get("url") or "").strip()
        highlights.append(
            {
                "subreddit": str(row.get("subreddit") or "").strip(),
                "title": title,
                "url": url,
                "score": _safe_int(row.get("score")),
                "comments": _safe_int(row.get("num_comments")),
            }
        )
        if len(highlights) >= 8:
            break

    if not highlights:
        return [], f"No recent high-signal Reddit threads found for {ticker}."
    top = highlights[0]
    subreddit = f"r/{top['subreddit']}" if top.get("subreddit") else "Reddit"
    return highlights, f"Sampled {len(highlights)} recent threads; top momentum from {subreddit}."


async def _create_browserbase_session(settings: Settings) -> tuple[dict[str, Any] | None, str | None, str | None, str | None]:
    if not settings.browserbase_api_key or not settings.browserbase_project_id:
        return None, None, None, "BROWSERBASE_API_KEY or BROWSERBASE_PROJECT_ID is missing."
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.browserbase.com/v1/sessions",
                headers={
                    "x-bb-api-key": settings.browserbase_api_key,
                    "Content-Type": "application/json",
                },
                json={"projectId": settings.browserbase_project_id, "keepAlive": False},
            )
            resp.raise_for_status()
            payload = resp.json()
            session_id = str(payload.get("id") or payload.get("sessionId") or "")
            connect_url = str(payload.get("connectUrl") or payload.get("wsEndpoint") or "")
            return payload, (session_id or None), (connect_url or None), None
    except Exception as exc:
        return None, None, None, str(exc)


def _insider_highlights(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for row in rows[:20]:
        date = row.get("start_date")
        name = row.get("filer_name")
        code = row.get("money_text")
        shares = _safe_float(row.get("shares"))
        value = _safe_float(row.get("value"))
        if not (date or name or code):
            continue
        highlights.append(
            {
                "date": date,
                "name": name,
                "code": code,
                "shares": shares,
                "price": None,
                "value_estimate": value,
            }
        )
        if len(highlights) >= 10:
            break
    return highlights


def _build_analyst_summary(ticker: str, timeline: list[dict[str, Any]], price_target: dict[str, Any]) -> str:
    if not timeline and not price_target:
        return f"Analyst recommendation detail for {ticker} not available from current providers."
    latest = timeline[0] if timeline else {}
    strong_buy = _safe_int(latest.get("strong_buy"))
    buy = _safe_int(latest.get("buy"))
    hold = _safe_int(latest.get("hold"))
    sell = _safe_int(latest.get("sell"))
    strong_sell = _safe_int(latest.get("strong_sell"))
    period = str(latest.get("period") or "").strip() or "latest period"
    pieces: list[str] = []
    if timeline:
        pieces.append(
            f"{period} analyst mix: Strong Buy {strong_buy}, Buy {buy}, Hold {hold}, Sell {sell}, Strong Sell {strong_sell}."
        )
    target_mean = _safe_float(price_target.get("target_mean"))
    target_low = _safe_float(price_target.get("target_low"))
    target_high = _safe_float(price_target.get("target_high"))
    if target_mean is not None:
        pieces.append(
            f"Street target mean {target_mean:.2f}"
            + (f" (range {target_low:.2f} - {target_high:.2f})." if target_low is not None and target_high is not None else ".")
        )
    return " ".join(pieces).strip()


def _build_insider_summary(ticker: str, highlights: list[dict[str, Any]]) -> str:
    if not highlights:
        return f"No recent insider filing highlights were returned for {ticker}."
    trades = 0
    total_value = 0.0
    coded_actions: dict[str, int] = {}
    for item in highlights:
        trades += 1
        code = str(item.get("code") or "").upper().strip()
        if code:
            coded_actions[code] = coded_actions.get(code, 0) + 1
        value = _safe_float(item.get("value_estimate"))
        if value is not None:
            total_value += value
    top_actions = ", ".join(f"{k} ({v})" for k, v in sorted(coded_actions.items(), key=lambda pair: pair[1], reverse=True)[:3])
    value_clause = f" disclosed value ~${total_value:,.0f}" if total_value > 0 else ""
    action_clause = f" Key actions: {top_actions}." if top_actions else ""
    return f"Captured {trades} recent insider filings for {ticker}{value_clause}.{action_clause}".strip()


def _build_deep_bullets(
    ticker: str,
    analyst_text: str,
    insider_text: str,
    reddit_summary: str,
    news_items: list[dict[str, Any]],
) -> list[str]:
    bullets = [analyst_text, insider_text, reddit_summary]
    if news_items:
        top_headline = str(news_items[0].get("headline") or "").strip()
        if top_headline:
            bullets.append(f"Latest company-news headline: {top_headline}")
    cleaned = [item.strip() for item in bullets if item and item.strip()]
    return cleaned[:6]


async def run_deep_research(symbol: str, settings: Settings) -> Dict[str, Any]:
    ticker = symbol.upper().strip()
    await log_agent_activity(
        module="research",
        agent_name="Browserbase Agent",
        action=f"Starting deep research scrape for {ticker}",
        status="running",
    )

    session_payload, session_id, connect_url, browserbase_error = await _create_browserbase_session(settings)

    advanced_task = asyncio.to_thread(fetch_advanced_stock_data, ticker)
    timeline_task = asyncio.to_thread(_recommendation_timeline, settings, ticker)
    target_task = asyncio.to_thread(_price_target, settings, ticker)
    news_task = asyncio.to_thread(_company_news, settings, ticker)
    reddit_task = _reddit_highlights(ticker, settings.reddit_user_agent)
    research_task = run_research(
        ResearchRequest(ticker=ticker, timeframe="30d", include_prediction_markets=True),
        settings,
    )

    advanced_res, timeline_res, target_res, news_res, reddit_res, research_res = await asyncio.gather(
        advanced_task,
        timeline_task,
        target_task,
        news_task,
        reddit_task,
        research_task,
        return_exceptions=True,
    )

    advanced = advanced_res if isinstance(advanced_res, dict) else {}
    recommendation_timeline = timeline_res if isinstance(timeline_res, list) else []
    price_target = target_res if isinstance(target_res, dict) else {}
    recent_news = news_res if isinstance(news_res, list) else []
    reddit_highlights, reddit_dd_summary = reddit_res if isinstance(reddit_res, tuple) else ([], f"Reddit data unavailable for {ticker}.")

    if _safe_float(price_target.get("target_mean")) is None:
        advanced_target_mean = _safe_float(advanced.get("target_mean_price"))
        if advanced_target_mean is not None:
            price_target = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "target_high": _safe_float(price_target.get("target_high")),
                "target_low": _safe_float(price_target.get("target_low")),
                "target_mean": advanced_target_mean,
                "target_median": _safe_float(price_target.get("target_median")),
            }

    insider_highlights = _insider_highlights(list(advanced.get("insider_transactions") or []))
    insider_trading = _build_insider_summary(ticker, insider_highlights)
    analyst_ratings = _build_analyst_summary(ticker, recommendation_timeline, price_target)

    source_breakdown = []
    narratives = []
    if not isinstance(research_res, Exception):
        source_breakdown = list(research_res.source_breakdown or [])
        narratives = list(research_res.narratives or [])
        if reddit_dd_summary.startswith("Reddit discussions") or reddit_dd_summary.startswith("No recent"):
            for row in source_breakdown:
                if str(row.source) == "Reddit API" and str(row.summary or "").strip():
                    reddit_dd_summary = str(row.summary)
                    break

    deep_bullets = _build_deep_bullets(ticker, analyst_ratings, insider_trading, reddit_dd_summary, recent_news)
    if narratives:
        deep_bullets.extend([item for item in narratives if item and item.strip()])
        deep_bullets = deep_bullets[:8]

    sources: list[str] = []
    if recommendation_timeline or price_target:
        sources.append("Finnhub")
    if insider_highlights:
        sources.append("Finnhub Insider / SEC Form 4")
    if recent_news:
        sources.append("Finnhub Company News")
    if reddit_highlights:
        sources.append("Reddit")
    for row in source_breakdown:
        label = str(getattr(row, "source", "") or "").strip()
        if label and label not in sources:
            sources.append(label)
    if session_id:
        sources.append("Browserbase Session")

    notes_parts: list[str] = []
    if browserbase_error:
        notes_parts.append(f"Browserbase session unavailable: {browserbase_error}")
    elif session_id:
        notes_parts.append("Browserbase session established for selector-driven extraction.")
    if not settings.finnhub_api_key:
        notes_parts.append("Finnhub key missing; analyst and news sections may be partial.")

    result = {
        "symbol": ticker,
        "source": "browserbase+fallback" if session_id else "fallback",
        "analyst_ratings": analyst_ratings,
        "insider_trading": insider_trading,
        "reddit_dd_summary": reddit_dd_summary,
        "recommendation_timeline": recommendation_timeline,
        "price_target": price_target,
        "insider_highlights": insider_highlights,
        "reddit_highlights": reddit_highlights,
        "recent_news": recent_news,
        "deep_bullets": deep_bullets,
        "sources": sources,
        "notes": " ".join(notes_parts).strip(),
        "session_id": session_id,
        "connect_url": connect_url,
        "raw_session": session_payload,
    }

    await log_agent_activity(
        module="research",
        agent_name="Browserbase Agent",
        action=f"Deep research complete for {ticker}",
        status="success",
        details={
            "session_id": session_id,
            "insider_highlights": len(insider_highlights),
            "reddit_highlights": len(reddit_highlights),
            "news_items": len(recent_news),
        },
    )
    return result
