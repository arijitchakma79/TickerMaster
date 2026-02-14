from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx

from app.config import Settings
from app.schemas import ResearchRequest, ResearchResponse, SentimentBreakdown, SourceLink
from app.services.prediction_markets import fetch_kalshi_markets, fetch_polymarket_markets


POSITIVE_WORDS = {
    "beat",
    "upside",
    "growth",
    "bullish",
    "surge",
    "breakout",
    "strong",
    "improve",
    "momentum",
    "upgrade",
}
NEGATIVE_WORDS = {
    "miss",
    "downgrade",
    "bearish",
    "drop",
    "selloff",
    "weak",
    "risk",
    "lawsuit",
    "decline",
    "recession",
}


def _label(score: float) -> str:
    if score >= 0.2:
        return "bullish"
    if score <= -0.2:
        return "bearish"
    return "neutral"


def _recommendation(score: float) -> str:
    if score > 0.65:
        return "strong_buy"
    if score > 0.25:
        return "buy"
    if score < -0.65:
        return "strong_sell"
    if score < -0.25:
        return "sell"
    return "hold"


def _sentiment_score_from_text(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return 0.0
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    score = (pos - neg) / max(1, int(math.sqrt(len(tokens))))
    return max(-1.0, min(1.0, score))


async def _perplexity_summary(ticker: str, settings: Settings) -> Dict[str, Any]:
    if not settings.perplexity_api_key:
        summary = (
            f"Demo summary for {ticker}: earnings sentiment is mixed, options flow leans slightly bullish, "
            "and macro rate path remains the key risk variable."
        )
        return {
            "summary": summary,
            "links": [
                SourceLink(source="Perplexity", title=f"{ticker} live context", url="https://www.perplexity.ai"),
            ],
            "score": _sentiment_score_from_text(summary),
        }

    headers = {
        "Authorization": f"Bearer {settings.perplexity_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings.perplexity_model,
        "messages": [
            {
                "role": "system",
                "content": "You summarize market catalysts with concise bullets and include source URLs.",
            },
            {
                "role": "user",
                "content": (
                    f"Explain the most important bullish and bearish catalysts for {ticker} over the last 7 days. "
                    "Return short text only."
                ),
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://api.perplexity.ai/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {
                "summary": content,
                "links": [SourceLink(source="Perplexity", title=f"{ticker} catalysts", url="https://www.perplexity.ai")],
                "score": _sentiment_score_from_text(content),
            }
    except Exception as exc:
        fallback = f"Perplexity request failed: {exc}. Using baseline narrative for {ticker}."
        return {
            "summary": fallback,
            "links": [SourceLink(source="Perplexity", title="Perplexity", url="https://www.perplexity.ai")],
            "score": 0.0,
        }


async def _x_summary(ticker: str, settings: Settings) -> Dict[str, Any]:
    if not settings.x_api_bearer_token:
        text = f"Demo X flow for {ticker}: traders are debating valuation vs. AI-driven revenue acceleration."
        return {
            "summary": text,
            "links": [SourceLink(source="X", title=f"${ticker} on X", url=f"https://x.com/search?q=%24{ticker}")],
            "score": _sentiment_score_from_text(text),
        }

    headers = {"Authorization": f"Bearer {settings.x_api_bearer_token}"}
    params = {
        "query": f"${ticker} lang:en -is:retweet",
        "max_results": 25,
        "tweet.fields": "created_at,text,public_metrics",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://api.twitter.com/2/tweets/search/recent", headers=headers, params=params)
            resp.raise_for_status()
            tweets = resp.json().get("data", [])
    except Exception as exc:
        tweets = []
        error_msg = f"X fetch failed: {exc}"
    else:
        error_msg = ""

    if not tweets:
        summary = error_msg or f"No high-signal X posts found for {ticker}."
        score = 0.0
    else:
        joined = " ".join(tweet.get("text", "") for tweet in tweets)
        score = _sentiment_score_from_text(joined)
        summary = f"Parsed {len(tweets)} recent X posts for {ticker}. Crowd tone is {_label(score)}."

    return {
        "summary": summary,
        "links": [SourceLink(source="X", title=f"{ticker} search", url=f"https://x.com/search?q=%24{ticker}")],
        "score": score,
    }


async def _reddit_summary(ticker: str, settings: Settings) -> Dict[str, Any]:
    headers = {"User-Agent": settings.reddit_user_agent or "TickerMaster/1.0"}
    params = {
        "q": ticker,
        "restrict_sr": "false",
        "sort": "new",
        "t": "week",
        "limit": 20,
    }

    # Public JSON search works for lightweight use and avoids auth complexity for MVP.
    url = "https://www.reddit.com/search.json"
    posts: List[Dict[str, Any]] = []
    error_msg = ""

    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
            posts = [child.get("data", {}) for child in children]
    except Exception as exc:
        error_msg = f"Reddit fetch failed: {exc}"

    if not posts:
        summary = error_msg or f"No notable Reddit discussions captured for {ticker}."
        score = 0.0
    else:
        joined = " ".join(post.get("title", "") + " " + post.get("selftext", "") for post in posts)
        score = _sentiment_score_from_text(joined)
        summary = f"Reddit chatter sampled from {len(posts)} posts. Sentiment skew is {_label(score)}."

    return {
        "summary": summary,
        "links": [
            SourceLink(source="Reddit", title=f"{ticker} search", url=f"https://www.reddit.com/search/?q={ticker}")
        ],
        "score": score,
    }


def _build_narratives(items: List[SentimentBreakdown]) -> List[str]:
    bullish = [i for i in items if i.sentiment == "bullish"]
    bearish = [i for i in items if i.sentiment == "bearish"]

    lines = []
    if bullish:
        lines.append("Upside case: momentum and positioning suggest upside continuation if macro remains stable.")
    if bearish:
        lines.append("Downside case: valuation and macro uncertainty can trigger drawdowns on negative surprises.")
    lines.append("Execution note: use smaller sizes around event windows due to elevated slippage risk.")
    return lines


async def run_research(request: ResearchRequest, settings: Settings) -> ResearchResponse:
    ticker = request.ticker.upper().strip()

    perplexity_task = _perplexity_summary(ticker, settings)
    x_task = _x_summary(ticker, settings)
    reddit_task = _reddit_summary(ticker, settings)

    perplexity, x_summary, reddit = await asyncio.gather(perplexity_task, x_task, reddit_task)

    breakdown = [
        SentimentBreakdown(
            source="Perplexity Sonar",
            sentiment=_label(perplexity["score"]),
            score=round(float(perplexity["score"]), 3),
            summary=perplexity["summary"],
            links=perplexity["links"],
        ),
        SentimentBreakdown(
            source="X API",
            sentiment=_label(x_summary["score"]),
            score=round(float(x_summary["score"]), 3),
            summary=x_summary["summary"],
            links=x_summary["links"],
        ),
        SentimentBreakdown(
            source="Reddit API",
            sentiment=_label(reddit["score"]),
            score=round(float(reddit["score"]), 3),
            summary=reddit["summary"],
            links=reddit["links"],
        ),
    ]

    prediction_markets = []
    if request.include_prediction_markets:
        kalshi_markets, polymarket_markets = await asyncio.gather(
            fetch_kalshi_markets(ticker, settings),
            fetch_polymarket_markets(ticker, settings),
        )
        prediction_markets = kalshi_markets + polymarket_markets

    aggregate = float(sum(item.score for item in breakdown) / max(1, len(breakdown)))

    links = [
        SourceLink(source="Perplexity Sonar", title="Perplexity", url="https://www.perplexity.ai"),
        SourceLink(source="X API", title="X Developer", url="https://developer.x.com/en/docs"),
        SourceLink(source="Reddit API", title="Reddit Dev", url="https://www.reddit.com/dev/api/"),
        SourceLink(source="Kalshi API", title="Kalshi Docs", url="https://docs.kalshi.com/"),
        SourceLink(source="Polymarket", title="Polymarket", url="https://docs.polymarket.com/"),
        SourceLink(source="Morningstar", title="Morningstar", url="https://www.morningstar.com/"),
        SourceLink(source="Reuters", title="Reuters Markets", url="https://www.reuters.com/markets/"),
        SourceLink(source="J.P. Morgan", title="J.P. Morgan Insights", url="https://www.jpmorgan.com/insights"),
        SourceLink(source="Yahoo Finance", title="Yahoo Finance", url="https://finance.yahoo.com/"),
    ]

    return ResearchResponse(
        ticker=ticker,
        generated_at=datetime.now(timezone.utc).isoformat(),
        aggregate_sentiment=round(aggregate, 3),
        recommendation=_recommendation(aggregate),
        narratives=_build_narratives(breakdown),
        source_breakdown=breakdown,
        prediction_markets=prediction_markets,
        tool_links=links,
    )
