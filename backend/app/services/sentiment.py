from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx
try:
    from requests_oauthlib import OAuth1Session
except Exception:  # pragma: no cover
    OAuth1Session = None  # type: ignore[assignment]

from app.config import Settings
from app.schemas import ResearchRequest, ResearchResponse, SentimentBreakdown, SourceLink
from app.services.agent_logger import log_agent_activity
from app.services.market_data import search_tickers
from app.services.prediction_markets import fetch_kalshi_markets, fetch_polymarket_markets, fetch_macro_fallback_markets
from app.services.research_cache import get_cached_research, set_cached_research


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
REDDIT_FOCUS_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "stockmarket",
    "securityanalysis",
    "valueinvesting",
    "pennystocks",
]
REDDIT_FOCUS_SUBREDDITS_SET = {name.lower() for name in REDDIT_FOCUS_SUBREDDITS}
_COMPANY_NAME_STOPWORDS = {
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "company",
    "co",
    "co.",
    "holdings",
    "group",
    "plc",
    "ltd",
    "limited",
    "class",
    "common",
    "stock",
    "the",
    "and",
}
_REDDIT_TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "have",
    "about",
    "into",
    "more",
    "than",
    "will",
    "would",
    "could",
    "should",
    "they",
    "them",
    "their",
    "what",
    "when",
    "where",
    "while",
    "been",
    "being",
    "over",
    "under",
    "just",
    "your",
    "yours",
    "also",
    "very",
    "after",
    "before",
    "because",
    "still",
    "only",
    "much",
    "many",
    "some",
    "such",
    "then",
    "than",
    "into",
    "onto",
    "across",
    "through",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "does",
    "did",
    "doing",
    "said",
    "says",
    "theyre",
    "dont",
    "cant",
    "wont",
    "im",
    "ive",
    "youre",
    "hes",
    "shes",
    "its",
    "weve",
    "theyd",
    "him",
    "her",
    "his",
    "hers",
    "our",
    "ours",
    "who",
    "whom",
    "whose",
    "which",
    "why",
    "how",
    "discussion",
    "thread",
    "post",
    "posts",
    "stock",
    "stocks",
    "market",
    "markets",
}

_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_MARKDOWN_CITATION_PATTERN = re.compile(r"\s*\[(?:\d+(?:\s*,\s*\d+)*)\]")


def _site_name_from_url(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return "Source"
    if not host:
        return "Source"
    host = host.removeprefix("www.")
    parts = [part for part in host.split(".") if part]
    if not parts:
        return "Source"
    if len(parts) >= 2:
        root = parts[-2]
        if root in {"co", "com", "org", "net", "gov", "edu"} and len(parts) >= 3:
            root = parts[-3]
    else:
        root = parts[0]
    clean = root.replace("-", " ").strip()
    return " ".join(token.capitalize() for token in clean.split()) or "Source"


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


def _sanitize_perplexity_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    cleaned_lines: List[str] = []
    for line in raw.splitlines():
        clean = line.strip()
        clean = _MARKDOWN_LINK_PATTERN.sub(r"\1", clean)
        clean = _MARKDOWN_CITATION_PATTERN.sub("", clean)
        clean = clean.replace("__", "**").replace("`", "")
        clean = re.sub(r"\s{2,}", " ", clean).strip()
        if not clean:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        heading_match = re.match(r"^\*\*([^*]+)\*\*:?\s*$", clean)
        if heading_match:
            cleaned_lines.append(f"**{heading_match.group(1).strip()}**:")
            continue

        bullet_match = re.match(r"^(?:[-*•]|\d+[.)])\s*(.+)$", clean)
        if bullet_match:
            cleaned_lines.append(f"- {bullet_match.group(1).strip()}")
            continue

        cleaned_lines.append(clean)

    compact = "\n".join(cleaned_lines).strip()
    if not compact:
        return ""

    if "- " not in compact:
        sentences = [segment.strip(" .") for segment in re.split(r"(?<=[.!?])\s+", compact) if segment.strip()]
        bullets = [f"- {sentence}" for sentence in sentences[:6]]
        if bullets:
            return "**Summary**:\n" + "\n".join(bullets)
    return compact


def _mentions_symbol(text: str, ticker: str) -> bool:
    clean = str(text or "")
    if not clean:
        return False
    escaped = re.escape(ticker)
    cashtag_pattern = rf"\${escaped}\b"
    token_pattern = rf"(?<![A-Z0-9-]){escaped}(?![A-Z0-9-])"
    return bool(
        re.search(cashtag_pattern, clean, flags=re.IGNORECASE)
        or re.search(token_pattern, clean, flags=re.IGNORECASE)
    )


def _company_alias_tokens(company_name: str) -> List[str]:
    if not company_name:
        return []
    parts = re.findall(r"[A-Za-z][A-Za-z&.-]{1,}", company_name.lower())
    out: List[str] = []
    for token in parts:
        normalized = token.strip(".").replace("&", "")
        if len(normalized) < 3:
            continue
        if normalized in _COMPANY_NAME_STOPWORDS:
            continue
        out.append(normalized)
    return list(dict.fromkeys(out))


def _macro_queries_for_context(ticker: str, company_name: str | None = None) -> List[str]:
    base = [
        "fed rates",
        "rate cuts",
        "cpi inflation",
        "jobs report unemployment",
        "recession probability",
        "gdp growth",
    ]

    name = (company_name or "").lower()
    if any(token in name for token in ["uber", "lyft", "airline", "travel", "booking", "expedia"]):
        base.extend(["consumer spending", "oil prices", "gas prices"])
    if any(token in name for token in ["tesla", "ford", "gm", "automotive", "ev"]):
        base.extend(["battery metals", "oil prices", "consumer auto demand"])
    if any(token in name for token in ["apple", "microsoft", "nvidia", "meta", "amazon", "alphabet", "software", "semiconductor"]):
        base.extend(["ai spending", "tech earnings", "nasdaq"])
    if any(token in name for token in ["bank", "jpmorgan", "goldman", "wells fargo", "financial"]):
        base.extend(["yield curve", "bank stress", "credit spreads"])
    if any(token in name for token in ["xom", "chevron", "oil", "energy", "exxon"]):
        base.extend(["crude oil", "opec", "energy demand"])

    # Include ticker-level query as a final attempt in case direct market listings are sparse.
    base.append(ticker.upper())
    seen: set[str] = set()
    out: List[str] = []
    for item in base:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


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
                "content": (
                    "You are a sell-side equity research analyst. "
                    "Return concise markdown only with section headers and bullet points. "
                    "Use this exact shape:\n"
                    "**Bullish Catalysts**:\n"
                    "- ...\n"
                    "**Bearish Catalysts**:\n"
                    "- ...\n"
                    "**What To Watch**:\n"
                    "- ..."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Explain the most important bullish and bearish catalysts for {ticker} over the last 7 days. "
                    "Include concrete data points and short actionable bullets."
                ),
            },
        ],
    }

    try:
        await log_agent_activity(
            module="research",
            agent_name="Perplexity Sonar",
            action=f"Fetching AI summary for {ticker}",
            status="running",
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://api.perplexity.ai/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            content = _sanitize_perplexity_text(data["choices"][0]["message"]["content"])
            citations = data.get("citations", [])
            links = [
                SourceLink(source="Perplexity", title=_site_name_from_url(url), url=url)
                for idx, url in enumerate(citations[:6])
                if isinstance(url, str)
            ]
            if not links:
                links = [SourceLink(source="Perplexity", title=f"{ticker} catalysts", url="https://www.perplexity.ai")]
            await log_agent_activity(
                module="research",
                agent_name="Perplexity Sonar",
                action=f"Summary complete for {ticker}",
                status="success",
                details={"citations": len(links)},
            )
            return {
                "summary": content,
                "links": links,
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
    bearer = settings.x_api_bearer_token.strip()
    # Ignore clearly invalid placeholders accidentally copied from other providers.
    if bearer.startswith("sk-or-"):
        bearer = ""

    if not bearer and not (
        settings.x_consumer_key
        and settings.x_consumer_secret
        and settings.x_access_token
        and settings.x_access_token_secret
    ):
        text = f"Demo X flow for {ticker}: traders are debating valuation vs. AI-driven revenue acceleration."
        return {
            "summary": text,
            "links": [SourceLink(source="X", title=f"${ticker} on X", url=f"https://x.com/search?q=%24{ticker}")],
            "score": _sentiment_score_from_text(text),
        }

    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    params = {
        "query": f"${ticker} lang:en -is:retweet",
        "max_results": 25,
        "tweet.fields": "created_at,text,public_metrics",
    }

    tweets: List[Dict[str, Any]] = []
    error_msg = ""

    if headers:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get("https://api.twitter.com/2/tweets/search/recent", headers=headers, params=params)
                resp.raise_for_status()
                tweets = resp.json().get("data", [])
        except Exception as exc:
            error_msg = f"X v2 bearer fetch failed: {exc}"

    # Fallback: OAuth1 v1.1 search for free-tier/user-token scenarios.
    if not tweets and OAuth1Session is not None and settings.x_consumer_key and settings.x_consumer_secret and settings.x_access_token and settings.x_access_token_secret:
        try:
            oauth = OAuth1Session(
                client_key=settings.x_consumer_key,
                client_secret=settings.x_consumer_secret,
                resource_owner_key=settings.x_access_token,
                resource_owner_secret=settings.x_access_token_secret,
            )
            resp = oauth.get(
                "https://api.twitter.com/1.1/search/tweets.json",
                params={"q": f"${ticker} -filter:retweets", "lang": "en", "count": 25, "result_type": "recent"},
                timeout=15,
            )
            resp.raise_for_status()
            statuses = resp.json().get("statuses", [])
            tweets = [{"text": status.get("text", ""), "created_at": status.get("created_at")} for status in statuses]
            error_msg = ""
        except Exception as exc:
            if error_msg:
                error_msg = f"{error_msg}; OAuth1 fallback failed: {exc}"
            else:
                error_msg = f"X OAuth1 fallback failed: {exc}"

    if not tweets:
        summary = error_msg or f"No high-signal X posts found for {ticker}."
        score = 0.0
    else:
        joined = " ".join(tweet.get("text", "") for tweet in tweets)
        score = _sentiment_score_from_text(joined)
        summary = f"Parsed {len(tweets)} recent X posts for {ticker}. Crowd tone is {_label(score)}."

    result = {
        "summary": summary,
        "links": [SourceLink(source="X", title=f"{ticker} search", url=f"https://x.com/search?q=%24{ticker}")],
        "score": score,
        "posts": tweets[:25],
    }
    await log_agent_activity(
        module="research",
        agent_name="X Sentiment",
        action=f"Collected X posts for {ticker}",
        status="success" if tweets else "pending",
        details={"count": len(tweets), "score": round(score, 3)},
    )
    return result


async def _reddit_summary(ticker: str, settings: Settings) -> Dict[str, Any]:
    headers = {"User-Agent": settings.reddit_user_agent or "TickerMaster/1.0"}
    company_aliases: List[str] = []
    try:
        lookup = await search_tickers(ticker, limit=1)
        if lookup:
            company_aliases = _company_alias_tokens(str(lookup[0].name or ""))
    except Exception:
        company_aliases = []
    base_params = {
        "q": f'"{ticker}" OR ${ticker} stock',
        "sort": "top",
        "t": "week",
        "limit": 15,
    }
    posts: List[Dict[str, Any]] = []
    error_msg = ""

    async def fetch_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
        return [child.get("data", {}) for child in children if isinstance(child, dict)]

    def normalize_permalink(post: Dict[str, Any]) -> str:
        permalink = str(post.get("permalink") or "").strip()
        if permalink.startswith("http://") or permalink.startswith("https://"):
            return permalink
        if permalink.startswith("/"):
            return f"https://www.reddit.com{permalink}"
        return f"https://www.reddit.com/search/?q={ticker}"

    def post_blob(post: Dict[str, Any]) -> str:
        title = str(post.get("title") or "")
        selftext = str(post.get("selftext") or "")
        return f"{title} {selftext}".strip()

    def is_allowed_subreddit(post: Dict[str, Any]) -> bool:
        subreddit = str(post.get("subreddit") or "").strip().lower()
        return subreddit in REDDIT_FOCUS_SUBREDDITS_SET

    def alias_hits(text: str) -> int:
        lower = text.lower()
        return sum(1 for token in company_aliases if re.search(rf"\b{re.escape(token)}\b", lower))

    def is_ticker_relevant(post: Dict[str, Any]) -> bool:
        title = str(post.get("title") or "")
        selftext = str(post.get("selftext") or "")
        text = f"{title} {selftext}".strip()
        if not text:
            return False

        title_has_ticker = _mentions_symbol(title, ticker)
        text_has_ticker = title_has_ticker or _mentions_symbol(text, ticker)
        alias_match_count = alias_hits(text)

        # Drop cross-ticker threads when our target is not explicitly discussed.
        other_cashtags = {
            match.upper()
            for match in re.findall(r"\$([A-Za-z]{1,5})\b", text)
            if match.upper() != ticker
        }
        if not text_has_ticker and alias_match_count < 2:
            return False
        if other_cashtags and not text_has_ticker and alias_match_count < 3:
            return False
        return True

    def post_relevance(post: Dict[str, Any]) -> float:
        if not is_ticker_relevant(post):
            return -1_000_000.0
        title = str(post.get("title") or "")
        text = post_blob(post)
        lower = text.lower()
        ticker_lower = ticker.lower()
        mentions = lower.count(ticker_lower)
        cashtag_mentions = lower.count(f"${ticker_lower}")
        alias_match_count = alias_hits(text)
        score = float(post.get("score") or 0.0)
        comments = float(post.get("num_comments") or 0.0)
        # Weight relevance to ticker plus crowd engagement.
        return (
            (mentions * 6.0)
            + (cashtag_mentions * 8.0)
            + (alias_match_count * 3.0)
            + (score * 0.015)
            + (comments * 0.04)
            + (6.0 if _mentions_symbol(title, ticker) else 0.0)
        )

    def top_terms(items: List[Dict[str, Any]], top_k: int = 4) -> List[str]:
        token_weights: Dict[str, float] = {}
        ticker_lower = ticker.lower()
        for post in items:
            text = post_blob(post).lower()
            post_weight = 1.0 + (float(post.get("score") or 0.0) / 300.0)
            for token in re.findall(r"[a-z]{3,}", text):
                if token in _REDDIT_TOKEN_STOPWORDS or token == ticker_lower:
                    continue
                if token in company_aliases:
                    continue
                if len(token) <= 3:
                    continue
                token_weights[token] = token_weights.get(token, 0.0) + post_weight
        ranked = sorted(token_weights.items(), key=lambda pair: pair[1], reverse=True)
        return [token for token, _ in ranked[:top_k]]

    def build_reddit_summary(items: List[Dict[str, Any]], net_score: float) -> str:
        if not items:
            return f"No notable Reddit discussions captured for {ticker}."

        bullish_count = 0
        bearish_count = 0
        for post in items:
            score = _sentiment_score_from_text(post_blob(post))
            if score >= 0.15:
                bullish_count += 1
            elif score <= -0.15:
                bearish_count += 1

        communities = sorted(
            {
                f"r/{str(post.get('subreddit') or '').strip()}"
                for post in items
                if str(post.get("subreddit") or "").strip()
            }
        )
        communities_str = ", ".join(communities[:4]) if communities else "core stock subreddits"

        featured = sorted(items, key=post_relevance, reverse=True)[:3]
        top_thread_lines: List[str] = []
        for post in featured:
            subreddit = str(post.get("subreddit") or "unknown")
            title = re.sub(r"\s+", " ", str(post.get("title") or "")).strip()
            if len(title) > 120:
                title = title[:117].rstrip() + "..."
            ups = int(post.get("score") or 0)
            comments = int(post.get("num_comments") or 0)
            local_score = _sentiment_score_from_text(post_blob(post))
            top_thread_lines.append(
                f"- r/{subreddit}: \"{title}\" ({ups} upvotes, {comments} comments, {_label(local_score)} tone)."
            )

        themes = top_terms(items)
        theme_line = ", ".join(themes) if themes else "earnings, positioning, and momentum"

        return "\n".join(
            [
                "**Coverage**:",
                f"- Reviewed {len(items)} high-engagement Reddit threads for {ticker} from {communities_str}.",
                f"- Net sentiment is {_label(net_score)} with {bullish_count} bullish vs {bearish_count} bearish threads.",
                "**Top Threads**:",
                *top_thread_lines,
                "**What Retail Is Watching**:",
                f"- Most repeated discussion themes: {theme_line}.",
                "- Watch whether highly-upvoted threads shift from catalyst-driven DD to short-term hype; that often precedes volatility.",
            ]
        )

    try:
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            scoped_calls = [
                fetch_json(
                    client,
                    f"https://www.reddit.com/r/{subreddit}/search.json",
                    {**base_params, "restrict_sr": "on"},
                )
                for subreddit in REDDIT_FOCUS_SUBREDDITS
            ]
            scoped_results = await asyncio.gather(*scoped_calls)

            merged: Dict[str, Dict[str, Any]] = {}
            for batch in scoped_results:
                for post in batch:
                    if not isinstance(post, dict):
                        continue
                    if not is_allowed_subreddit(post):
                        continue
                    key = str(post.get("name") or post.get("id") or "")
                    if not key:
                        continue
                    existing = merged.get(key)
                    if not existing:
                        merged[key] = post
                        continue
                    # Keep the richer record if one has longer body text.
                    if len(str(post.get("selftext") or "")) > len(str(existing.get("selftext") or "")):
                        merged[key] = post
            posts = list(merged.values())
    except Exception as exc:
        error_msg = f"Reddit fetch failed: {exc}"

    if not posts:
        summary = error_msg or f"No notable Reddit discussions captured for {ticker}."
        score = 0.0
    else:
        relevant = [post for post in posts if is_ticker_relevant(post)]
        ranked = sorted(relevant, key=post_relevance, reverse=True)[:20] if relevant else []
        if not ranked:
            summary = f"No sufficiently ticker-relevant Reddit threads found for {ticker} in core stock subreddits."
            score = 0.0
            posts = []
        else:
            posts = ranked
            joined = " ".join(post_blob(post) for post in ranked)
            score = _sentiment_score_from_text(joined)
            summary = build_reddit_summary(ranked, score)

    result = {
        "summary": summary,
        "links": (
            [
                SourceLink(source="Reddit", title=f"{ticker} search", url=f"https://www.reddit.com/search/?q={ticker}")
            ]
            if not posts
            else [
                SourceLink(
                    source="Reddit",
                    title=re.sub(r"\s+", " ", str(post.get("title") or "")).strip()[:72] or f"{ticker} thread",
                    url=normalize_permalink(post),
                )
                for post in sorted(posts, key=post_relevance, reverse=True)[:6]
            ]
        ),
        "score": score,
        "posts": posts[:20],
    }
    await log_agent_activity(
        module="research",
        agent_name="Reddit Sentiment",
        action=f"Collected Reddit posts for {ticker}",
        status="success" if posts else "pending",
        details={"count": len(posts), "score": round(score, 3)},
    )
    return result


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
    cache_key = f"research:v8:{request.timeframe}:{int(request.include_prediction_markets)}"
    cached = get_cached_research(ticker, cache_key)
    if cached:
        return ResearchResponse(**cached)

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
        await log_agent_activity(
            module="research",
            agent_name="Prediction Markets",
            action=f"Fetching prediction markets for {ticker}",
            status="running",
        )
        company_name = ""
        try:
            lookup = await search_tickers(ticker, limit=1)
            if lookup:
                company_name = str(lookup[0].name or "")
        except Exception:
            company_name = ""
        kalshi_markets, polymarket_markets = await asyncio.gather(
            fetch_kalshi_markets(ticker, settings, company_name=company_name),
            fetch_polymarket_markets(ticker, settings, company_name=company_name),
        )
        prediction_markets = sorted(
            kalshi_markets + polymarket_markets,
            key=lambda item: float(item.get("relevance_score", 0.0) or 0.0),
            reverse=True,
        )

        # Fallback: if no stock-specific markets found, fetch macro/economic markets
        # These are relevant for demos and provide context on factors affecting all stocks
        if not prediction_markets:
            macro_markets = await fetch_macro_fallback_markets(settings, limit=5)
            if macro_markets:
                prediction_markets = macro_markets
                await log_agent_activity(
                    module="research",
                    agent_name="Prediction Markets",
                    action=f"Using {len(macro_markets)} macro/economic markets as fallback for {ticker}",
                    status="success",
                    details={"fallback": True, "macro_count": len(macro_markets)},
                )
            else:
                await log_agent_activity(
                    module="research",
                    agent_name="Prediction Markets",
                    action=f"No prediction markets found for {ticker}",
                    status="pending",
                    details={"kalshi_count": 0, "polymarket_count": 0, "macro_count": 0},
                )
        else:
            await log_agent_activity(
                module="research",
                agent_name="Prediction Markets",
                action=f"Found {len(prediction_markets)} relevant prediction markets for {ticker}",
                status="success",
                details={"kalshi_count": len(kalshi_markets), "polymarket_count": len(polymarket_markets)},
            )

    # Composite weighting from spec: Perplexity 45%, Reddit 30%, X 25%
    aggregate = float((breakdown[0].score * 0.45) + (breakdown[2].score * 0.30) + (breakdown[1].score * 0.25))

    links = [
        SourceLink(source="Perplexity Sonar", title="Perplexity", url="https://www.perplexity.ai"),
        SourceLink(source="X API", title="X Developer", url="https://developer.x.com/en/docs"),
        SourceLink(source="Reddit API", title="Reddit Dev", url="https://www.reddit.com/dev/api/"),
        SourceLink(source="Kalshi API", title="Kalshi Docs", url="https://docs.kalshi.com/"),
        SourceLink(source="Polymarket", title="Polymarket", url="https://docs.polymarket.com/"),
        SourceLink(source="Morningstar", title="Morningstar", url="https://www.morningstar.com/"),
        SourceLink(source="Reuters", title="Reuters Markets", url="https://www.reuters.com/markets/"),
        SourceLink(source="J.P. Morgan", title="J.P. Morgan Insights", url="https://www.jpmorgan.com/insights"),
        SourceLink(source="Alpaca Market Data", title="Alpaca Docs", url="https://docs.alpaca.markets/docs/about-market-data-api"),
        SourceLink(source="Finnhub", title="Finnhub API", url="https://finnhub.io/"),
    ]

    response = ResearchResponse(
        ticker=ticker,
        generated_at=datetime.now(timezone.utc).isoformat(),
        aggregate_sentiment=round(aggregate, 3),
        recommendation=_recommendation(aggregate),
        narratives=_build_narratives(breakdown),
        source_breakdown=breakdown,
        prediction_markets=prediction_markets,
        tool_links=links,
    )
    set_cached_research(ticker, cache_key, response.model_dump(), ttl_minutes=15)
    await log_agent_activity(
        module="research",
        agent_name="Composite Sentiment",
        action=f"Computed weighted sentiment for {ticker}",
        status="success",
        details={"score": response.aggregate_sentiment, "recommendation": response.recommendation},
    )
    return response


async def get_x_sentiment(ticker: str, settings: Settings) -> Dict[str, Any]:
    out = await _x_summary(ticker.upper().strip(), settings)
    return {
        "ticker": ticker.upper().strip(),
        "sentiment_score": round(float(out.get("score", 0.0)), 3),
        "summary": str(out.get("summary", "")),
        "posts": out.get("posts", []),
        "links": [link.model_dump() if hasattr(link, "model_dump") else link for link in out.get("links", [])],
    }
