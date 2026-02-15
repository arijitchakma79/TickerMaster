from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Any, Dict, List

import httpx

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "tickermaster-tracker-tool-router"
SERVER_VERSION = "1.1.0"
TOOL_NAME = "route_tracker_tools"
TOOL_REDDIT = "get_reddit_sentiment"
TOOL_X = "get_x_sentiment"
TOOL_PREDICTION = "get_prediction_markets"
TOOL_PERPLEXITY = "get_perplexity_brief"
TOOL_DEEP = "get_deep_research_brief"

ALLOWED_TOOLS = {
    "price",
    "volume",
    "sentiment",
    "news",
    "prediction_markets",
    "deep_research",
    "simulation",
}
ALLOWED_SOURCES = {"perplexity", "x", "reddit", "prediction_markets", "deep"}
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


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        token = str(item or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _label(score: float) -> str:
    if score >= 0.2:
        return "bullish"
    if score <= -0.2:
        return "bearish"
    return "neutral"


def _sentiment_score_from_text(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z]+", str(text or "").lower())
    if not tokens:
        return 0.0
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    score = (pos - neg) / max(1, int(math.sqrt(len(tokens))))
    return max(-1.0, min(1.0, score))


def _safe_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(low, min(high, parsed))


def _truncate(text: str, limit: int = 320) -> str:
    token = str(text or "").strip()
    if len(token) <= limit:
        return token
    return token[: max(0, limit - 3)].rstrip() + "..."


def _route_plan(arguments: Dict[str, Any]) -> Dict[str, Any]:
    manager_prompt = str(arguments.get("manager_prompt") or "")
    available_tools = [str(item).strip().lower() for item in (arguments.get("available_tools") or [])]
    available_sources = [str(item).strip().lower() for item in (arguments.get("available_sources") or [])]
    event_hint = str(arguments.get("event_hint") or "cycle").strip().lower()

    tool_set = {token for token in available_tools if token in ALLOWED_TOOLS}
    if not tool_set:
        tool_set = set(ALLOWED_TOOLS)
    source_set = {token for token in available_sources if token in ALLOWED_SOURCES}
    if not source_set:
        source_set = set(ALLOWED_SOURCES)

    lower = f" {manager_prompt.lower()} "

    signals_price = any(token in lower for token in {" price ", " drop ", " rally ", " up ", " down ", "%", "breakout", "drawdown"})
    signals_volume = any(token in lower for token in {" volume ", "flow", "volume spike", "liquidity"})
    signals_sentiment = any(token in lower for token in {" sentiment ", "tone", "bullish", "bearish", "social"})
    signals_news = any(
        token in lower
        for token in {" news ", "headline", "catalyst", "filing", "investigate", "search", "what happened", "why moved"}
    )
    tools: List[str] = []
    if "price" in tool_set and (signals_price or not signals_volume):
        tools.append("price")
    if "volume" in tool_set and (signals_volume or signals_price):
        tools.append("volume")
    sources: List[str] = []

    source_mentions: Dict[str, tuple[str, ...]] = {
        "reddit": (" reddit ", " r/", " at reddit"),
        "perplexity": (" perplexity ",),
        "x": (" x ", " twitter ", " from x", " on x", " at x", " from twitter", " on twitter", " at twitter"),
        "prediction_markets": (" prediction market", " prediction markets", " polymarket", " kalshi", " trading market"),
        "deep": (" deep research", " deep-research", " browserbase", " at deep research"),
    }
    directional_patterns: Dict[str, tuple[str, ...]] = {
        "reddit": (" from reddit", " on reddit", " at reddit", " via reddit", " reddit sentiment"),
        "perplexity": (" from perplexity", " at perplexity", " via perplexity", " perplexity search"),
        "x": (" from x", " on x", " at x", " via x", " from twitter", " on twitter", " at twitter", " twitter sentiment"),
        "prediction_markets": (" from prediction market", " at prediction market", " from prediction markets", " from polymarket", " from kalshi", " from trading market"),
        "deep": (" from deep research", " at deep research", " via deep research", " from browserbase"),
    }

    mentioned_sources = [
        source
        for source, needles in source_mentions.items()
        if source in source_set and any(needle in lower for needle in needles)
    ]
    directional_sources = [
        source
        for source, needles in directional_patterns.items()
        if source in source_set and any(needle in lower for needle in needles)
    ]
    exclusive = any(token in lower for token in {" only ", " just ", " strictly ", " exclusively "})
    broad_source_markers = {
        " all sources",
        " cross-source",
        " cross source",
        " combine sources",
        " blend sources",
        " multi-source",
    }
    source_specific_request = bool(mentioned_sources) and not any(token in lower for token in broad_source_markers)

    if exclusive and mentioned_sources:
        sources = list(mentioned_sources)
    elif directional_sources and len(mentioned_sources) <= 1:
        sources = list(directional_sources)
    elif directional_sources:
        sources = list(_dedupe(directional_sources + mentioned_sources))
    elif source_specific_request:
        sources = list(mentioned_sources)
    elif mentioned_sources:
        sources = [token for token in ["perplexity", "x", "reddit"] if token in source_set] + mentioned_sources
    else:
        sources = [token for token in ["perplexity", "x", "reddit"] if token in source_set]

    if signals_sentiment and "sentiment" in tool_set:
        tools.append("sentiment")
    if signals_news and "news" in tool_set:
        tools.append("news")

    if any(token in lower for token in {"bad thing", "bad news", "negative", "bearish", "investigate", "search"}):
        if "news" in tool_set:
            tools.append("news")
        if "sentiment" in tool_set:
            tools.append("sentiment")
        if "perplexity" in source_set and not source_specific_request:
            sources.append("perplexity")

    source_tokens = set(sources)
    if source_tokens.intersection({"reddit", "x"}) and "sentiment" in tool_set:
        tools.append("sentiment")
    if "perplexity" in source_tokens:
        if "sentiment" in tool_set:
            tools.append("sentiment")
        if "news" in tool_set:
            tools.append("news")
    if "prediction_markets" in source_tokens and "prediction_markets" in tool_set:
        tools.append("prediction_markets")
    if "deep" in source_tokens and "deep_research" in tool_set:
        tools.append("deep_research")

    simulate_on_alert = any(token in lower for token in {"simulate", "simulation", "scenario", "backtest", "sandbox"})
    if simulate_on_alert and "simulation" in tool_set:
        tools.append("simulation")

    notification_style = "auto"
    if any(token in lower for token in {"brief", "short", "quick", "concise"}):
        notification_style = "short"
    elif any(token in lower for token in {"long", "detailed", "thesis", "full analysis", "deep dive"}):
        notification_style = "long"
    elif event_hint == "report":
        notification_style = "long"
    elif event_hint == "alert":
        notification_style = "short"

    deduped_tools = [token for token in _dedupe(tools) if token in tool_set]
    if not deduped_tools:
        deduped_tools = [token for token in ["price", "volume"] if token in tool_set] or list(tool_set)[:2]
    deduped_sources = [token for token in _dedupe(sources) if token in source_set]

    return {
        "tools": deduped_tools,
        "research_sources": deduped_sources,
        "simulate_on_alert": bool(simulate_on_alert),
        "notification_style": notification_style,
        "rationale": "mcp-tool-router",
    }


def _semantic_score(text: str, ticker: str) -> float:
    upper_text = str(text or "").upper()
    token = str(ticker or "").upper().strip()
    if not token:
        return 0.0
    if f"${token}" in upper_text:
        return 1.0
    if re.search(rf"\b{re.escape(token)}\b", upper_text):
        return 0.8
    compact = token.replace(".", "").replace("-", "")
    if compact and compact in upper_text.replace(".", "").replace("-", ""):
        return 0.35
    return 0.0


def _tool_reddit_sentiment(arguments: Dict[str, Any]) -> Dict[str, Any]:
    ticker = str(arguments.get("ticker") or "").upper().strip()
    if not ticker:
        return {"source": "reddit", "ticker": "", "score": 0.0, "summary": "Ticker missing.", "posts": []}
    limit = _safe_int(arguments.get("limit"), 15, 3, 40)
    params = {
        "q": f'"{ticker}" OR ${ticker} stock',
        "sort": "top",
        "t": "week",
        "limit": limit,
    }
    headers = {"User-Agent": os.getenv("REDDIT_USER_AGENT", "TickerMaster/1.0")}
    posts: list[dict[str, Any]] = []
    error = ""
    try:
        with httpx.Client(timeout=15.0, headers=headers) as client:
            resp = client.get("https://www.reddit.com/search.json", params=params)
            resp.raise_for_status()
            children = resp.json().get("data", {}).get("children", [])
            for child in children:
                data = child.get("data", {}) if isinstance(child, dict) else {}
                title = str(data.get("title") or "").strip()
                selftext = str(data.get("selftext") or "").strip()
                blob = f"{title} {selftext}".strip()
                rel = _semantic_score(blob, ticker)
                if rel <= 0:
                    continue
                permalink = str(data.get("permalink") or "").strip()
                if permalink.startswith("/"):
                    permalink = f"https://www.reddit.com{permalink}"
                posts.append(
                    {
                        "title": title,
                        "subreddit": str(data.get("subreddit") or ""),
                        "score": int(data.get("score") or 0),
                        "num_comments": int(data.get("num_comments") or 0),
                        "url": permalink or f"https://www.reddit.com/search/?q={ticker}",
                        "created_utc": data.get("created_utc"),
                        "relevance": round(rel, 3),
                    }
                )
            posts.sort(key=lambda item: (float(item.get("relevance") or 0.0), int(item.get("score") or 0)), reverse=True)
            posts = posts[:limit]
    except Exception as exc:
        error = str(exc)

    if posts:
        text_blob = " ".join(f"{item.get('title','')}" for item in posts)
        score = _sentiment_score_from_text(text_blob)
        summary = f"Parsed {len(posts)} Reddit threads for {ticker}; tone is {_label(score)}."
    else:
        score = 0.0
        summary = f"No high-signal Reddit threads returned for {ticker}."
        if error:
            summary = f"Reddit request failed for {ticker}: {_truncate(error, 180)}"
    return {
        "source": "reddit",
        "ticker": ticker,
        "score": round(score, 3),
        "summary": summary,
        "posts": posts,
        "error": error or None,
    }


def _tool_x_sentiment(arguments: Dict[str, Any]) -> Dict[str, Any]:
    ticker = str(arguments.get("ticker") or "").upper().strip()
    if not ticker:
        return {"source": "x", "ticker": "", "score": 0.0, "summary": "Ticker missing.", "posts": []}
    limit = _safe_int(arguments.get("limit"), 25, 5, 80)
    bearer = str(os.getenv("X_API_BEARER_TOKEN") or "").strip() or str(os.getenv("X_BEARER_TOKEN") or "").strip()
    if bearer.startswith("sk-or-"):
        bearer = ""

    posts: list[dict[str, Any]] = []
    error = ""
    if bearer:
        try:
            with httpx.Client(timeout=15.0, headers={"Authorization": f"Bearer {bearer}"}) as client:
                resp = client.get(
                    "https://api.twitter.com/2/tweets/search/recent",
                    params={
                        "query": f"${ticker} lang:en -is:retweet",
                        "max_results": min(limit, 100),
                        "tweet.fields": "created_at,text,public_metrics",
                    },
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    text = str(row.get("text") or "").strip()
                    if not text:
                        continue
                    posts.append(
                        {
                            "text": text,
                            "created_at": row.get("created_at"),
                            "public_metrics": row.get("public_metrics") if isinstance(row.get("public_metrics"), dict) else {},
                        }
                    )
        except Exception as exc:
            error = str(exc)
    else:
        error = "X bearer token missing."

    if posts:
        joined = " ".join(str(item.get("text") or "") for item in posts)
        score = _sentiment_score_from_text(joined)
        summary = f"Parsed {len(posts)} X posts for {ticker}; crowd tone is {_label(score)}."
    elif "missing" in error.lower():
        score = 0.0
        summary = f"X token missing; no live X posts fetched for {ticker}."
    else:
        score = 0.0
        summary = f"No X posts returned for {ticker}."
        if error:
            summary = f"X request failed for {ticker}: {_truncate(error, 180)}"

    return {
        "source": "x",
        "ticker": ticker,
        "score": round(score, 3),
        "summary": summary,
        "posts": posts[:limit],
        "error": error or None,
    }


def _collect_kalshi_markets(ticker: str, limit: int = 6) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                "https://api.elections.kalshi.com/trade-api/v2/markets",
                params={"limit": 40, "status": "open"},
            )
            resp.raise_for_status()
            markets = resp.json().get("markets", [])
            for item in markets:
                if not isinstance(item, dict):
                    continue
                text = " ".join(str(item.get(key) or "") for key in ("title", "subtitle", "ticker"))
                relevance = _semantic_score(text, ticker)
                if relevance <= 0:
                    continue
                out.append(
                    {
                        "source": "Kalshi",
                        "market": str(item.get("title") or item.get("ticker") or "Unknown"),
                        "yes_price": item.get("yes_bid") or item.get("yes_ask"),
                        "no_price": item.get("no_bid") or item.get("no_ask"),
                        "link": f"https://kalshi.com/markets/{str(item.get('ticker') or '').strip()}",
                        "relevance_score": round(relevance, 3),
                    }
                )
    except Exception:
        return []
    out.sort(key=lambda item: float(item.get("relevance_score") or 0.0), reverse=True)
    return out[:limit]


def _collect_polymarket_markets(ticker: str, limit: int = 6) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    gamma_url = str(os.getenv("POLYMARKET_GAMMA_URL") or "https://gamma-api.polymarket.com").rstrip("/")
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"{gamma_url}/markets",
                params={"limit": 40, "closed": "false", "search": ticker},
            )
            resp.raise_for_status()
            payload = resp.json()
            rows = payload if isinstance(payload, list) else payload.get("data", [])
            for item in rows:
                if not isinstance(item, dict):
                    continue
                text = " ".join(str(item.get(key) or "") for key in ("question", "title", "description", "slug"))
                relevance = _semantic_score(text, ticker)
                if relevance <= 0:
                    continue
                out.append(
                    {
                        "source": "Polymarket",
                        "market": str(item.get("question") or item.get("title") or "Unknown market"),
                        "probability": item.get("probability") or item.get("outcomePrices"),
                        "volume": item.get("volumeNum") or item.get("volume"),
                        "link": f"https://polymarket.com/event/{str(item.get('slug') or '').strip()}",
                        "relevance_score": round(relevance, 3),
                    }
                )
    except Exception:
        return []
    out.sort(key=lambda item: float(item.get("relevance_score") or 0.0), reverse=True)
    return out[:limit]


def _tool_prediction_markets(arguments: Dict[str, Any]) -> Dict[str, Any]:
    ticker = str(arguments.get("ticker") or "").upper().strip()
    if not ticker:
        return {"source": "prediction_markets", "ticker": "", "markets": [], "summary": "Ticker missing."}
    limit = _safe_int(arguments.get("limit"), 8, 3, 20)
    kalshi = _collect_kalshi_markets(ticker, limit=limit)
    polymarket = _collect_polymarket_markets(ticker, limit=limit)
    markets = sorted(
        [*kalshi, *polymarket],
        key=lambda item: float(item.get("relevance_score") or 0.0),
        reverse=True,
    )[:limit]
    top = markets[0] if markets else {}
    summary = (
        f"Collected {len(markets)} prediction markets for {ticker}."
        if markets
        else f"No ticker-relevant prediction markets found for {ticker}."
    )
    return {
        "source": "prediction_markets",
        "ticker": ticker,
        "summary": summary,
        "top_market": top,
        "markets": markets,
    }


def _tool_perplexity_brief(arguments: Dict[str, Any]) -> Dict[str, Any]:
    ticker = str(arguments.get("ticker") or "").upper().strip()
    context = str(arguments.get("context") or "").strip()
    prompt = str(arguments.get("prompt") or "").strip()
    api_key = str(os.getenv("PERPLEXITY_API_KEY") or "").strip()
    model = str(os.getenv("PERPLEXITY_MODEL") or "sonar").strip() or "sonar"
    if not ticker:
        return {"source": "perplexity", "ticker": "", "score": 0.0, "summary": "Ticker missing.", "citations": []}

    if not api_key:
        summary = f"Perplexity key missing. Fallback brief for {ticker}: social tone is mixed and catalyst risk remains event-driven."
        return {
            "source": "perplexity",
            "ticker": ticker,
            "score": round(_sentiment_score_from_text(summary), 3),
            "summary": summary,
            "citations": [],
            "fallback": True,
        }

    user_prompt = prompt or (
        f"Give a concise catalyst brief for {ticker}. Return 4-6 bullet points covering bullish catalysts, bearish catalysts, and what to watch next."
    )
    if context:
        user_prompt = f"{user_prompt}\n\nContext: {context[:1200]}"

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Return concise markdown bullet points only."},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            content = str((payload.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
            citations = payload.get("citations") if isinstance(payload.get("citations"), list) else []
            return {
                "source": "perplexity",
                "ticker": ticker,
                "score": round(_sentiment_score_from_text(content), 3),
                "summary": _truncate(content, 2200),
                "citations": [str(item) for item in citations[:10] if isinstance(item, str)],
                "fallback": False,
            }
    except Exception as exc:
        summary = f"Perplexity request failed for {ticker}: {_truncate(str(exc), 220)}"
        return {
            "source": "perplexity",
            "ticker": ticker,
            "score": 0.0,
            "summary": summary,
            "citations": [],
            "fallback": True,
        }


def _tool_deep_research_brief(arguments: Dict[str, Any]) -> Dict[str, Any]:
    ticker = str(arguments.get("ticker") or "").upper().strip()
    analysis_goal = str(arguments.get("analysis_goal") or "").strip()
    base_context = str(arguments.get("context") or "").strip()
    prompt = (
        f"Run a deep research brief for {ticker}. Focus on nuanced risks, second-order effects, and catalyst timing over the next 1-4 weeks."
    )
    if analysis_goal:
        prompt = f"{prompt} Goal: {analysis_goal[:300]}"
    payload = _tool_perplexity_brief(
        {
            "ticker": ticker,
            "prompt": prompt,
            "context": base_context[:1200],
        }
    )
    payload["source"] = "deep"
    return payload


def _result(payload_id: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": payload_id, "result": data}


def _error(payload_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": payload_id, "error": {"code": code, "message": message}}


def _tool_registry() -> list[dict[str, Any]]:
    return [
        {
            "name": TOOL_NAME,
            "description": "Route tracker prompts to tools/sources for monitoring, research, and simulation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "manager_prompt": {"type": "string"},
                    "available_tools": {"type": "array", "items": {"type": "string"}},
                    "available_sources": {"type": "array", "items": {"type": "string"}},
                    "event_hint": {"type": "string"},
                    "market_state": {"type": "object"},
                },
                "required": ["manager_prompt"],
            },
        },
        {
            "name": TOOL_REDDIT,
            "description": "Fetch Reddit discussion sentiment for a ticker.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
        {
            "name": TOOL_X,
            "description": "Fetch X discussion sentiment for a ticker.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
        {
            "name": TOOL_PREDICTION,
            "description": "Fetch prediction market contracts relevant to a ticker.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["ticker"],
            },
        },
        {
            "name": TOOL_PERPLEXITY,
            "description": "Fetch Perplexity catalyst brief for a ticker.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "prompt": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
        {
            "name": TOOL_DEEP,
            "description": "Fetch deep research style brief for a ticker.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "analysis_goal": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
    ]


def _call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == TOOL_NAME:
        return _route_plan(arguments)
    if name == TOOL_REDDIT:
        return _tool_reddit_sentiment(arguments)
    if name == TOOL_X:
        return _tool_x_sentiment(arguments)
    if name == TOOL_PREDICTION:
        return _tool_prediction_markets(arguments)
    if name == TOOL_PERPLEXITY:
        return _tool_perplexity_brief(arguments)
    if name == TOOL_DEEP:
        return _tool_deep_research_brief(arguments)
    raise ValueError(f"Unknown tool: {name}")


def _handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    payload_id = request.get("id")
    method = str(request.get("method") or "")
    params = request.get("params") if isinstance(request.get("params"), dict) else {}

    if method == "initialize":
        return _result(
            payload_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )

    if method == "tools/list":
        return _result(payload_id, {"tools": _tool_registry()})

    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        try:
            payload = _call_tool(name, arguments)
        except Exception as exc:
            return _error(payload_id, -32602, str(exc))
        return _result(
            payload_id,
            {
                "content": [{"type": "text", "text": f"{name} completed."}],
                "structuredContent": payload,
            },
        )

    return _error(payload_id, -32601, f"Method not found: {method}")


def main() -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception:
            response = _error(None, -32700, "Invalid JSON")
            print(json.dumps(response, ensure_ascii=True), flush=True)
            continue
        if not isinstance(request, dict):
            response = _error(None, -32600, "Invalid request payload")
            print(json.dumps(response, ensure_ascii=True), flush=True)
            continue
        response = _handle_request(request)
        print(json.dumps(response, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()