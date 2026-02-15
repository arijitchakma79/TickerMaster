from __future__ import annotations

import asyncio
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Deque, Dict, List, Tuple

import httpx
import numpy as np

from app.config import Settings
from app.schemas import AgentConfig, SimulationStartRequest, SimulationState, TradeRecord
from app.services.agent_logger import log_agent_activity
from app.services.llm import generate_agent_decision
from app.services.market_data import fetch_sp500_returns_window
from app.services.modal_engine import calibrate_personas
from app.services.reddit_client import reddit_search_posts
from app.services.simulation_store import complete_simulation_record, create_simulation_record
from app.ws_manager import WSManager


NEWS_SENTIMENT_POSITIVE = {
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "growth",
    "surge",
    "strong",
    "bullish",
    "profit",
    "record",
    "buyback",
    "outperform",
    "raise",
    "raised",
    "expansion",
}

NEWS_SENTIMENT_NEGATIVE = {
    "miss",
    "missed",
    "downgrade",
    "downgraded",
    "warning",
    "lawsuit",
    "probe",
    "weak",
    "bearish",
    "decline",
    "selloff",
    "cut",
    "cuts",
    "slump",
    "recession",
    "layoff",
    "layoffs",
}


_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_MARKDOWN_CITATION_PATTERN = re.compile(r"\s*\[(?:\d+(?:\s*,\s*\d+)*)\]")


def _sanitize_external_news_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    cleaned = _MARKDOWN_LINK_PATTERN.sub(r"\1", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    cleaned = _MARKDOWN_CITATION_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


@dataclass
class Position:
    holdings: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def _normalize_ticker(self, ticker: str) -> str:
        return ticker.upper().strip().replace(".", "-")

    def _position(self, ticker: str) -> Position:
        key = self._normalize_ticker(ticker)
        position = self.positions.get(key)
        if position is None:
            position = Position()
            self.positions[key] = position
        return position

    def holdings_for(self, ticker: str) -> int:
        key = self._normalize_ticker(ticker)
        position = self.positions.get(key)
        return position.holdings if position else 0

    def buy(self, ticker: str, qty: int, price: float) -> None:
        if qty <= 0:
            return
        position = self._position(ticker)
        total_cost = qty * price
        new_holdings = position.holdings + qty
        if new_holdings > 0:
            position.avg_cost = ((position.avg_cost * position.holdings) + total_cost) / new_holdings
        self.cash -= total_cost
        position.holdings = new_holdings

    def sell(self, ticker: str, qty: int, price: float) -> None:
        if qty <= 0:
            return
        key = self._normalize_ticker(ticker)
        position = self.positions.get(key)
        if not position:
            return
        qty = min(qty, position.holdings)
        if qty <= 0:
            return

        proceeds = qty * price
        self.cash += proceeds
        position.realized_pnl += (price - position.avg_cost) * qty
        position.holdings -= qty
        if position.holdings == 0:
            position.avg_cost = 0.0

    def mark_to_market(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        total_market_value = 0.0
        total_unrealized = 0.0
        total_realized = 0.0
        total_shares = 0
        total_cost_basis = 0.0

        positions_out: Dict[str, Dict[str, float]] = {}
        for ticker, position in sorted(self.positions.items()):
            market_price = float(current_prices.get(ticker, 0.0))
            market_value = position.holdings * market_price
            unrealized = (market_price - position.avg_cost) * position.holdings
            net_gain = position.realized_pnl + unrealized

            positions_out[ticker] = {
                "holdings": float(position.holdings),
                "avg_cost": round(position.avg_cost, 4),
                "market_price": round(market_price, 4),
                "market_value": round(market_value, 2),
                "realized_pnl": round(position.realized_pnl, 2),
                "unrealized_pnl": round(unrealized, 2),
                "net_gain": round(net_gain, 2),
            }

            total_market_value += market_value
            total_unrealized += unrealized
            total_realized += position.realized_pnl
            total_shares += position.holdings
            total_cost_basis += position.avg_cost * position.holdings

        equity = self.cash + total_market_value
        blended_avg_cost = (total_cost_basis / total_shares) if total_shares > 0 else 0.0

        return {
            "cash": round(self.cash, 2),
            "holdings": float(total_shares),
            "avg_cost": round(blended_avg_cost, 4),
            "realized_pnl": round(total_realized, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "total_pnl": round(total_realized + total_unrealized, 2),
            "equity": round(equity, 2),
            "positions": positions_out,
        }


@dataclass
class RuntimeAgent:
    config: AgentConfig
    portfolio: Portfolio
    bias_memory: float = 0.0


@dataclass
class SessionRuntime:
    session_id: str
    ticker: str
    tickers: List[str]
    duration_seconds: int
    inference_runtime: str
    started_at: datetime
    ends_at: datetime
    current_price: float
    market_prices: Dict[str, float]
    volatility: float
    agents: Dict[str, RuntimeAgent] = field(default_factory=dict)
    tick: int = 0
    crash_mode: bool = False
    recent_news: Deque[str] = field(default_factory=lambda: deque(maxlen=6))
    news_events: Deque[Tuple[int, float, str]] = field(default_factory=lambda: deque(maxlen=20))
    trades: Deque[TradeRecord] = field(default_factory=lambda: deque(maxlen=300))
    recent_prices_by_ticker: Dict[str, Deque[float]] = field(default_factory=dict)
    last_returns_by_ticker: Dict[str, float] = field(default_factory=dict)
    running: bool = True
    simulation_record_id: str | None = None
    paused: bool = False
    paused_at: datetime | None = None
    last_news_refresh_at: datetime | None = None
    news_dedupe_keys: Deque[str] = field(default_factory=lambda: deque(maxlen=250))
    news_focus_index: int = 0


class SimulationOrchestrator:
    def __init__(self, settings: Settings, ws_manager: WSManager) -> None:
        self.settings = settings
        self.ws_manager = ws_manager
        self.sessions: Dict[str, SessionRuntime] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

        self._sp500_returns, self._sp500_annualized_vol = fetch_sp500_returns_window()
        self._rng = np.random.default_rng()

    def _default_agents(self, starting_cash: float) -> List[AgentConfig]:
        return [
            AgentConfig(name="Quant Pulse", personality="quant_momentum", aggressiveness=0.72, risk_limit=0.65, trade_size=30),
            AgentConfig(name="Value Anchor", personality="fundamental_value", aggressiveness=0.42, risk_limit=0.45, trade_size=20),
            AgentConfig(name="Retail Wave", personality="retail_reactive", aggressiveness=0.58, risk_limit=0.55, trade_size=15),
        ]

    def _normalize_ticker(self, raw: str) -> str:
        return raw.upper().strip().replace(".", "-")

    def _is_valid_ticker(self, ticker: str) -> bool:
        if not ticker or len(ticker) > 10:
            return False
        return all(char.isalnum() or char == "-" for char in ticker)

    def _resolve_ticker_universe(self, primary_ticker: str, target_tickers: List[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()

        for candidate in [primary_ticker, *target_tickers]:
            ticker = self._normalize_ticker(candidate)
            if not self._is_valid_ticker(ticker):
                continue
            if ticker in seen:
                continue
            out.append(ticker)
            seen.add(ticker)
            if len(out) >= 12:
                break

        if not out:
            return ["AAPL"]
        return out

    def _seed_market_prices(self, tickers: List[str], primary_ticker: str, initial_price: float) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        for ticker in tickers:
            if ticker == primary_ticker:
                prices[ticker] = float(initial_price)
                continue
            multiplier = float(self._rng.uniform(0.55, 1.65))
            prices[ticker] = max(2.0, float(initial_price) * multiplier)
        return prices

    async def start(self, request: SimulationStartRequest) -> SimulationState:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        ends_at = now + timedelta(seconds=request.duration_seconds)

        primary_ticker = self._normalize_ticker(request.ticker) or "AAPL"
        tickers = self._resolve_ticker_universe(primary_ticker, request.target_tickers)

        agent_configs = request.agents or self._default_agents(request.starting_cash)
        runtime_agents = {
            agent.name: RuntimeAgent(config=agent, portfolio=Portfolio(cash=request.starting_cash))
            for agent in agent_configs
            if agent.active
        }

        market_prices = self._seed_market_prices(tickers, primary_ticker, request.initial_price)
        current_price = market_prices.get(primary_ticker, request.initial_price)

        runtime = SessionRuntime(
            session_id=session_id,
            ticker=primary_ticker,
            tickers=tickers,
            duration_seconds=request.duration_seconds,
            inference_runtime=request.inference_runtime,
            started_at=now,
            ends_at=ends_at,
            current_price=current_price,
            market_prices=market_prices,
            volatility=request.volatility,
            agents=runtime_agents,
            recent_prices_by_ticker={
                ticker: deque([price], maxlen=50)
                for ticker, price in market_prices.items()
            },
            last_returns_by_ticker={ticker: 0.0 for ticker in tickers},
            simulation_record_id=create_simulation_record(
                config=request.model_dump(),
                user_id=request.user_id,
            ),
        )
        self.sessions[session_id] = runtime
        self.tasks[session_id] = asyncio.create_task(self._run_loop(runtime), name=f"simulation-{session_id}")

        # Calibrate known personas from public profiles (SEC 13F summaries) without blocking session start.
        if request.inference_runtime == "modal" and runtime_agents:
            asyncio.create_task(self._calibrate_runtime_personas(runtime), name=f"persona-calibration-{session_id}")

        await self.ws_manager.broadcast(
            {
                "channel": "simulation",
                "type": "simulation_started",
                "session_id": session_id,
                "ticker": runtime.ticker,
                "tickers": runtime.tickers,
                "timestamp": now.isoformat(),
            },
            channel="simulation",
        )
        await log_agent_activity(
            module="simulation",
            agent_name="Simulation Orchestrator",
            action=f"Started simulation {session_id} for {runtime.ticker}",
            status="success",
            details={
                "agents": list(runtime_agents.keys()),
                "duration_seconds": request.duration_seconds,
                "tickers": runtime.tickers,
            },
        )

        return self._to_state(runtime)

    async def _calibrate_runtime_personas(self, runtime: SessionRuntime) -> None:
        try:
            configs = [entry.config for entry in runtime.agents.values()]
            calibrated = await calibrate_personas(self.settings, configs, prefer_modal=True, max_concurrency=2)
            updated = 0
            for config in calibrated:
                slot = runtime.agents.get(config.name)
                if not slot:
                    continue
                slot.config = config
                updated += 1

            if updated:
                await log_agent_activity(
                    module="simulation",
                    agent_name="Persona Calibrator",
                    action=f"Updated {updated} persona configs from public profiles",
                    status="success",
                    details={"session_id": runtime.session_id, "updated": updated},
                )
        except Exception as exc:
            await log_agent_activity(
                module="simulation",
                agent_name="Persona Calibrator",
                action="Persona calibration failed",
                status="error",
                details={"session_id": runtime.session_id, "error": str(exc)},
            )

    async def stop(self, session_id: str) -> bool:
        runtime = self.sessions.get(session_id)
        if not runtime:
            return False

        runtime.running = False
        runtime.paused = False
        runtime.paused_at = None
        task = self.tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await task

        await self.ws_manager.broadcast(
            {
                "channel": "simulation",
                "type": "simulation_stopped",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            channel="simulation",
        )
        if runtime.simulation_record_id:
            complete_simulation_record(
                record_id=runtime.simulation_record_id,
                results=self._to_state(runtime).model_dump(),
                status="completed",
            )
        await log_agent_activity(
            module="simulation",
            agent_name="Simulation Orchestrator",
            action=f"Stopped simulation {session_id}",
            status="success",
        )
        return True

    async def pause(self, session_id: str) -> SimulationState | None:
        runtime = self.sessions.get(session_id)
        if not runtime or not runtime.running:
            return None
        if runtime.paused:
            return self._to_state(runtime)

        runtime.paused = True
        runtime.paused_at = datetime.now(timezone.utc)
        await self.ws_manager.broadcast(
            {
                "channel": "simulation",
                "type": "simulation_paused",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            channel="simulation",
        )
        return self._to_state(runtime)

    async def resume(self, session_id: str) -> SimulationState | None:
        runtime = self.sessions.get(session_id)
        if not runtime or not runtime.running:
            return None
        if not runtime.paused:
            return self._to_state(runtime)

        now = datetime.now(timezone.utc)
        if runtime.paused_at:
            runtime.ends_at = runtime.ends_at + (now - runtime.paused_at)
        runtime.paused = False
        runtime.paused_at = None
        await self.ws_manager.broadcast(
            {
                "channel": "simulation",
                "type": "simulation_resumed",
                "session_id": session_id,
                "timestamp": now.isoformat(),
            },
            channel="simulation",
        )
        return self._to_state(runtime)

    def get(self, session_id: str) -> SimulationState | None:
        runtime = self.sessions.get(session_id)
        if not runtime:
            return None
        return self._to_state(runtime)

    def list(self) -> List[SimulationState]:
        return [self._to_state(runtime) for runtime in self.sessions.values()]

    def _order_book(self, mid: float, volatility: float) -> Dict[str, List[Dict[str, float]]]:
        spread = max(0.01, mid * 0.0008 * (1 + volatility * 12))
        bids = []
        asks = []
        for level in range(1, 6):
            offset = spread * level
            size = int(1200 / level)
            bids.append({"price": round(mid - offset, 2), "size": size})
            asks.append({"price": round(mid + offset, 2), "size": size})
        return {"bids": bids, "asks": asks}

    def _sample_market_return(self, target_vol: float) -> float:
        raw = float(self._rng.choice(self._sp500_returns))
        scale = target_vol / max(1e-6, (self._sp500_annualized_vol / np.sqrt(252)))
        scaled = raw * scale

        # Random tail shocks to surface crash scenarios.
        if self._rng.random() < 0.015:
            scaled -= float(self._rng.uniform(0.03, 0.09))
        return scaled

    def _news_sentiment_score(self, text: str) -> float:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        if not tokens:
            return 0.0

        positive = sum(token in NEWS_SENTIMENT_POSITIVE for token in tokens)
        negative = sum(token in NEWS_SENTIMENT_NEGATIVE for token in tokens)
        if positive == 0 and negative == 0:
            return 0.0

        score = (positive - negative) / max(1, int(np.sqrt(len(tokens))))
        return float(max(-1.0, min(1.0, score)))

    def _parse_news_timestamp(self, raw: Any) -> datetime | None:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            if raw <= 0:
                return None
            try:
                return datetime.fromtimestamp(float(raw), tz=timezone.utc)
            except Exception:
                return None
        if isinstance(raw, str):
            value = raw.strip()
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                pass
            try:
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except Exception:
                return None
        return None

    def _safe_float(self, raw: Any) -> float | None:
        try:
            if raw is None:
                return None
            value = float(raw)
            if np.isnan(value):
                return None
            return value
        except Exception:
            return None

    def _next_focus_ticker(self, runtime: SessionRuntime) -> str:
        if not runtime.tickers:
            return runtime.ticker
        index = runtime.news_focus_index % len(runtime.tickers)
        runtime.news_focus_index = (runtime.news_focus_index + 1) % max(1, len(runtime.tickers))
        return runtime.tickers[index]

    def _record_news_event(
        self,
        runtime: SessionRuntime,
        ticker: str,
        headline: str,
        *,
        source: str = "",
        published_at: datetime | None = None,
        sentiment_override: float | None = None,
    ) -> bool:
        clean_headline = " ".join(headline.split()).strip()
        clean_ticker = self._normalize_ticker(ticker or runtime.ticker)
        if not clean_headline:
            return False

        key = f"{clean_ticker}|{clean_headline.lower()}"
        if published_at:
            key = f"{key}|{published_at.replace(microsecond=0).isoformat()}"
        if key in runtime.news_dedupe_keys:
            return False
        runtime.news_dedupe_keys.append(key)

        sentiment = sentiment_override if sentiment_override is not None else self._news_sentiment_score(clean_headline)
        if runtime.crash_mode:
            sentiment = min(sentiment, -0.05)

        source_prefix = f"{clean_ticker} | {source}" if source else clean_ticker
        display = f"{source_prefix}: {clean_headline}"

        runtime.news_events.append((runtime.tick, float(max(-1.0, min(1.0, sentiment))), display))
        runtime.recent_news.appendleft(display)
        return True

    async def _fetch_yahoo_news(self, ticker: str) -> List[Dict[str, Any]]:
        symbol = self._normalize_ticker(ticker)
        if not symbol:
            return []

        try:
            async with httpx.AsyncClient(timeout=6.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
                response = await client.get(
                    "https://query1.finance.yahoo.com/v1/finance/search",
                    params={"q": symbol, "newsCount": 8, "quotesCount": 0},
                )
                if response.status_code in {404, 429}:
                    return []
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        out: List[Dict[str, Any]] = []
        items = payload.get("news", []) if isinstance(payload, dict) else []
        for item in items[:6]:
            if not isinstance(item, dict):
                continue
            headline = str(item.get("title") or "").strip()
            if not headline:
                continue
            out.append(
                {
                    "ticker": symbol,
                    "headline": headline,
                    "source": "Yahoo",
                    "published_at": self._parse_news_timestamp(item.get("providerPublishTime")),
                }
            )
        return out

    async def _fetch_reddit_news(self, ticker: str) -> List[Dict[str, Any]]:
        symbol = self._normalize_ticker(ticker)
        if not symbol:
            return []

        children = await reddit_search_posts(
            self.settings,
            query=f"${symbol} OR {symbol} stock",
            sort="new",
            timeframe="day",
            limit=8,
            timeout_seconds=6.0,
        )
        out: List[Dict[str, Any]] = []
        for post in children[:6]:
            if not isinstance(post, dict):
                continue
            title = str(post.get("title") or "").strip()
            if not title:
                continue
            text_blob = f"{title} {str(post.get('selftext') or '')}".upper()
            symbol_pattern = re.compile(rf"(^|[^A-Z0-9])\$?{re.escape(symbol)}([^A-Z0-9]|$)")
            if not symbol_pattern.search(text_blob):
                continue
            subreddit = str(post.get("subreddit") or "").strip()
            headline = f"r/{subreddit}: {title}" if subreddit else title
            out.append(
                {
                    "ticker": symbol,
                    "headline": headline,
                    "source": "Reddit",
                    "published_at": self._parse_news_timestamp(post.get("created_utc")),
                }
            )
        return out

    async def _fetch_x_news(self, ticker: str) -> List[Dict[str, Any]]:
        symbol = self._normalize_ticker(ticker)
        if not symbol:
            return []

        bearer = self.settings.x_api_bearer_token.strip()
        if bearer.startswith("sk-or-"):
            bearer = ""
        has_oauth = bool(
            self.settings.x_consumer_key
            and self.settings.x_consumer_secret
            and self.settings.x_access_token
            and self.settings.x_access_token_secret
        )
        if not bearer and not has_oauth:
            return []

        try:
            from app.services.sentiment import get_x_sentiment

            payload = await get_x_sentiment(symbol, self.settings)
        except Exception:
            return []

        posts = payload.get("posts", []) if isinstance(payload, dict) else []
        out: List[Dict[str, Any]] = []
        for post in posts[:6]:
            if not isinstance(post, dict):
                continue
            text = str(post.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue
            out.append(
                {
                    "ticker": symbol,
                    "headline": text,
                    "source": "X",
                    "published_at": self._parse_news_timestamp(post.get("created_at")),
                }
            )
        return out

    async def _fetch_perplexity_news(self, ticker: str) -> List[Dict[str, Any]]:
        symbol = self._normalize_ticker(ticker)
        if not symbol or not self.settings.perplexity_api_key:
            return []

        headers = {
            "Authorization": f"Bearer {self.settings.perplexity_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.settings.perplexity_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a market news summarizer. Return concise bullets only.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Give 3 latest market-moving headlines or catalysts for {symbol} in the last 24 hours. "
                        "Return short bullet lines only, no intro. No markdown. No citation markers like [1]."
                    ),
                },
            ],
            "temperature": 0.1,
        }

        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                response = await client.post("https://api.perplexity.ai/chat/completions", headers=headers, json=body)
                if response.status_code in {401, 403, 429}:
                    return []
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        choices = payload.get("choices", []) if isinstance(payload, dict) else []
        content = ""
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                content = str(message.get("content") or "")
        if not content:
            return []

        lines: List[str] = []
        for raw_line in content.splitlines():
            clean = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw_line).strip()
            clean = _sanitize_external_news_text(clean)
            if clean:
                lines.append(clean)

        if not lines:
            lines = [
                _sanitize_external_news_text(part.strip())
                for part in re.split(r"(?<=[.!?])\s+", content)
                if part.strip()
            ]
            lines = [line for line in lines if line]

        now = datetime.now(timezone.utc)
        return [
            {
                "ticker": symbol,
                "headline": line,
                "source": "Perplexity Sonar",
                "published_at": now,
            }
            for line in lines[:3]
        ]

    async def _fetch_polymarket_news(self, ticker: str) -> List[Dict[str, Any]]:
        symbol = self._normalize_ticker(ticker)
        if not symbol:
            return []

        endpoint = f"{self.settings.polymarket_gamma_url.rstrip('/')}/markets"
        finance_keywords = {
            "stock",
            "stocks",
            "market",
            "nasdaq",
            "s&p",
            "earnings",
            "fed",
            "inflation",
            "rate",
            "recession",
            "economy",
            "treasury",
            "etf",
            "option",
        }

        def parse_items(items: List[Any], *, require_symbol: bool) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for item in items[:8]:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question") or item.get("title") or "").strip()
                if not question:
                    continue

                lower_question = question.lower()
                upper_question = question.upper()
                has_finance_signal = any(keyword in lower_question for keyword in finance_keywords)
                has_symbol = symbol in upper_question
                if require_symbol and not has_symbol:
                    continue
                if not has_symbol and not has_finance_signal:
                    continue

                probability = self._safe_float(item.get("probability"))
                if probability is None:
                    outcomes = item.get("outcomePrices")
                    if isinstance(outcomes, list) and outcomes:
                        probability = self._safe_float(outcomes[0])
                if probability is not None and probability > 1:
                    probability = probability / 100.0

                volume = self._safe_float(item.get("volumeNum") or item.get("volume"))
                headline = question
                if probability is not None and 0 <= probability <= 1:
                    headline = f"{headline} (yes {probability * 100:.0f}%)"
                if volume is not None and volume > 0:
                    headline = f"{headline} | vol {volume:,.0f}"

                out.append(
                    {
                        "ticker": symbol,
                        "headline": headline,
                        "source": "Polymarket",
                        "published_at": self._parse_news_timestamp(item.get("updatedAt") or item.get("createdAt")),
                    }
                )
            return out

        async def fetch(search: str) -> List[Any]:
            try:
                async with httpx.AsyncClient(timeout=6.0) as client:
                    response = await client.get(endpoint, params={"limit": 12, "closed": "false", "search": search})
                    if response.status_code in {404, 429}:
                        return []
                    response.raise_for_status()
                    payload = response.json()
            except Exception:
                return []

            data = payload if isinstance(payload, list) else payload.get("data", []) if isinstance(payload, dict) else []
            return data if isinstance(data, list) else []

        ticker_items = await fetch(symbol)
        ticker_news = parse_items(ticker_items, require_symbol=True)
        if ticker_news:
            return ticker_news

        market_items = await fetch("stock market")
        return parse_items(market_items, require_symbol=False)

    async def _fetch_multi_source_news(self, runtime: SessionRuntime, *, include_perplexity: bool) -> List[Dict[str, Any]]:
        focus_ticker = self._next_focus_ticker(runtime)
        tasks: List[Any] = [
            self._fetch_yahoo_news(focus_ticker),
            self._fetch_reddit_news(focus_ticker),
            self._fetch_x_news(focus_ticker),
            self._fetch_polymarket_news(focus_ticker),
        ]
        if include_perplexity:
            tasks.append(self._fetch_perplexity_news(focus_ticker))

        batches = await asyncio.gather(*tasks, return_exceptions=True)
        merged: List[Dict[str, Any]] = []
        for batch in batches:
            if isinstance(batch, Exception):
                continue
            merged.extend(batch)

        merged.sort(
            key=lambda item: item.get("published_at") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return merged[:18]

    def _append_synthetic_news(self, runtime: SessionRuntime) -> None:
        if self._rng.random() > 0.18:
            return

        sentiment = float(self._rng.uniform(-1, 1))
        if runtime.crash_mode:
            sentiment = min(sentiment, -0.4)

        focus_ticker = runtime.tickers[int(self._rng.integers(0, len(runtime.tickers)))]

        if sentiment > 0.25:
            headline = f"{focus_ticker}: positive catalyst with upgrades and demand commentary."
        elif sentiment < -0.25:
            headline = f"{focus_ticker}: negative catalyst with macro risk-off and guidance concerns."
        else:
            headline = f"{focus_ticker}: mixed catalyst with conflicting macro and company signals."

        self._record_news_event(
            runtime,
            ticker=focus_ticker,
            headline=headline.split(":", 1)[-1].strip(),
            source="Simulation",
            sentiment_override=sentiment,
        )

    async def _maybe_news_event(self, runtime: SessionRuntime) -> None:
        now = datetime.now(timezone.utc)
        poll_due = (
            runtime.last_news_refresh_at is None
            or (now - runtime.last_news_refresh_at) >= timedelta(seconds=15)
        )

        added = 0
        if poll_due:
            runtime.last_news_refresh_at = now
            include_perplexity = runtime.tick <= 2 or (runtime.tick % 30 == 0)
            try:
                external_news = await asyncio.wait_for(
                    self._fetch_multi_source_news(runtime, include_perplexity=include_perplexity),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                external_news = []

            seen_sources: set[str] = set()
            for item in external_news:
                source_name = str(item.get("source") or "").strip().lower()
                if source_name and source_name in seen_sources and len(seen_sources) < 3:
                    continue
                if self._record_news_event(
                    runtime,
                    ticker=str(item.get("ticker") or runtime.ticker),
                    headline=str(item.get("headline") or ""),
                    source=str(item.get("source") or ""),
                    published_at=item.get("published_at"),
                ):
                    added += 1
                    if source_name:
                        seen_sources.add(source_name)
                if added >= 4:
                    break

        # Keep a minimal fallback so the card is never empty in case APIs are unavailable.
        if added == 0 and not runtime.recent_news:
            self._append_synthetic_news(runtime)

    def _news_bias_for_agent(self, runtime: SessionRuntime, personality: str) -> float:
        lag = {"quant_momentum": 1, "fundamental_value": 3, "retail_reactive": 5}.get(personality, 3)
        sentiment_sum = 0.0
        for event_tick, sentiment, _headline in runtime.news_events:
            if runtime.tick - event_tick >= lag:
                sentiment_sum += sentiment
        return sentiment_sum

    def _ticker_features(self, runtime: SessionRuntime, ticker: str) -> Dict[str, float]:
        prices = list(runtime.recent_prices_by_ticker.get(ticker, deque([runtime.market_prices.get(ticker, runtime.current_price)], maxlen=50)))
        latest_price = float(runtime.market_prices.get(ticker, runtime.current_price))

        short_momentum = 0.0
        if len(prices) >= 6 and prices[-6] > 0:
            short_momentum = (prices[-1] - prices[-6]) / prices[-6]

        fair_value = float(np.mean(prices[-12:])) if len(prices) >= 12 else latest_price
        deviation = (fair_value - latest_price) / max(1e-6, latest_price)

        return {
            "short_momentum": short_momentum,
            "deviation": deviation,
            "price": latest_price,
        }

    async def _policy_decision(self, agent: RuntimeAgent, runtime: SessionRuntime) -> Dict[str, Any]:
        config = agent.config
        news_bias = self._news_bias_for_agent(runtime, config.personality)

        target_ticker = runtime.ticker
        side = "hold"
        confidence = 0.4
        rationale = "No strong signal."
        quantity = 0
        selected_features = self._ticker_features(runtime, target_ticker)

        for ticker in runtime.tickers:
            features = self._ticker_features(runtime, ticker)
            short_momentum = features["short_momentum"]
            deviation = features["deviation"]

            ticker_side = "hold"
            ticker_confidence = 0.4
            ticker_rationale = f"{ticker}: No strong signal."

            if config.personality == "quant_momentum":
                signal = (short_momentum * 6) + (news_bias * 0.2)
                if signal > 0.06:
                    ticker_side = "buy"
                    ticker_confidence = min(0.92, 0.5 + signal)
                    ticker_rationale = f"{ticker}: momentum and lead-time news signal favor upside continuation."
                elif signal < -0.06:
                    ticker_side = "sell"
                    ticker_confidence = min(0.92, 0.5 + abs(signal))
                    ticker_rationale = f"{ticker}: momentum reversal and early news skew suggest downside pressure."
            elif config.personality == "fundamental_value":
                signal = (deviation * 10) + (news_bias * 0.15)
                if signal > 0.08:
                    ticker_side = "buy"
                    ticker_confidence = min(0.9, 0.45 + signal)
                    ticker_rationale = f"{ticker}: trading below rolling fair value with manageable macro risk."
                elif signal < -0.08:
                    ticker_side = "sell"
                    ticker_confidence = min(0.9, 0.45 + abs(signal))
                    ticker_rationale = f"{ticker}: trading above fair value with weaker catalyst quality."
            else:
                signal = (news_bias * 0.4) + (short_momentum * 2)
                if signal > 0.1:
                    ticker_side = "buy"
                    ticker_confidence = min(0.88, 0.4 + signal)
                    ticker_rationale = f"{ticker}: crowd flow turned bullish after delayed catalyst diffusion."
                elif signal < -0.1:
                    ticker_side = "sell"
                    ticker_confidence = min(0.88, 0.4 + abs(signal))
                    ticker_rationale = f"{ticker}: crowd flow turned bearish as negative narrative spread."

            should_take = False
            if ticker_side != "hold" and side == "hold":
                should_take = True
            elif ticker_side != "hold" and ticker_confidence > confidence:
                should_take = True

            if should_take:
                target_ticker = ticker
                side = ticker_side
                confidence = ticker_confidence
                rationale = ticker_rationale
                selected_features = features

        if side != "hold":
            quantity = int(max(1, config.trade_size * (0.6 + config.aggressiveness)))

        # Periodically let the model refine the action.
        if runtime.tick % 6 == 0:
            llm = await generate_agent_decision(
                settings=self.settings,
                agent_name=config.name,
                model=config.model,
                personality=config.personality,
                market_state={
                    "ticker": target_ticker,
                    "primary_ticker": runtime.ticker,
                    "price": selected_features["price"],
                    "tickers": runtime.tickers,
                    "prices": {ticker: round(price, 4) for ticker, price in runtime.market_prices.items()},
                    "tick": runtime.tick,
                    "volatility": runtime.volatility,
                    "short_momentum": selected_features["short_momentum"],
                    "deviation": selected_features["deviation"],
                    "news_bias": news_bias,
                    "crash_mode": runtime.crash_mode,
                },
                user_constraints={
                    "risk_limit": config.risk_limit,
                    "aggressiveness": config.aggressiveness,
                    "max_trade_size": config.trade_size * 4,
                    "strategy_prompt": config.strategy_prompt,
                    "allowed_tickers": runtime.tickers,
                    "current_holdings": {
                        ticker: agent.portfolio.holdings_for(ticker)
                        for ticker in runtime.tickers
                    },
                },
                use_modal_inference=runtime.inference_runtime == "modal",
            )

            llm_side = llm.get("side", "hold")
            llm_quantity = int(llm.get("quantity", 0) or 0)
            llm_ticker = str(llm.get("target_ticker", "")).upper().strip().replace(".", "-")

            if llm_side in {"buy", "sell", "hold"}:
                side = llm_side
            if llm_quantity >= 0:
                quantity = min(max(0, llm_quantity), config.trade_size * 4)
            if llm_ticker and llm_ticker in runtime.tickers:
                target_ticker = llm_ticker
            confidence = float(max(0, min(1, llm.get("confidence", confidence))))
            rationale = str(llm.get("rationale", rationale))

        if side != "hold":
            risk_scalar = max(0.1, 1 - (runtime.volatility * 8)) * config.risk_limit
            quantity = int(max(1, quantity * risk_scalar))

            # Confidence gate to reduce overtrading and transaction-cost bleed.
            min_confidence = 0.5 + (runtime.volatility * 0.6)
            if confidence < min_confidence:
                side = "hold"
                quantity = 0
                rationale = f"{target_ticker}: signal confidence below execution threshold."

        return {
            "side": side,
            "quantity": quantity,
            "confidence": confidence,
            "rationale": rationale,
            "target_ticker": target_ticker,
        }

    def _validate_trade(self, side: str, qty: int, portfolio: Portfolio, ticker: str, price: float) -> int:
        if qty <= 0:
            return 0
        if side == "buy":
            affordable = int(portfolio.cash // price)
            return max(0, min(qty, affordable))
        if side == "sell":
            return max(0, min(qty, portfolio.holdings_for(ticker)))
        return 0

    def _execute_trade(self, runtime: SessionRuntime, agent: RuntimeAgent, ticker: str, side: str, qty: int, rationale: str) -> TradeRecord | None:
        reference_price = float(runtime.market_prices.get(ticker, runtime.current_price))
        valid_qty = self._validate_trade(side, qty, agent.portfolio, ticker, reference_price)
        if valid_qty <= 0:
            return None

        direction = 1 if side == "buy" else -1
        spread_bps = 0.6 + (runtime.volatility * 85)
        impact_bps = (valid_qty / 4000) * (3 + runtime.volatility * 60)
        slippage_bps = spread_bps + impact_bps
        fill_price = reference_price * (1 + direction * slippage_bps / 10_000)

        if side == "buy":
            agent.portfolio.buy(ticker, valid_qty, fill_price)
        else:
            agent.portfolio.sell(ticker, valid_qty, fill_price)

        # Price impact from execution on the traded ticker.
        runtime.market_prices[ticker] = max(0.5, reference_price * (1 + direction * (impact_bps / 60_000)))
        runtime.current_price = float(runtime.market_prices.get(runtime.ticker, runtime.current_price))

        trade = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=runtime.session_id,
            ticker=ticker,
            side=side,
            agent=agent.config.name,
            quantity=valid_qty,
            price=round(fill_price, 4),
            slippage_bps=round(slippage_bps, 2),
            rationale=rationale,
        )
        runtime.trades.appendleft(trade)
        asyncio.create_task(
            log_agent_activity(
                module="simulation",
                agent_name=agent.config.name,
                action=f"{side.upper()} {valid_qty} {ticker} @ {round(fill_price, 2)}",
                status="success",
                details={"slippage_bps": round(slippage_bps, 2), "rationale": rationale},
            )
        )
        return trade

    async def _run_loop(self, runtime: SessionRuntime) -> None:
        while runtime.running and datetime.now(timezone.utc) < runtime.ends_at:
            if runtime.paused:
                await asyncio.sleep(0.2)
                continue

            runtime.tick += 1

            market_return = self._sample_market_return(runtime.volatility)
            if not runtime.crash_mode:
                market_return += 0.0001
            for ticker in runtime.tickers:
                idiosyncratic = float(self._rng.normal(0, runtime.volatility * 0.25))
                trend_memory = float(runtime.last_returns_by_ticker.get(ticker, 0.0))
                ticker_return = market_return + idiosyncratic + (0.24 * trend_memory)
                runtime.market_prices[ticker] = max(0.5, runtime.market_prices[ticker] * (1 + ticker_return))
                runtime.last_returns_by_ticker[ticker] = ticker_return
            runtime.current_price = float(runtime.market_prices.get(runtime.ticker, runtime.current_price))

            if market_return < -0.035:
                runtime.crash_mode = True
                self._record_news_event(
                    runtime,
                    ticker=runtime.ticker,
                    headline="Crash alert: liquidity thinned and spreads widened.",
                    source="Market Update",
                    published_at=datetime.now(timezone.utc),
                    sentiment_override=-0.9,
                )
            elif runtime.crash_mode and self._rng.random() < 0.08:
                runtime.crash_mode = False

            await self._maybe_news_event(runtime)

            trades_this_tick: List[TradeRecord] = []
            for agent in runtime.agents.values():
                decision = await self._policy_decision(agent, runtime)
                if decision["side"] == "hold":
                    continue
                trade = self._execute_trade(
                    runtime=runtime,
                    agent=agent,
                    ticker=str(decision.get("target_ticker") or runtime.ticker),
                    side=decision["side"],
                    qty=decision["quantity"],
                    rationale=decision["rationale"],
                )
                if trade:
                    trades_this_tick.append(trade)

            for ticker, price in runtime.market_prices.items():
                runtime.recent_prices_by_ticker.setdefault(ticker, deque(maxlen=50)).append(price)

            tick_event = {
                "channel": "simulation",
                "type": "tick",
                "session_id": runtime.session_id,
                "ticker": runtime.ticker,
                "tickers": runtime.tickers,
                "tick": runtime.tick,
                "price": round(runtime.current_price, 4),
                "market_prices": {ticker: round(price, 4) for ticker, price in runtime.market_prices.items()},
                "crash_mode": runtime.crash_mode,
                "news": list(runtime.recent_news),
                "order_book": self._order_book(runtime.current_price, runtime.volatility),
                "portfolio_snapshot": {
                    name: agent.portfolio.mark_to_market(runtime.market_prices)
                    for name, agent in runtime.agents.items()
                },
                "trades": [trade.model_dump() for trade in trades_this_tick],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.ws_manager.broadcast(tick_event, channel="simulation")
            await asyncio.sleep(1)

        runtime.running = False
        if runtime.simulation_record_id:
            complete_simulation_record(
                record_id=runtime.simulation_record_id,
                results=self._to_state(runtime).model_dump(),
                status="completed",
            )
        await self.ws_manager.broadcast(
            {
                "channel": "simulation",
                "type": "simulation_completed",
                "session_id": runtime.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            channel="simulation",
        )

    def _to_state(self, runtime: SessionRuntime) -> SimulationState:
        return SimulationState(
            session_id=runtime.session_id,
            ticker=runtime.ticker,
            tickers=runtime.tickers,
            running=runtime.running,
            paused=runtime.paused,
            tick=runtime.tick,
            current_price=round(runtime.current_price, 4),
            market_prices={ticker: round(price, 4) for ticker, price in runtime.market_prices.items()},
            volatility=runtime.volatility,
            started_at=runtime.started_at.isoformat(),
            ends_at=runtime.ends_at.isoformat(),
            crash_mode=runtime.crash_mode,
            recent_news=list(runtime.recent_news),
            trades=list(runtime.trades),
            portfolios={
                name: agent.portfolio.mark_to_market(runtime.market_prices)
                for name, agent in runtime.agents.items()
            },
            order_book=self._order_book(runtime.current_price, runtime.volatility),
        )
