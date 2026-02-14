from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

from app.config import Settings
from app.schemas import AgentConfig, SimulationStartRequest, SimulationState, TradeRecord
from app.services.llm import generate_agent_decision
from app.services.market_data import fetch_sp500_returns_window
from app.ws_manager import WSManager


@dataclass
class Portfolio:
    cash: float
    holdings: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    def buy(self, qty: int, price: float) -> None:
        if qty <= 0:
            return
        total_cost = qty * price
        new_holdings = self.holdings + qty
        if new_holdings > 0:
            self.avg_cost = ((self.avg_cost * self.holdings) + total_cost) / new_holdings
        self.cash -= total_cost
        self.holdings = new_holdings

    def sell(self, qty: int, price: float) -> None:
        if qty <= 0:
            return
        qty = min(qty, self.holdings)
        proceeds = qty * price
        self.cash += proceeds
        self.realized_pnl += (price - self.avg_cost) * qty
        self.holdings -= qty
        if self.holdings == 0:
            self.avg_cost = 0.0

    def mark_to_market(self, current_price: float) -> Dict[str, float]:
        unrealized = (current_price - self.avg_cost) * self.holdings
        equity = self.cash + (self.holdings * current_price)
        return {
            "cash": round(self.cash, 2),
            "holdings": float(self.holdings),
            "avg_cost": round(self.avg_cost, 4),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "equity": round(equity, 2),
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
    duration_seconds: int
    started_at: datetime
    ends_at: datetime
    current_price: float
    volatility: float
    agents: Dict[str, RuntimeAgent] = field(default_factory=dict)
    tick: int = 0
    crash_mode: bool = False
    recent_news: Deque[str] = field(default_factory=lambda: deque(maxlen=6))
    news_events: Deque[Tuple[int, float, str]] = field(default_factory=lambda: deque(maxlen=20))
    trades: Deque[TradeRecord] = field(default_factory=lambda: deque(maxlen=200))
    recent_prices: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    running: bool = True


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

    async def start(self, request: SimulationStartRequest) -> SimulationState:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        ends_at = now + timedelta(seconds=request.duration_seconds)

        agent_configs = request.agents or self._default_agents(request.starting_cash)
        runtime_agents = {
            agent.name: RuntimeAgent(config=agent, portfolio=Portfolio(cash=request.starting_cash))
            for agent in agent_configs
            if agent.active
        }

        runtime = SessionRuntime(
            session_id=session_id,
            ticker=request.ticker.upper().strip(),
            duration_seconds=request.duration_seconds,
            started_at=now,
            ends_at=ends_at,
            current_price=request.initial_price,
            volatility=request.volatility,
            agents=runtime_agents,
            recent_prices=deque([request.initial_price], maxlen=50),
        )
        self.sessions[session_id] = runtime
        self.tasks[session_id] = asyncio.create_task(self._run_loop(runtime), name=f"simulation-{session_id}")

        await self.ws_manager.broadcast(
            {
                "channel": "simulation",
                "type": "simulation_started",
                "session_id": session_id,
                "ticker": runtime.ticker,
                "timestamp": now.isoformat(),
            },
            channel="simulation",
        )

        return self._to_state(runtime)

    async def stop(self, session_id: str) -> bool:
        runtime = self.sessions.get(session_id)
        if not runtime:
            return False

        runtime.running = False
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
        return True

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

    def _maybe_news_event(self, runtime: SessionRuntime) -> None:
        if self._rng.random() > 0.18:
            return

        sentiment = float(self._rng.uniform(-1, 1))
        if runtime.crash_mode:
            sentiment = min(sentiment, -0.4)

        if sentiment > 0.25:
            headline = "Positive catalyst: analyst upgrades and strong demand commentary."
        elif sentiment < -0.25:
            headline = "Negative catalyst: macro risk-off move and guidance concerns."
        else:
            headline = "Mixed catalyst: conflicting signals across macro and company-specific updates."

        runtime.news_events.append((runtime.tick, sentiment, headline))
        runtime.recent_news.appendleft(headline)

    def _news_bias_for_agent(self, runtime: SessionRuntime, personality: str) -> float:
        lag = {"quant_momentum": 1, "fundamental_value": 3, "retail_reactive": 5}.get(personality, 3)
        sentiment_sum = 0.0
        for event_tick, sentiment, _headline in runtime.news_events:
            if runtime.tick - event_tick >= lag:
                sentiment_sum += sentiment
        return sentiment_sum

    async def _policy_decision(self, agent: RuntimeAgent, runtime: SessionRuntime) -> Dict[str, Any]:
        config = agent.config
        prices = list(runtime.recent_prices)
        short_momentum = 0.0
        if len(prices) >= 6 and prices[-6] > 0:
            short_momentum = (prices[-1] - prices[-6]) / prices[-6]

        news_bias = self._news_bias_for_agent(runtime, config.personality)

        fair_value = float(np.mean(prices[-12:])) if len(prices) >= 12 else runtime.current_price
        deviation = (fair_value - runtime.current_price) / max(1e-6, runtime.current_price)

        side = "hold"
        confidence = 0.4
        rationale = "No strong signal."
        quantity = 0

        if config.personality == "quant_momentum":
            signal = (short_momentum * 6) + (news_bias * 0.2)
            if signal > 0.06:
                side = "buy"
                confidence = min(0.92, 0.5 + signal)
                rationale = "Momentum and lead-time news signal favor upside continuation."
            elif signal < -0.06:
                side = "sell"
                confidence = min(0.92, 0.5 + abs(signal))
                rationale = "Momentum reversal and early news skew suggest downside pressure."
        elif config.personality == "fundamental_value":
            signal = (deviation * 10) + (news_bias * 0.15)
            if signal > 0.08:
                side = "buy"
                confidence = min(0.9, 0.45 + signal)
                rationale = "Price below rolling fair value with manageable macro risk."
            elif signal < -0.08:
                side = "sell"
                confidence = min(0.9, 0.45 + abs(signal))
                rationale = "Price premium over fair value with weak catalyst quality."
        else:
            signal = (news_bias * 0.4) + (short_momentum * 2)
            if signal > 0.1:
                side = "buy"
                confidence = min(0.88, 0.4 + signal)
                rationale = "Retail crowd flow turned bullish after delayed catalyst diffusion."
            elif signal < -0.1:
                side = "sell"
                confidence = min(0.88, 0.4 + abs(signal))
                rationale = "Retail crowd flow turned bearish as negative narrative spread."

        if side != "hold":
            quantity = int(max(1, config.trade_size * (0.6 + config.aggressiveness)))

        # Periodically let the OpenRouter model refine the action.
        if runtime.tick % 6 == 0:
            llm = await generate_agent_decision(
                settings=self.settings,
                agent_name=config.name,
                model=config.model,
                personality=config.personality,
                market_state={
                    "ticker": runtime.ticker,
                    "price": runtime.current_price,
                    "tick": runtime.tick,
                    "volatility": runtime.volatility,
                    "short_momentum": short_momentum,
                    "news_bias": news_bias,
                    "crash_mode": runtime.crash_mode,
                },
                user_constraints={
                    "risk_limit": config.risk_limit,
                    "aggressiveness": config.aggressiveness,
                    "max_trade_size": config.trade_size * 4,
                },
            )

            llm_side = llm.get("side", "hold")
            llm_quantity = int(llm.get("quantity", 0) or 0)
            if llm_side in {"buy", "sell", "hold"}:
                side = llm_side
            if llm_quantity >= 0:
                quantity = min(max(0, llm_quantity), config.trade_size * 4)
            confidence = float(max(0, min(1, llm.get("confidence", confidence))))
            rationale = str(llm.get("rationale", rationale))

        if side != "hold":
            risk_scalar = max(0.1, 1 - (runtime.volatility * 8)) * config.risk_limit
            quantity = int(max(1, quantity * risk_scalar))

        return {
            "side": side,
            "quantity": quantity,
            "confidence": confidence,
            "rationale": rationale,
        }

    def _validate_trade(self, side: str, qty: int, portfolio: Portfolio, price: float) -> int:
        if qty <= 0:
            return 0
        if side == "buy":
            affordable = int(portfolio.cash // price)
            return max(0, min(qty, affordable))
        if side == "sell":
            return max(0, min(qty, portfolio.holdings))
        return 0

    def _execute_trade(self, runtime: SessionRuntime, agent: RuntimeAgent, side: str, qty: int, rationale: str) -> TradeRecord | None:
        valid_qty = self._validate_trade(side, qty, agent.portfolio, runtime.current_price)
        if valid_qty <= 0:
            return None

        direction = 1 if side == "buy" else -1
        spread_bps = 4 + (runtime.volatility * 450)
        impact_bps = (valid_qty / 1000) * (12 + runtime.volatility * 220)
        slippage_bps = spread_bps + impact_bps
        fill_price = runtime.current_price * (1 + direction * slippage_bps / 10_000)

        if side == "buy":
            agent.portfolio.buy(valid_qty, fill_price)
        else:
            agent.portfolio.sell(valid_qty, fill_price)

        # Price impact from execution.
        runtime.current_price *= 1 + direction * (impact_bps / 30_000)

        trade = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=runtime.session_id,
            ticker=runtime.ticker,
            side=side,
            agent=agent.config.name,
            quantity=valid_qty,
            price=round(fill_price, 4),
            slippage_bps=round(slippage_bps, 2),
            rationale=rationale,
        )
        runtime.trades.appendleft(trade)
        return trade

    async def _run_loop(self, runtime: SessionRuntime) -> None:
        while runtime.running and datetime.now(timezone.utc) < runtime.ends_at:
            runtime.tick += 1

            market_return = self._sample_market_return(runtime.volatility)
            runtime.current_price = max(0.5, runtime.current_price * (1 + market_return))

            if market_return < -0.035:
                runtime.crash_mode = True
                runtime.recent_news.appendleft("Crash regime detected: liquidity thinned and spreads widened.")
            elif runtime.crash_mode and self._rng.random() < 0.08:
                runtime.crash_mode = False

            self._maybe_news_event(runtime)

            trades_this_tick: List[TradeRecord] = []
            for agent in runtime.agents.values():
                decision = await self._policy_decision(agent, runtime)
                if decision["side"] == "hold":
                    continue
                trade = self._execute_trade(
                    runtime=runtime,
                    agent=agent,
                    side=decision["side"],
                    qty=decision["quantity"],
                    rationale=decision["rationale"],
                )
                if trade:
                    trades_this_tick.append(trade)

            runtime.recent_prices.append(runtime.current_price)

            tick_event = {
                "channel": "simulation",
                "type": "tick",
                "session_id": runtime.session_id,
                "ticker": runtime.ticker,
                "tick": runtime.tick,
                "price": round(runtime.current_price, 4),
                "crash_mode": runtime.crash_mode,
                "news": list(runtime.recent_news),
                "order_book": self._order_book(runtime.current_price, runtime.volatility),
                "portfolio_snapshot": {
                    name: agent.portfolio.mark_to_market(runtime.current_price)
                    for name, agent in runtime.agents.items()
                },
                "trades": [trade.model_dump() for trade in trades_this_tick],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.ws_manager.broadcast(tick_event, channel="simulation")
            await asyncio.sleep(1)

        runtime.running = False
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
            running=runtime.running,
            tick=runtime.tick,
            current_price=round(runtime.current_price, 4),
            volatility=runtime.volatility,
            started_at=runtime.started_at.isoformat(),
            ends_at=runtime.ends_at.isoformat(),
            crash_mode=runtime.crash_mode,
            recent_news=list(runtime.recent_news),
            trades=list(runtime.trades),
            portfolios={
                name: agent.portfolio.mark_to_market(runtime.current_price)
                for name, agent in runtime.agents.items()
            },
            order_book=self._order_book(runtime.current_price, runtime.volatility),
        )
