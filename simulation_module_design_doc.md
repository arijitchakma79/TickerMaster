# TickerMaster — Simulation Module: Technical Implementation Spec

> **Context**: This is a hackathon project. The Simulation module is one of three core features in **TickerMaster** (Research, Simulation, Tracker). TickerMaster is a sandbox world of financial AI agents that helps users learn trading and provides real-time sentiment/news watching for tickers. This document covers the **Simulation** module end-to-end. The shared backend is Python FastAPI and the frontend is React/TypeScript. The simulation engine runs inside **Modal Sandboxes** (this is critical for the Modal Sandbox prize). Every decision has been made — the coding agent should execute, not deliberate.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Modal Sandbox Integration](#3-modal-sandbox-integration)
4. [Simulation Engine](#4-simulation-engine)
5. [AI Agent System](#5-ai-agent-system)
6. [Market Dynamics & Realism](#6-market-dynamics--realism)
7. [Real-Time Communication (WebSockets)](#7-real-time-communication-websockets)
8. [AI Market Commentator](#8-ai-market-commentator)
9. [Frontend Implementation](#9-frontend-implementation)
10. [Data Models & Schemas](#10-data-models--schemas)
11. [API Endpoints & WebSocket Events](#11-api-endpoints--websocket-events)
12. [Integration with Research Module](#12-integration-with-research-module)
13. [Environment & Configuration](#13-environment--configuration)
14. [File Structure](#14-file-structure)
15. [Implementation Order](#15-implementation-order)

---

## 1. Project Overview

### What This Is
A real-time market simulation arena where LLM-powered AI agents with distinct trading personalities (quant momentum traders, fundamental value investors, retail YOLO traders) trade against each other in a sandboxed environment. Users can spin up agents with natural language, watch them trade in real-time, and learn how different strategies perform under various market conditions. The entire simulation engine runs inside **Modal Sandboxes** for safe, isolated code execution.

### Core Value Proposition
"A trading simulator powered by AI agents where you can see how your strategy would actually perform against the market." Users describe a trading strategy in plain English, an LLM-powered agent executes it, and they watch it compete against other AI traders in real-time — all in a sandboxed, safe environment.

### What Makes This Impressive (Prize Alignment)

| Prize | How We Qualify |
|-------|---------------|
| **Modal — Sandbox Challenge** | The entire simulation engine runs inside Modal Sandboxes. Each simulation is a separate sandbox with isolated execution, streaming stdout for real-time updates, and dynamic image building for custom agent dependencies. This is the textbook use case for Modal Sandboxes. |
| **Greylock — Best Multi-Turn Agent** | Agents engage in multi-step reasoning: they analyze the market → form a thesis → place orders → react to other agents' trades → adjust strategy → repeat. The user can intervene mid-simulation to change parameters, creating a true multi-turn interaction. |
| **OpenAI — AI Prize** | LLM-powered agents with distinct personalities and reasoning chains, visible "thinking" traces that show why each agent made a trade. |
| **Perplexity Sonar** | Agents can optionally pull real-time market context from Perplexity Sonar (via the Research module) to inform their trading decisions — e.g., a fundamental investor checks recent news before buying. |
| **Neo — Most Likely to Become a Product** | Educational trading simulators are a real market. This is a differentiated product: AI agents instead of static backtesting. |

### Key Demo Moment
User types: *"Create a momentum trader that buys when RSI < 30 and sells when RSI > 70, with $100K starting capital"* → Agent spawns in the arena → Immediately starts analyzing the order book → Places its first trade → User watches the P&L update in real-time alongside 3 other AI agents with different strategies → The AI commentator narrates: *"The Momentum Bot just went long on the dip while the Value Investor is staying cautious..."*

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      React Frontend                               │
│  ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌────────────────┐  │
│  │ Agent      │ │ Live     │ │ Order Book │ │ AI Commentator │  │
│  │ Setup/Chat │ │ Charts   │ │ & Trades   │ │ Panel          │  │
│  └────────────┘ └──────────┘ └────────────┘ └────────────────┘  │
└──────────────────────┬───────────────────────────────────────────┘
                       │ WebSocket (real-time) + REST (config)
┌──────────────────────▼───────────────────────────────────────────┐
│                    FastAPI Backend                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐   │
│  │ Simulation   │ │ WebSocket    │ │ Agent Config           │   │
│  │ Orchestrator │ │ Manager      │ │ Service (LLM prompts)  │   │
│  └──────┬───────┘ └──────────────┘ └────────────────────────┘   │
│         │                                                        │
│  ┌──────▼────────────────────────────────────────────────┐      │
│  │              Modal Sandbox Manager                     │      │
│  │  Creates, monitors, and streams from Modal Sandboxes   │      │
│  └──────┬────────────────────────────────────────────────┘      │
└─────────┼────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────────┐
│                    Modal Sandbox (per simulation)                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                 Simulation Engine                        │     │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────┐  │     │
│  │  │ Order Book │  │ Portfolio  │  │ Market Data      │  │     │
│  │  │ Engine     │  │ Manager    │  │ Generator        │  │     │
│  │  └────────────┘  └────────────┘  └──────────────────┘  │     │
│  │  ┌─────────────────────────────────────────────────────┐│     │
│  │  │              AI Agent Pool                          ││     │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ││     │
│  │  │  │ Quant   │ │ Value   │ │ Retail  │ │ Custom  │ ││     │
│  │  │  │ Trader  │ │Investor │ │ Trader  │ │ (User)  │ ││     │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ ││     │
│  │  └─────────────────────────────────────────────────────┘│     │
│  └─────────────────────────────────────────────────────────┘     │
│  stdout → streams JSON events back to FastAPI → WebSocket → UI   │
└──────────────────────────────────────────────────────────────────┘
```

### Tech Stack (Simulation-Specific)

| Layer | Technology | Notes |
|-------|-----------|-------|
| **Simulation Runtime** | Modal Sandbox | Isolated container per simulation, streaming stdout |
| **Simulation Engine** | Pure Python | Order book, matching engine, portfolio tracking |
| **AI Agent LLM** | OpenRouter API | OpenAI-compatible endpoint, access to GPT-4o, Claude, Llama |
| **Market Data** | S&P 500 Historical (yfinance) | Replay real market data for realism |
| **Real-Time Comms** | FastAPI WebSockets | Stream simulation events to frontend |
| **AI Commentator** | OpenAI GPT-4o-mini (via OpenRouter) | Narrates market events in real-time |
| **Backend Framework** | FastAPI (shared with Research) | Same server, new router |
| **Frontend** | React 18 + lightweight-charts + Recharts | Consistent with Research module |

---

## 3. Modal Sandbox Integration

### 3.1 Why Modal Sandboxes

Modal Sandboxes are the perfect fit for this simulation because:
1. **Isolation**: Each simulation runs in its own container — no risk of one sim crashing another
2. **Streaming stdout**: We can stream simulation events (trades, price updates, agent decisions) in real-time back to the backend via `sandbox.exec()` stdout iteration
3. **Dynamic images**: We can install custom Python packages per simulation if needed
4. **Cleanup**: Sandboxes auto-terminate, no zombie processes
5. **Prize eligibility**: This is a flagship use of Modal Sandboxes

### 3.2 Modal App Setup

```python
# simulation/modal_manager.py
import modal
import json
import asyncio
from typing import AsyncGenerator

# Create or look up the Modal app
app = modal.App.lookup("tickermaster-simulation", create_if_missing=True)

# Define the simulation image with all dependencies
sim_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy",
    "pandas",
    "httpx",
    "openai",  # OpenRouter uses OpenAI-compatible SDK
)
```

### 3.3 Creating a Simulation Sandbox

Each simulation gets its own Modal Sandbox. The simulation engine code is injected into the sandbox and executed. Events are streamed back via stdout as newline-delimited JSON.

```python
async def create_simulation_sandbox(
    config: dict,
    openrouter_api_key: str,
) -> tuple[modal.Sandbox, str]:
    """
    Create a Modal Sandbox for a simulation.
    
    Returns: (sandbox, sandbox_id)
    """
    sandbox = await modal.Sandbox.create.aio(
        app=app,
        image=sim_image,
        timeout=600,  # 10 minute max per simulation
        secrets=[
            modal.Secret.from_dict({
                "OPENROUTER_API_KEY": openrouter_api_key,
            })
        ],
    )
    
    sandbox_id = sandbox.object_id
    return sandbox, sandbox_id
```

### 3.4 Injecting and Running the Simulation Engine

The simulation engine is a self-contained Python script that gets exec'd inside the sandbox. It outputs JSON events to stdout which our backend reads in real-time.

```python
async def run_simulation_in_sandbox(
    sandbox: modal.Sandbox,
    config: dict,
) -> AsyncGenerator[dict, None]:
    """
    Execute the simulation engine inside the Modal Sandbox.
    Yields JSON events as they stream from stdout.
    """
    # Serialize the config as a JSON string to pass into the sandbox
    config_json = json.dumps(config)
    
    # The simulation engine code (see Section 4) is stored as a string
    # and executed inside the sandbox
    engine_code = get_simulation_engine_code()
    
    # Write the engine code to a file in the sandbox
    # Then execute it with the config as an argument
    process = await sandbox.exec.aio(
        "python", "-c", f"""
import sys
import json

# Simulation config passed as embedded JSON
CONFIG = json.loads('''{config_json}''')

# ---- BEGIN SIMULATION ENGINE ----
{engine_code}
# ---- END SIMULATION ENGINE ----

# Run the simulation
engine = SimulationEngine(CONFIG)
engine.run()
""",
        bufsize=1,  # Line-buffered for real-time streaming
    )
    
    # Stream events from stdout as they arrive
    async for line in process.stdout:
        line = line.strip()
        if line and line.startswith("{"):
            try:
                event = json.loads(line)
                yield event
            except json.JSONDecodeError:
                continue
    
    # Wait for process to complete
    await process.wait.aio()
```

### 3.5 Alternative: File-Based Engine Injection

For cleaner code organization, write the engine to a file inside the sandbox:

```python
async def run_simulation_v2(sandbox: modal.Sandbox, config: dict):
    """
    Write engine code to a file in the sandbox, then execute it.
    This is cleaner than embedding in a string.
    """
    # Write the engine code
    with sandbox.open("/app/engine.py", "w") as f:
        f.write(get_simulation_engine_code())
    
    # Write the config
    with sandbox.open("/app/config.json", "w") as f:
        f.write(json.dumps(config))
    
    # Execute
    process = await sandbox.exec.aio(
        "python", "/app/engine.py",
        workdir="/app",
        bufsize=1,
    )
    
    async for line in process.stdout:
        line = line.strip()
        if line and line.startswith("{"):
            yield json.loads(line)
```

---

## 4. Simulation Engine

The simulation engine is a self-contained Python module that runs INSIDE the Modal Sandbox. It must be fully self-contained — no imports from the main backend. Everything it needs must be defined inline or available in the sandbox image.

### 4.1 Core Engine Architecture

The engine runs a tick-based simulation loop:

```
For each tick (1 tick = 1 simulated minute of market time):
  1. Generate/replay market data for this tick
  2. Feed market data to each AI agent
  3. Agents decide on orders (LLM call or rule-based)
  4. Submit orders to the order book
  5. Match orders, execute trades
  6. Update all portfolios
  7. Emit events to stdout (JSON per line)
```

### 4.2 Simulation Engine Code

This is the code that runs inside the Modal Sandbox:

```python
# simulation_engine.py — runs INSIDE Modal Sandbox
# Must be fully self-contained

import json
import sys
import time
import random
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict
import numpy as np

# ============================================================
# UTILITY: Emit events to stdout as JSON (backend reads these)
# ============================================================

def emit_event(event_type: str, data: dict):
    """Print a JSON event to stdout. Backend reads this via streaming."""
    event = {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }
    print(json.dumps(event), flush=True)


# ============================================================
# ORDER BOOK
# ============================================================

@dataclass
class Order:
    id: str
    agent_id: str
    side: str          # "buy" or "sell"
    symbol: str
    price: float
    quantity: int
    order_type: str    # "limit" or "market"
    timestamp: str
    status: str = "open"  # open, filled, partially_filled, cancelled

class OrderBook:
    """
    A simple order book with price-time priority matching.
    Supports limit and market orders.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: list[Order] = []  # Buy orders, sorted highest price first
        self.asks: list[Order] = []  # Sell orders, sorted lowest price first
        self.trade_history: list[dict] = []
        self.last_price: float = 0.0
        self._order_counter = 0
    
    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD-{self._order_counter:06d}"
    
    def submit_order(self, agent_id: str, side: str, price: float, 
                     quantity: int, order_type: str = "limit") -> Order:
        order = Order(
            id=self._next_order_id(),
            agent_id=agent_id,
            side=side,
            symbol=self.symbol,
            price=price,
            quantity=quantity,
            order_type=order_type,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        trades = self._match_order(order)
        
        # If order not fully filled, add remainder to book
        if order.quantity > 0 and order.order_type == "limit":
            if order.side == "buy":
                self.bids.append(order)
                self.bids.sort(key=lambda o: (-o.price, o.timestamp))
            else:
                self.asks.append(order)
                self.asks.sort(key=lambda o: (o.price, o.timestamp))
        
        return order
    
    def _match_order(self, incoming: Order) -> list[dict]:
        trades = []
        
        if incoming.side == "buy":
            book_side = self.asks
            match_condition = lambda ask: (
                incoming.order_type == "market" or incoming.price >= ask.price
            )
        else:
            book_side = self.bids
            match_condition = lambda bid: (
                incoming.order_type == "market" or incoming.price <= bid.price
            )
        
        i = 0
        while i < len(book_side) and incoming.quantity > 0:
            resting = book_side[i]
            if not match_condition(resting):
                break
            
            fill_qty = min(incoming.quantity, resting.quantity)
            fill_price = resting.price  # Price-time priority: resting order's price
            
            trade = {
                "trade_id": f"TRD-{len(self.trade_history)+1:06d}",
                "symbol": self.symbol,
                "price": fill_price,
                "quantity": fill_qty,
                "buyer_id": incoming.agent_id if incoming.side == "buy" else resting.agent_id,
                "seller_id": resting.agent_id if incoming.side == "buy" else incoming.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "aggressor": incoming.agent_id,
            }
            trades.append(trade)
            self.trade_history.append(trade)
            self.last_price = fill_price
            
            incoming.quantity -= fill_qty
            resting.quantity -= fill_qty
            
            if resting.quantity == 0:
                resting.status = "filled"
                book_side.pop(i)
            else:
                resting.status = "partially_filled"
                i += 1
        
        if incoming.quantity == 0:
            incoming.status = "filled"
        elif trades:
            incoming.status = "partially_filled"
        
        # Emit trade events
        for trade in trades:
            emit_event("trade", trade)
        
        return trades
    
    def get_best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    def get_spread(self) -> Optional[float]:
        bid, ask = self.get_best_bid(), self.get_best_ask()
        if bid and ask:
            return ask - bid
        return None
    
    def get_depth(self, levels: int = 5) -> dict:
        """Return top N levels of the order book."""
        return {
            "bids": [
                {"price": o.price, "quantity": o.quantity, "agent_id": o.agent_id}
                for o in self.bids[:levels]
            ],
            "asks": [
                {"price": o.price, "quantity": o.quantity, "agent_id": o.agent_id}
                for o in self.asks[:levels]
            ],
            "last_price": self.last_price,
            "spread": self.get_spread(),
        }


# ============================================================
# PORTFOLIO MANAGER
# ============================================================

class Portfolio:
    """Tracks an agent's cash, positions, and P&L."""
    
    def __init__(self, agent_id: str, initial_cash: float = 100_000.0):
        self.agent_id = agent_id
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, int] = defaultdict(int)  # symbol → quantity
        self.trade_count = 0
        self.realized_pnl = 0.0
        self.cost_basis: dict[str, float] = defaultdict(float)  # symbol → avg cost
    
    def execute_trade(self, symbol: str, side: str, price: float, quantity: int):
        """Update portfolio after a trade execution."""
        self.trade_count += 1
        
        if side == "buy":
            total_cost = price * quantity
            if self.cash < total_cost:
                return False  # Insufficient funds
            self.cash -= total_cost
            
            # Update cost basis (weighted average)
            old_qty = self.positions[symbol]
            old_cost = self.cost_basis[symbol]
            new_qty = old_qty + quantity
            if new_qty > 0:
                self.cost_basis[symbol] = (old_cost * old_qty + price * quantity) / new_qty
            self.positions[symbol] = new_qty
            
        elif side == "sell":
            if self.positions[symbol] < quantity:
                return False  # Insufficient position
            
            # Realize P&L
            avg_cost = self.cost_basis[symbol]
            self.realized_pnl += (price - avg_cost) * quantity
            
            self.cash += price * quantity
            self.positions[symbol] -= quantity
            if self.positions[symbol] == 0:
                self.cost_basis[symbol] = 0.0
        
        return True
    
    def get_total_value(self, current_prices: dict[str, float]) -> float:
        """Total portfolio value = cash + mark-to-market positions."""
        position_value = sum(
            current_prices.get(sym, 0) * qty
            for sym, qty in self.positions.items()
        )
        return self.cash + position_value
    
    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        return sum(
            (current_prices.get(sym, 0) - self.cost_basis[sym]) * qty
            for sym, qty in self.positions.items()
            if qty > 0
        )
    
    def get_summary(self, current_prices: dict[str, float]) -> dict:
        total_value = self.get_total_value(current_prices)
        return {
            "agent_id": self.agent_id,
            "cash": round(self.cash, 2),
            "positions": dict(self.positions),
            "total_value": round(total_value, 2),
            "total_return_pct": round((total_value / self.initial_cash - 1) * 100, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.get_unrealized_pnl(current_prices), 2),
            "trade_count": self.trade_count,
        }


# ============================================================
# MARKET DATA GENERATOR
# ============================================================

class MarketDataGenerator:
    """
    Generates realistic market data using geometric Brownian motion (GBM)
    seeded with real S&P 500 parameters. Can also replay historical data.
    """
    
    def __init__(self, config: dict):
        self.symbol = config.get("symbol", "SIM")
        self.initial_price = config.get("initial_price", 450.0)  # ~S&P 500 / 10
        self.volatility = config.get("volatility", 0.02)  # Daily vol
        self.drift = config.get("drift", 0.0001)  # Slight upward drift
        self.current_price = self.initial_price
        self.tick = 0
        self.price_history = [self.initial_price]
        
        # Market regime parameters
        self.crash_probability = config.get("crash_probability", 0.005)  # 0.5% per tick
        self.rally_probability = config.get("rally_probability", 0.005)
        self.regime = "normal"  # normal, crash, rally
        self.regime_ticks_remaining = 0
        
        # Technical indicator state
        self.returns_history = []
    
    def generate_tick(self) -> dict:
        """Generate the next price tick using GBM with regime switching."""
        self.tick += 1
        
        # Check for regime changes
        if self.regime == "normal":
            if random.random() < self.crash_probability:
                self.regime = "crash"
                self.regime_ticks_remaining = random.randint(3, 15)
            elif random.random() < self.rally_probability:
                self.regime = "rally"
                self.regime_ticks_remaining = random.randint(3, 10)
        
        # Adjust parameters based on regime
        if self.regime == "crash":
            vol = self.volatility * 3.0
            drift = -0.005
            self.regime_ticks_remaining -= 1
            if self.regime_ticks_remaining <= 0:
                self.regime = "normal"
        elif self.regime == "rally":
            vol = self.volatility * 1.5
            drift = 0.003
            self.regime_ticks_remaining -= 1
            if self.regime_ticks_remaining <= 0:
                self.regime = "normal"
        else:
            vol = self.volatility
            drift = self.drift
        
        # Geometric Brownian Motion
        random_return = np.random.normal(drift, vol)
        self.current_price *= (1 + random_return)
        self.current_price = max(self.current_price, 1.0)  # Floor at $1
        
        self.price_history.append(self.current_price)
        self.returns_history.append(random_return)
        
        # Compute technical indicators
        technicals = self._compute_technicals()
        
        tick_data = {
            "tick": self.tick,
            "symbol": self.symbol,
            "price": round(self.current_price, 2),
            "open": round(self.price_history[-2] if len(self.price_history) > 1 else self.current_price, 2),
            "high": round(max(self.price_history[-1], self.price_history[-2]) if len(self.price_history) > 1 else self.current_price, 2),
            "low": round(min(self.price_history[-1], self.price_history[-2]) if len(self.price_history) > 1 else self.current_price, 2),
            "close": round(self.current_price, 2),
            "volume": random.randint(100_000, 500_000),
            "regime": self.regime,
            "return_pct": round(random_return * 100, 4),
            **technicals,
        }
        
        return tick_data
    
    def _compute_technicals(self) -> dict:
        """Compute RSI, SMA, and volatility for agents to use."""
        prices = self.price_history
        
        # RSI (14-period)
        rsi = 50.0  # Default
        if len(self.returns_history) >= 14:
            recent = self.returns_history[-14:]
            gains = [r for r in recent if r > 0]
            losses = [-r for r in recent if r < 0]
            avg_gain = sum(gains) / 14 if gains else 0.001
            avg_loss = sum(losses) / 14 if losses else 0.001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Simple Moving Averages
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
        
        # Volatility (20-period)
        vol_20 = np.std(self.returns_history[-20:]) if len(self.returns_history) >= 20 else self.volatility
        
        return {
            "rsi": round(rsi, 2),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "volatility_20": round(vol_20, 6),
            "price_vs_sma20": round((self.current_price / sma_20 - 1) * 100, 2) if sma_20 > 0 else 0,
        }


# ============================================================
# AI AGENTS
# ============================================================

class BaseAgent:
    """Base class for all trading agents."""
    
    def __init__(self, agent_id: str, name: str, personality: str, 
                 initial_cash: float = 100_000.0, config: dict = None):
        self.agent_id = agent_id
        self.name = name
        self.personality = personality
        self.portfolio = Portfolio(agent_id, initial_cash)
        self.config = config or {}
        self.decision_history: list[dict] = []
        self.ticks_since_last_trade = 0
    
    def decide(self, market_data: dict, order_book_depth: dict) -> Optional[dict]:
        """
        Given current market data and order book, decide what to do.
        Returns an order dict or None.
        """
        raise NotImplementedError
    
    def _create_order(self, side: str, price: float, quantity: int, 
                      reasoning: str) -> dict:
        decision = {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "side": side,
            "price": round(price, 2),
            "quantity": quantity,
            "reasoning": reasoning,
        }
        self.decision_history.append(decision)
        return decision


class QuantMomentumAgent(BaseAgent):
    """
    Quantitative momentum trader.
    - Reacts FAST (every tick)
    - Uses technical indicators (RSI, SMA crossovers)
    - High frequency, small positions
    """
    
    def __init__(self, agent_id: str, initial_cash: float = 100_000.0):
        super().__init__(
            agent_id=agent_id,
            name="Quant Momentum Bot",
            personality="quant",
            initial_cash=initial_cash,
        )
        self.position_size_pct = 0.05  # 5% of portfolio per trade
    
    def decide(self, market_data: dict, order_book_depth: dict) -> Optional[dict]:
        self.ticks_since_last_trade += 1
        price = market_data["price"]
        rsi = market_data.get("rsi", 50)
        sma_20 = market_data.get("sma_20", price)
        sma_50 = market_data.get("sma_50", price)
        symbol = market_data["symbol"]
        
        position = self.portfolio.positions.get(symbol, 0)
        max_qty = int((self.portfolio.cash * self.position_size_pct) / price) if price > 0 else 0
        
        # Strategy: RSI mean reversion + SMA trend following
        if rsi < 30 and price > sma_50 and max_qty > 0:
            return self._create_order(
                "buy", price * 1.001,  # Slightly above market
                max(1, max_qty),
                f"RSI oversold ({rsi:.1f}), price above SMA50 — buying the dip"
            )
        elif rsi > 70 and position > 0:
            sell_qty = max(1, position // 2)
            return self._create_order(
                "sell", price * 0.999,
                sell_qty,
                f"RSI overbought ({rsi:.1f}) — taking profits on {sell_qty} shares"
            )
        elif sma_20 > sma_50 and price > sma_20 and max_qty > 0 and self.ticks_since_last_trade > 5:
            return self._create_order(
                "buy", price * 1.001,
                max(1, max_qty // 2),
                f"Bullish SMA crossover (SMA20 > SMA50) — riding momentum"
            )
        elif sma_20 < sma_50 and position > 0:
            return self._create_order(
                "sell", price * 0.999,
                position,
                f"Bearish SMA crossover — exiting all positions"
            )
        
        return None


class FundamentalValueAgent(BaseAgent):
    """
    Fundamental value investor.
    - Reacts SLOWLY (every 5-10 ticks)
    - Buys when price is below "fair value" (estimated from moving avg)
    - Large positions, long holding periods
    - Sometimes consults news (Perplexity via Research module)
    """
    
    def __init__(self, agent_id: str, initial_cash: float = 100_000.0):
        super().__init__(
            agent_id=agent_id,
            name="Value Investor",
            personality="fundamental",
            initial_cash=initial_cash,
        )
        self.fair_value_estimate = None
        self.conviction = 0.0  # -1 to 1
        self.min_ticks_between_trades = 8
    
    def decide(self, market_data: dict, order_book_depth: dict) -> Optional[dict]:
        self.ticks_since_last_trade += 1
        
        if self.ticks_since_last_trade < self.min_ticks_between_trades:
            return None
        
        price = market_data["price"]
        sma_50 = market_data.get("sma_50", price)
        symbol = market_data["symbol"]
        
        # Estimate fair value as SMA50 (simplified fundamental analysis)
        self.fair_value_estimate = sma_50
        discount = (self.fair_value_estimate - price) / self.fair_value_estimate
        
        position = self.portfolio.positions.get(symbol, 0)
        max_qty = int((self.portfolio.cash * 0.15) / price) if price > 0 else 0  # 15% position size
        
        if discount > 0.03 and max_qty > 0:  # Price 3%+ below fair value
            self.ticks_since_last_trade = 0
            return self._create_order(
                "buy", price * 1.002,
                max(1, max_qty),
                f"Trading at {discount*100:.1f}% below fair value (${self.fair_value_estimate:.2f}) — accumulating"
            )
        elif discount < -0.05 and position > 0:  # Price 5%+ above fair value
            self.ticks_since_last_trade = 0
            return self._create_order(
                "sell", price * 0.998,
                position,
                f"Trading {abs(discount)*100:.1f}% above fair value — fully exiting, overvalued"
            )
        
        return None


class RetailTraderAgent(BaseAgent):
    """
    Retail/YOLO trader.
    - Reacts with DELAY (2-5 ticks after events)
    - FOMO-driven: buys after rallies, panic sells in crashes
    - Random position sizing, emotional reasoning
    - Represents the "dumb money" for educational contrast
    """
    
    def __init__(self, agent_id: str, initial_cash: float = 100_000.0):
        super().__init__(
            agent_id=agent_id,
            name="Retail YOLO Trader",
            personality="retail",
            initial_cash=initial_cash,
        )
        self.fomo_threshold = 0.02  # 2% move triggers FOMO
        self.panic_threshold = -0.03  # 3% drop triggers panic
        self.recent_returns: list[float] = []
    
    def decide(self, market_data: dict, order_book_depth: dict) -> Optional[dict]:
        self.ticks_since_last_trade += 1
        price = market_data["price"]
        ret = market_data.get("return_pct", 0) / 100
        symbol = market_data["symbol"]
        regime = market_data.get("regime", "normal")
        
        self.recent_returns.append(ret)
        if len(self.recent_returns) > 5:
            self.recent_returns.pop(0)
        
        # Delayed reaction: only act every 3+ ticks
        if self.ticks_since_last_trade < 3:
            return None
        
        cumulative_return = sum(self.recent_returns)
        position = self.portfolio.positions.get(symbol, 0)
        
        # FOMO buying: "it's going up, I need to get in!"
        if cumulative_return > self.fomo_threshold and position == 0:
            yolo_pct = random.uniform(0.1, 0.3)  # 10-30% of portfolio
            qty = max(1, int((self.portfolio.cash * yolo_pct) / price))
            self.ticks_since_last_trade = 0
            return self._create_order(
                "buy", price * 1.005,  # Market buy, willing to pay up
                qty,
                f"🚀 Stock is MOONING (+{cumulative_return*100:.1f}% in 5 ticks)! FOMO buying!"
            )
        
        # Panic selling: "it's crashing, get me out!"
        elif (cumulative_return < self.panic_threshold or regime == "crash") and position > 0:
            self.ticks_since_last_trade = 0
            return self._create_order(
                "sell", price * 0.995,  # Desperate to sell
                position,
                f"😱 PANIC SELL! Down {abs(cumulative_return)*100:.1f}%! Cutting losses on all {position} shares!"
            )
        
        # Random "I saw it on Reddit" trade
        elif random.random() < 0.03 and position == 0:  # 3% chance per tick
            qty = max(1, int((self.portfolio.cash * 0.05) / price))
            self.ticks_since_last_trade = 0
            return self._create_order(
                "buy", price * 1.003,
                qty,
                f"💎🙌 Someone on Reddit said this is going to the moon. YOLO!"
            )
        
        return None


class LLMAgent(BaseAgent):
    """
    Custom LLM-powered agent. Uses OpenRouter to make trading decisions.
    The user defines the strategy in natural language.
    This agent calls the LLM every N ticks to decide on trades.
    """
    
    def __init__(self, agent_id: str, name: str, strategy_prompt: str,
                 initial_cash: float = 100_000.0, think_interval: int = 5):
        super().__init__(
            agent_id=agent_id,
            name=name,
            personality="custom_llm",
            initial_cash=initial_cash,
        )
        self.strategy_prompt = strategy_prompt
        self.think_interval = think_interval  # LLM call every N ticks
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
    
    def decide(self, market_data: dict, order_book_depth: dict) -> Optional[dict]:
        self.ticks_since_last_trade += 1
        
        # Only call LLM every N ticks to manage costs / rate limits
        if market_data["tick"] % self.think_interval != 0:
            return None
        
        # Build context for the LLM
        symbol = market_data["symbol"]
        position = self.portfolio.positions.get(symbol, 0)
        portfolio_summary = self.portfolio.get_summary({symbol: market_data["price"]})
        
        prompt = f"""You are a trading agent in a simulated market. Your strategy is:
{self.strategy_prompt}

CURRENT MARKET STATE:
- Symbol: {symbol}
- Price: ${market_data['price']:.2f}
- RSI: {market_data.get('rsi', 'N/A')}
- SMA 20: ${market_data.get('sma_20', 'N/A')}
- SMA 50: ${market_data.get('sma_50', 'N/A')}
- Volatility: {market_data.get('volatility_20', 'N/A')}
- Market Regime: {market_data.get('regime', 'normal')}
- Last Return: {market_data.get('return_pct', 0):.2f}%
- Tick: {market_data['tick']}

ORDER BOOK:
- Best Bid: {order_book_depth.get('bids', [{}])[0].get('price', 'N/A') if order_book_depth.get('bids') else 'Empty'}
- Best Ask: {order_book_depth.get('asks', [{}])[0].get('price', 'N/A') if order_book_depth.get('asks') else 'Empty'}
- Spread: {order_book_depth.get('spread', 'N/A')}

YOUR PORTFOLIO:
- Cash: ${portfolio_summary['cash']:.2f}
- Position in {symbol}: {position} shares
- Total Value: ${portfolio_summary['total_value']:.2f}
- Return: {portfolio_summary['total_return_pct']:.2f}%

RECENT DECISIONS: {json.dumps(self.decision_history[-3:]) if self.decision_history else 'None yet'}

Based on your strategy and the current market state, what should you do?
Respond with EXACTLY one of these JSON formats (no other text):
{{"action": "buy", "quantity": <int>, "price": <float>, "reasoning": "<1 sentence>"}}
{{"action": "sell", "quantity": <int>, "price": <float>, "reasoning": "<1 sentence>"}}
{{"action": "hold", "reasoning": "<1 sentence>"}}
"""
        
        try:
            import httpx
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",  # Cheap + fast
                    "messages": [
                        {"role": "system", "content": "You are a precise trading bot. Only output valid JSON. No markdown, no explanation."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200,
                },
                timeout=10.0,
            )
            
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Clean up response (remove markdown backticks if present)
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
            
            decision = json.loads(content)
            
            # Emit the agent's thinking
            emit_event("agent_thinking", {
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "reasoning": decision.get("reasoning", ""),
                "action": decision.get("action", "hold"),
            })
            
            if decision["action"] == "buy":
                qty = min(decision["quantity"], 
                          int(self.portfolio.cash / market_data["price"]))
                if qty > 0:
                    self.ticks_since_last_trade = 0
                    return self._create_order(
                        "buy", decision["price"], qty, decision["reasoning"]
                    )
            elif decision["action"] == "sell":
                qty = min(decision["quantity"], position)
                if qty > 0:
                    self.ticks_since_last_trade = 0
                    return self._create_order(
                        "sell", decision["price"], qty, decision["reasoning"]
                    )
        
        except Exception as e:
            emit_event("agent_error", {
                "agent_id": self.agent_id,
                "error": str(e),
            })
        
        return None


# ============================================================
# SIMULATION ENGINE (MAIN LOOP)
# ============================================================

class SimulationEngine:
    """
    Main simulation orchestrator. Runs the tick loop, coordinates agents,
    manages the order book, and emits events.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.symbol = config.get("symbol", "SIM")
        self.total_ticks = config.get("total_ticks", 200)
        self.tick_delay = config.get("tick_delay_seconds", 0.3)  # Time between ticks
        
        # Initialize components
        self.market_data_gen = MarketDataGenerator({
            "symbol": self.symbol,
            "initial_price": config.get("initial_price", 450.0),
            "volatility": config.get("volatility", 0.02),
            "crash_probability": config.get("crash_probability", 0.005),
        })
        self.order_book = OrderBook(self.symbol)
        self.agents: list[BaseAgent] = []
        
        # Create agents based on config
        self._create_agents(config.get("agents", []))
        
        # Seed the order book with initial liquidity
        self._seed_order_book(config.get("initial_price", 450.0))
    
    def _create_agents(self, agent_configs: list[dict]):
        """Create agents from config. Always include the 3 preset agents."""
        initial_cash = self.config.get("initial_cash_per_agent", 100_000.0)
        
        # Always create the 3 archetype agents
        self.agents.append(QuantMomentumAgent("agent-quant", initial_cash))
        self.agents.append(FundamentalValueAgent("agent-value", initial_cash))
        self.agents.append(RetailTraderAgent("agent-retail", initial_cash))
        
        # Create user-defined LLM agents
        for i, ac in enumerate(agent_configs):
            agent = LLMAgent(
                agent_id=f"agent-custom-{i}",
                name=ac.get("name", f"Custom Agent {i+1}"),
                strategy_prompt=ac.get("strategy", "Buy low, sell high."),
                initial_cash=initial_cash,
                think_interval=ac.get("think_interval", 5),
            )
            self.agents.append(agent)
        
        emit_event("agents_created", {
            "agents": [
                {
                    "id": a.agent_id,
                    "name": a.name,
                    "personality": a.personality,
                    "initial_cash": a.portfolio.initial_cash,
                }
                for a in self.agents
            ]
        })
    
    def _seed_order_book(self, price: float):
        """Place initial orders to provide liquidity."""
        # Create a market maker agent (not tracked in leaderboard)
        for i in range(10):
            spread = 0.01 * (i + 1)
            self.order_book.submit_order(
                "market-maker", "buy", round(price * (1 - spread), 2),
                random.randint(50, 200), "limit"
            )
            self.order_book.submit_order(
                "market-maker", "sell", round(price * (1 + spread), 2),
                random.randint(50, 200), "limit"
            )
    
    def run(self):
        """Main simulation loop."""
        emit_event("simulation_started", {
            "symbol": self.symbol,
            "total_ticks": self.total_ticks,
            "num_agents": len(self.agents),
            "initial_price": self.market_data_gen.initial_price,
        })
        
        for tick in range(1, self.total_ticks + 1):
            # 1. Generate market data
            market_data = self.market_data_gen.generate_tick()
            
            emit_event("market_tick", market_data)
            
            # 2. Replenish market maker liquidity
            if tick % 10 == 0:
                self._seed_order_book(market_data["price"])
            
            # 3. Each agent makes a decision
            # Simulate information asymmetry: quants react first, retail last
            agent_order = sorted(self.agents, key=lambda a: {
                "quant": 0, "fundamental": 1, "custom_llm": 2, "retail": 3
            }.get(a.personality, 2))
            
            for agent in agent_order:
                order_depth = self.order_book.get_depth()
                decision = agent.decide(market_data, order_depth)
                
                if decision:
                    # Validate the trade
                    if decision["side"] == "buy":
                        cost = decision["price"] * decision["quantity"]
                        if agent.portfolio.cash < cost:
                            continue  # Can't afford it
                    elif decision["side"] == "sell":
                        pos = agent.portfolio.positions.get(self.symbol, 0)
                        if pos < decision["quantity"]:
                            decision["quantity"] = pos  # Sell what we have
                            if decision["quantity"] <= 0:
                                continue
                    
                    # Submit to order book
                    order = self.order_book.submit_order(
                        agent.agent_id,
                        decision["side"],
                        decision["price"],
                        decision["quantity"],
                        "limit",
                    )
                    
                    # Emit decision event
                    emit_event("agent_decision", {
                        **decision,
                        "order_id": order.id,
                        "order_status": order.status,
                    })
            
            # 4. Process any fills and update portfolios
            for trade in self.order_book.trade_history:
                # Find buyer and seller agents
                for agent in self.agents:
                    if agent.agent_id == trade.get("buyer_id"):
                        agent.portfolio.execute_trade(
                            self.symbol, "buy", trade["price"], trade["quantity"]
                        )
                        agent.ticks_since_last_trade = 0
                    elif agent.agent_id == trade.get("seller_id"):
                        agent.portfolio.execute_trade(
                            self.symbol, "sell", trade["price"], trade["quantity"]
                        )
                        agent.ticks_since_last_trade = 0
            
            # Clear processed trades (already emitted in order book)
            self.order_book.trade_history.clear()
            
            # 5. Emit portfolio snapshots every 5 ticks
            if tick % 5 == 0:
                current_prices = {self.symbol: market_data["price"]}
                leaderboard = sorted(
                    [a.portfolio.get_summary(current_prices) for a in self.agents],
                    key=lambda p: p["total_value"],
                    reverse=True,
                )
                emit_event("leaderboard_update", {
                    "tick": tick,
                    "leaderboard": leaderboard,
                    "order_book_depth": self.order_book.get_depth(),
                })
            
            # 6. Delay between ticks
            time.sleep(self.tick_delay)
        
        # Final summary
        current_prices = {self.symbol: self.market_data_gen.current_price}
        final_results = sorted(
            [a.portfolio.get_summary(current_prices) for a in self.agents],
            key=lambda p: p["total_value"],
            reverse=True,
        )
        
        emit_event("simulation_complete", {
            "final_results": final_results,
            "final_price": round(self.market_data_gen.current_price, 2),
            "total_ticks": self.total_ticks,
            "price_history": [round(p, 2) for p in self.market_data_gen.price_history],
        })
```

---

## 5. AI Agent System

### 5.1 Agent Archetypes (Pre-built)

These 3 agents are ALWAYS present in every simulation. They represent the real market:

| Agent | Reaction Speed | Strategy | Personality | Educational Purpose |
|-------|---------------|----------|-------------|-------------------|
| **Quant Momentum Bot** | Instant (every tick) | RSI mean reversion + SMA crossovers | Cold, calculated, data-driven | Shows how quants exploit technical signals |
| **Value Investor** | Slow (every 8+ ticks) | Buy below fair value, sell above | Patient, contrarian, thesis-driven | Shows long-term value approach |
| **Retail YOLO Trader** | Delayed (3+ ticks) | FOMO buying, panic selling, Reddit hype | Emotional, reactive, meme-driven | Shows why retail loses money to institutions |

### 5.2 Information Asymmetry (Key Educational Feature)

Agents process information at different speeds, simulating real-world information asymmetry:

```
Tick N: Market event occurs (e.g., crash begins)
  → Tick N+0: Quant sees it immediately, starts selling
  → Tick N+2: Value Investor evaluates fundamentals
  → Tick N+3: Retail Trader notices the price drop
  → Tick N+5: Retail panics and sells at the bottom
  → Tick N+8: Quant is already buying the dip
```

This is enforced by the `ticks_since_last_trade` cooldown and the sorted agent execution order in the main loop.

### 5.3 Custom LLM Agents (User-Created)

Users describe a strategy in natural language. The backend wraps it into a structured prompt and creates an `LLMAgent`. The LLM is called every N ticks (default: 5) to decide on trades.

**User input flow**:
1. User types: "Create an agent that buys when there's a crash and sells after 10% gains"
2. Backend sends this to the simulation config as `strategy_prompt`
3. Inside the sandbox, `LLMAgent` uses this prompt + current market state to make decisions
4. LLM responds with JSON: `{"action": "buy", "quantity": 10, "price": 445.50, "reasoning": "Market crash detected, buying the dip per strategy"}`
5. This decision is submitted to the order book like any other agent

**OpenRouter API call** (inside the sandbox):
- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Model: `openai/gpt-4o-mini` (cheap, fast, good enough for trading decisions)
- Auth: `Authorization: Bearer {OPENROUTER_API_KEY}`
- Temperature: 0.3 (mostly deterministic)
- Max tokens: 200 (just need a JSON response)
- The OpenAI Python SDK works with OpenRouter by setting `base_url="https://openrouter.ai/api/v1"`

---

## 6. Market Dynamics & Realism

### 6.1 Price Generation: Geometric Brownian Motion

Prices are generated using GBM with regime switching:
- **Normal regime**: drift=0.01%, vol=2% per tick
- **Crash regime**: drift=-0.5%, vol=6% per tick, lasts 3-15 ticks
- **Rally regime**: drift=+0.3%, vol=3% per tick, lasts 3-10 ticks
- Regime transitions are stochastic (0.5% probability per tick)

### 6.2 Price Impact

When agents place large orders, the fill price includes slippage:
- Market orders fill at progressively worse prices through the order book
- This prevents agents from buying/selling unlimited quantities at one price
- Realistic: a $50K market buy will move the price slightly

### 6.3 Order Book Mechanics

The order book uses strict price-time priority:
- Limit orders wait in the book until a matching counter-order arrives
- Market orders execute immediately against the best available price
- A "market maker" bot replenishes liquidity every 10 ticks

### 6.4 Execution Slippage

```python
# Slippage model: larger orders get worse fills
def apply_slippage(price: float, quantity: int, side: str, volatility: float) -> float:
    impact = volatility * (quantity / 1000) * 0.1  # Proportional to size
    if side == "buy":
        return price * (1 + impact)
    else:
        return price * (1 - impact)
```

---

## 7. Real-Time Communication (WebSockets)

### 7.1 WebSocket Architecture

The simulation streams events from Modal Sandbox → FastAPI backend → WebSocket → React frontend. This is the real-time backbone of the simulation.

```python
# backend/routers/simulation.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
import json

router = APIRouter(prefix="/api/simulation", tags=["simulation"])

# Store active simulations
active_simulations: dict[str, dict] = {}  # sim_id → {sandbox, clients, config}


@router.post("/create")
async def create_simulation(config: dict):
    """
    Create a new simulation. Returns a simulation_id.
    The user then connects via WebSocket to receive real-time events.
    """
    sim_id = f"sim-{int(time.time())}-{random.randint(1000,9999)}"
    
    active_simulations[sim_id] = {
        "config": config,
        "clients": [],
        "status": "created",
    }
    
    return {"simulation_id": sim_id, "status": "created"}


@router.websocket("/ws/{sim_id}")
async def simulation_websocket(websocket: WebSocket, sim_id: str):
    """
    WebSocket endpoint for a simulation.
    1. Client connects
    2. If simulation hasn't started, start it
    3. Stream events from Modal Sandbox to client
    """
    await websocket.accept()
    
    if sim_id not in active_simulations:
        await websocket.send_json({"type": "error", "data": {"message": "Simulation not found"}})
        await websocket.close()
        return
    
    sim = active_simulations[sim_id]
    sim["clients"].append(websocket)
    
    try:
        # Start the simulation if it hasn't started yet
        if sim["status"] == "created":
            sim["status"] = "running"
            
            # Create Modal Sandbox
            sandbox, sandbox_id = await create_simulation_sandbox(
                sim["config"],
                os.environ.get("OPENROUTER_API_KEY", ""),
            )
            sim["sandbox"] = sandbox
            sim["sandbox_id"] = sandbox_id
            
            # Run simulation and stream events
            async for event in run_simulation_in_sandbox(sandbox, sim["config"]):
                # Broadcast to all connected clients
                for client in sim["clients"]:
                    try:
                        await client.send_json(event)
                    except Exception:
                        sim["clients"].remove(client)
                
                # Also check for incoming messages (user interventions)
                # Non-blocking check
                try:
                    msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                    if msg.get("type") == "user_intervention":
                        # Handle mid-simulation user actions
                        pass
                except asyncio.TimeoutError:
                    pass
            
            sim["status"] = "complete"
            
            # Cleanup
            await sandbox.terminate.aio()
        
        elif sim["status"] == "running":
            # Late joiner — just add them to the client list
            # They'll receive events from the broadcast loop
            while sim["status"] == "running":
                await asyncio.sleep(0.5)
    
    except WebSocketDisconnect:
        if websocket in sim["clients"]:
            sim["clients"].remove(websocket)
    except Exception as e:
        await websocket.send_json({"type": "error", "data": {"message": str(e)}})
```

### 7.2 Event Types (Sandbox → Frontend)

All events are JSON objects with `type` and `data` fields:

| Event Type | Frequency | Data |
|-----------|-----------|------|
| `simulation_started` | Once | symbol, total_ticks, num_agents, initial_price |
| `agents_created` | Once | Array of agent info (id, name, personality, cash) |
| `market_tick` | Every tick | tick, price, OHLCV, regime, RSI, SMA, volatility |
| `trade` | Per trade | trade_id, price, qty, buyer_id, seller_id |
| `agent_decision` | Per decision | agent_id, agent_name, side, price, qty, reasoning |
| `agent_thinking` | Per LLM call | agent_id, reasoning, action (for custom agents) |
| `leaderboard_update` | Every 5 ticks | Sorted portfolio summaries for all agents |
| `simulation_complete` | Once | final_results, price_history, total_ticks |
| `agent_error` | On error | agent_id, error message |

---

## 8. AI Market Commentator

### 8.1 Overview

A separate LLM call generates real-time market commentary, narrating the simulation like a sports commentator. This is the "fun" factor that makes the demo memorable.

### 8.2 Implementation

The commentator runs on the **backend** (not inside the sandbox) and consumes the same event stream. It generates commentary every 10 ticks or on significant events.

```python
# backend/services/commentator.py
import httpx

async def generate_commentary(
    recent_events: list[dict],
    market_state: dict,
    leaderboard: list[dict],
) -> str:
    """
    Generate AI market commentary based on recent events.
    Call this every 10 ticks or on significant events.
    """
    events_summary = "\n".join([
        f"- [{e['type']}] {json.dumps(e['data'])[:200]}"
        for e in recent_events[-10:]
    ])
    
    leader = leaderboard[0] if leaderboard else {}
    laggard = leaderboard[-1] if leaderboard else {}
    
    prompt = f"""You are a witty, engaging financial market commentator narrating a live trading simulation.
Keep it SHORT (2-3 sentences max), entertaining, and educational.
Use trader slang and emoji sparingly. Sound like a mix of CNBC anchor and Twitch streamer.

CURRENT MARKET:
- Price: ${market_state.get('price', 'N/A')} | Regime: {market_state.get('regime', 'normal')}
- RSI: {market_state.get('rsi', 'N/A')} | Tick: {market_state.get('tick', 0)}/{market_state.get('total_ticks', 200)}

LEADERBOARD:
- #1: {leader.get('agent_id', 'N/A')} at ${leader.get('total_value', 0):.0f} ({leader.get('total_return_pct', 0):.1f}%)
- Last: {laggard.get('agent_id', 'N/A')} at ${laggard.get('total_value', 0):.0f} ({laggard.get('total_return_pct', 0):.1f}%)

RECENT EVENTS:
{events_summary}

Generate a brief, entertaining commentary on what just happened:"""
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a charismatic market commentator. Be brief, funny, and insightful."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.8,
                "max_tokens": 150,
            },
            timeout=8.0,
        )
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
```

The commentary is emitted as a special WebSocket event:
```json
{"type": "commentary", "data": {"text": "The Quant Bot just sniped 50 shares at $447.20 while Retail YOLO was still refreshing Reddit 😂 Classic information asymmetry in action!", "tick": 45}}
```

---

## 9. Frontend Implementation

### 9.1 Simulation Page Layout (`/simulation`)

```
┌──────────────────────────────────────────────────────────────────┐
│ HEADER: TickerMaster Simulation Arena  |  SIM Symbol  |  Status │
├──────────────────────────────────────────────────────────────────┤
│ ┌───────────────────────────────────┐ ┌────────────────────────┐ │
│ │                                   │ │ AGENT SETUP PANEL      │ │
│ │   LIVE PRICE CHART                │ │                        │ │
│ │   (Candlestick + Volume)          │ │ [Natural language      │ │
│ │   Updates in real-time            │ │  input to create       │ │
│ │   Shows regime (crash/rally bg)   │ │  custom agent]         │ │
│ │                                   │ │                        │ │
│ │   Regime indicator:               │ │ Agent Cards:           │ │
│ │   🟢 Normal  🔴 Crash  🟡 Rally  │ │ - Quant Bot ✅         │ │
│ │                                   │ │ - Value Investor ✅    │ │
│ │                                   │ │ - Retail YOLO ✅       │ │
│ │                                   │ │ - [Your Agent] ✏️      │ │
│ └───────────────────────────────────┘ └────────────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────────┐│
│ │ LEADERBOARD     │ │ LIVE ORDER BOOK │ │ AI COMMENTATOR       ││
│ │                 │ │                 │ │                      ││
│ │ 1. Quant +3.2%  │ │ BIDS    ASKS    │ │ "The Quant Bot just  ││
│ │ 2. Value +1.1%  │ │ $447 ██ $448 ██ │ │  sniped 50 shares    ││
│ │ 3. Custom +0.5% │ │ $446 ██ $449 ██ │ │  while Retail was    ││
│ │ 4. Retail -2.8% │ │ $445 █  $450 █  │ │  still refreshing    ││
│ │                 │ │                 │ │  Reddit 😂"           ││
│ │ [P&L chart]     │ │ Last: $447.50   │ │                      ││
│ └─────────────────┘ └─────────────────┘ └──────────────────────┘│
├──────────────────────────────────────────────────────────────────┤
│ TRADE LOG (scrollable, auto-updating)                            │
│ 10:05:23  Quant Bot BOUGHT 50 @ $447.20 — "RSI oversold, dip"  │
│ 10:05:21  Retail SOLD 100 @ $446.90 — "😱 PANIC SELL!"         │
│ 10:05:18  Value Investor HOLDING — "Below fair value, patience" │
└──────────────────────────────────────────────────────────────────┘
```

### 9.2 Key Components

**SimulationPage.tsx**: Main page, manages WebSocket connection and state.

```typescript
// pages/SimulationPage.tsx
import { useState, useEffect, useRef, useCallback } from 'react';

interface SimulationState {
  status: 'setup' | 'running' | 'complete';
  simulationId: string | null;
  agents: AgentInfo[];
  priceHistory: PricePoint[];
  leaderboard: PortfolioSummary[];
  trades: TradeEvent[];
  commentary: string[];
  currentTick: number;
  totalTicks: number;
  marketRegime: string;
}

export function SimulationPage() {
  const [state, setState] = useState<SimulationState>({
    status: 'setup',
    simulationId: null,
    agents: [],
    priceHistory: [],
    leaderboard: [],
    trades: [],
    commentary: [],
    currentTick: 0,
    totalTicks: 200,
    marketRegime: 'normal',
  });
  
  const wsRef = useRef<WebSocket | null>(null);
  const [customStrategy, setCustomStrategy] = useState('');
  
  const startSimulation = async () => {
    // 1. Create simulation via REST
    const res = await fetch('/api/simulation/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        total_ticks: 200,
        tick_delay_seconds: 0.3,
        initial_price: 450.0,
        agents: customStrategy ? [{
          name: "My Custom Agent",
          strategy: customStrategy,
          think_interval: 5,
        }] : [],
      }),
    });
    const { simulation_id } = await res.json();
    
    // 2. Connect WebSocket
    const ws = new WebSocket(`ws://localhost:8000/api/simulation/ws/${simulation_id}`);
    wsRef.current = ws;
    
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleEvent(msg);
    };
    
    setState(s => ({ ...s, status: 'running', simulationId: simulation_id }));
  };
  
  const handleEvent = useCallback((event: any) => {
    const { type, data } = event;
    
    setState(prev => {
      switch (type) {
        case 'agents_created':
          return { ...prev, agents: data.agents };
        
        case 'market_tick':
          return {
            ...prev,
            priceHistory: [...prev.priceHistory, {
              time: data.tick,
              open: data.open,
              high: data.high,
              low: data.low,
              close: data.close,
              volume: data.volume,
            }],
            currentTick: data.tick,
            marketRegime: data.regime,
          };
        
        case 'trade':
          return {
            ...prev,
            trades: [data, ...prev.trades].slice(0, 100), // Keep last 100
          };
        
        case 'agent_decision':
          return {
            ...prev,
            trades: [{...data, type: 'decision'}, ...prev.trades].slice(0, 100),
          };
        
        case 'leaderboard_update':
          return { ...prev, leaderboard: data.leaderboard };
        
        case 'commentary':
          return {
            ...prev,
            commentary: [data.text, ...prev.commentary].slice(0, 20),
          };
        
        case 'simulation_complete':
          return { ...prev, status: 'complete' };
        
        default:
          return prev;
      }
    });
  }, []);
  
  // Render setup screen or live simulation...
}
```

**LivePriceChart.tsx**: Real-time candlestick chart that updates with each tick.

```typescript
// Use lightweight-charts, same as Research module
// Key difference: chart updates incrementally via .update() instead of .setData()

useEffect(() => {
  if (latestCandle && candlestickSeriesRef.current) {
    candlestickSeriesRef.current.update(latestCandle);
  }
}, [latestCandle]);
```

**OrderBookVisualizer.tsx**: Horizontal bar chart showing bid/ask depth.

```typescript
// Bids on the left (green), Asks on the right (red)
// Each bar width proportional to quantity
// Updates every leaderboard_update event
```

**AgentLeaderboard.tsx**: Ranked list of agents with live P&L sparklines.

```typescript
// Show: rank, agent name, personality icon, total value, return %, trade count
// Color: green for positive return, red for negative
// Sparkline: mini chart of portfolio value over time (recharts)
```

**TradeLog.tsx**: Scrollable feed of agent decisions and trades.

```typescript
// Each entry: timestamp | agent avatar | action | price | quantity | reasoning
// Color-coded: green for buys, red for sells, gray for holds
// Agent reasoning is the "thinking" text that makes it educational
```

**AICommentaryPanel.tsx**: Chat-like panel with commentator messages.

```typescript
// Scrollable, newest at top
// Each message styled like a chat bubble
// Occasional use of emoji and trader slang
```

### 9.3 Regime Visual Indicators

The chart background color subtly changes based on market regime:
- **Normal**: transparent / dark background (default)
- **Crash**: subtle red tint (`rgba(239, 68, 68, 0.05)`)
- **Rally**: subtle green tint (`rgba(34, 197, 94, 0.05)`)

This is a small detail that makes the UI feel alive and helps users understand what's happening.

### 9.4 Color Scheme

Same dark theme as Research module for consistency:
```
Background:     #0f172a (slate-900)
Surface:        #1e293b (slate-800)
Text Primary:   #f1f5f9 (slate-100)
Green (Buy):    #22c55e
Red (Sell):     #ef4444
Amber (Hold):   #f59e0b
Blue (Custom):  #3b82f6

Agent Colors (for chart/leaderboard):
  Quant:    #8b5cf6 (violet)
  Value:    #3b82f6 (blue)
  Retail:   #f97316 (orange)
  Custom:   #06b6d4 (cyan)
```

---

## 10. Data Models & Schemas

### Pydantic Models (Backend)

```python
from pydantic import BaseModel
from typing import Optional
from enum import Enum

class AgentType(str, Enum):
    QUANT = "quant"
    FUNDAMENTAL = "fundamental"
    RETAIL = "retail"
    CUSTOM_LLM = "custom_llm"

class CustomAgentConfig(BaseModel):
    name: str = "My Agent"
    strategy: str  # Natural language strategy description
    think_interval: int = 5  # LLM call frequency in ticks

class SimulationConfig(BaseModel):
    total_ticks: int = 200
    tick_delay_seconds: float = 0.3
    symbol: str = "SIM"
    initial_price: float = 450.0
    initial_cash_per_agent: float = 100_000.0
    volatility: float = 0.02
    crash_probability: float = 0.005
    agents: list[CustomAgentConfig] = []  # User-defined agents

class SimulationCreateResponse(BaseModel):
    simulation_id: str
    status: str  # "created"

class AgentInfo(BaseModel):
    id: str
    name: str
    personality: str
    initial_cash: float

class TradeEvent(BaseModel):
    trade_id: str
    symbol: str
    price: float
    quantity: int
    buyer_id: str
    seller_id: str
    timestamp: str

class AgentDecision(BaseModel):
    agent_id: str
    agent_name: str
    side: str
    price: float
    quantity: int
    reasoning: str

class PortfolioSummary(BaseModel):
    agent_id: str
    cash: float
    positions: dict[str, int]
    total_value: float
    total_return_pct: float
    realized_pnl: float
    unrealized_pnl: float
    trade_count: int

class MarketTick(BaseModel):
    tick: int
    symbol: str
    price: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    regime: str
    return_pct: float
    rsi: float
    sma_20: float
    sma_50: float
```

---

## 11. API Endpoints & WebSocket Events

### REST Endpoints

```
POST /api/simulation/create
     Body: SimulationConfig
     Returns: { simulation_id: str, status: "created" }

GET  /api/simulation/{sim_id}/status
     Returns: { status: "created"|"running"|"complete", tick: int, agents: [...] }

POST /api/simulation/{sim_id}/stop
     Returns: { status: "stopped" }
     Side effect: terminates the Modal Sandbox
```

### WebSocket

```
WS   /api/simulation/ws/{sim_id}
     Bidirectional:
       Server → Client: simulation events (JSON, see event types in Section 7.2)
       Client → Server: user interventions (future: pause, adjust params)
```

---

## 12. Integration with Research Module

The Simulation and Research modules share the same backend and frontend. Key integration points:

### 12.1 Shared Backend
- Same FastAPI app (`main.py`), simulation gets its own router
- Shared config (`.env`, `config.py`)
- Shared Supabase connection (if used for persistence)

### 12.2 Research → Simulation Data Feed
Custom LLM agents can optionally pull real-time sentiment from the Research module's Perplexity Sonar integration:

```python
# Inside LLMAgent.decide(), optionally enrich context:
# "According to Perplexity, current sentiment for {ticker} is Bullish because..."
# This creates a feedback loop: Research data → Simulation decisions
```

### 12.3 Frontend Navigation
- Shared navigation bar: **Research** | **Simulation** | **Tracker** | **Agents**
- React Router: `/research/:symbol`, `/simulation`, `/tracker`, `/agents`
- Shared components: dark theme, chart library, layout components

### 12.4 Agent Observability Integration
Every simulation event (agent decisions, trades, commentary) MUST be logged to the `agent_activity` table and broadcast via WebSocket:
```python
await log_agent_activity(
    user_id=user_id, module="simulation",
    agent_name="Quant Bot", action=f"BUY 50 shares at $873.20",
    details={"reasoning": "RSI oversold", "portfolio_value": 97340},
    status="success"
)
```
This shows up in real-time in the /agents observability dashboard, even when the user is not on the simulation page.

### 12.5 Poke Integration
When a simulation completes, optionally send a Poke notification:
```python
await send_poke_message(
    f"🎮 Simulation complete!\n"
    f"Winner: Quant Bot (+4.2%)\n"
    f"Your agent: +1.8%\n"
    f"🔗 View results: https://tickermaster.vercel.app/simulation/{sim_id}"
)
```

---

## 13. Environment & Configuration

### Additional API Keys (beyond Research module)

Add to `.env`:
```env
# Modal (for Sandboxes)
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret

# OpenRouter (for LLM agents + commentator)
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxx
```

### Additional Python Dependencies

Add to `requirements.txt`:
```
modal==0.73.0
openai==1.50.0        # OpenRouter uses OpenAI-compatible SDK
numpy==1.26.0         # For GBM price generation (also in sandbox image)
websockets==13.0      # WebSocket support
```

### Frontend Dependencies (Additional)

```json
{
  "dependencies": {
    "reconnecting-websocket": "^4.4.0"
  }
}
```

---

## 14. File Structure

```
project/
├── backend/
│   ├── main.py                         # FastAPI app (shared, add simulation router)
│   ├── config.py                       # Environment variables (shared)
│   ├── routers/
│   │   ├── research.py                 # Research module routes
│   │   └── simulation.py               # Simulation routes + WebSocket
│   ├── services/
│   │   ├── ... (research services)
│   │   ├── modal_manager.py            # Modal Sandbox creation + management
│   │   ├── simulation_engine.py        # Self-contained engine code (string/file)
│   │   └── commentator.py             # AI market commentator
│   ├── models/
│   │   ├── schemas.py                  # Shared Pydantic models
│   │   └── simulation_schemas.py       # Simulation-specific models
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Router (add /simulation route)
│   │   ├── pages/
│   │   │   ├── HomePage.tsx
│   │   │   ├── TickerPage.tsx          # Research
│   │   │   └── SimulationPage.tsx      # Simulation arena
│   │   ├── components/
│   │   │   ├── ... (research components)
│   │   │   ├── simulation/
│   │   │   │   ├── AgentSetupPanel.tsx       # Create custom agent UI
│   │   │   │   ├── LivePriceChart.tsx        # Real-time candlestick
│   │   │   │   ├── OrderBookVisualizer.tsx   # Bid/ask depth bars
│   │   │   │   ├── AgentLeaderboard.tsx      # Ranked P&L table
│   │   │   │   ├── TradeLog.tsx              # Scrollable trade feed
│   │   │   │   ├── AICommentaryPanel.tsx     # Commentator chat
│   │   │   │   ├── AgentCard.tsx             # Individual agent info card
│   │   │   │   ├── RegimeIndicator.tsx       # Market regime badge
│   │   │   │   └── SimulationControls.tsx    # Start/stop/config buttons
│   │   │   └── common/
│   │   │       └── ... (shared components)
│   │   ├── hooks/
│   │   │   └── useSimulationWebSocket.ts     # WebSocket connection hook
│   │   └── utils/
│   │       └── ...
│   └── ...
```

---

## 15. Implementation Order

### Phase 1: Simulation Engine (Get it running locally first)
1. Write the complete simulation engine code (`simulation_engine.py`) as a standalone Python script
2. Test locally: run it, verify JSON events print to stdout correctly
3. Verify: order book matching works, agents trade, portfolios update, leaderboard sorts correctly
4. This should work as `python simulation_engine.py` with a hardcoded config

### Phase 2: Modal Sandbox Integration
5. Set up Modal app and image (`modal_manager.py`)
6. Write the sandbox creation + engine injection code
7. Test: create a sandbox, inject the engine, read events from stdout
8. Verify: events stream back correctly from inside the sandbox

### Phase 3: WebSocket + Backend
9. Create the simulation router with REST + WebSocket endpoints
10. Wire up: POST /create → Modal Sandbox → stdout → WebSocket → client
11. Add the AI commentator service
12. Test with a simple WebSocket client (wscat or browser console)

### Phase 4: Frontend — Setup Screen
13. Build `SimulationPage.tsx` with the setup screen
14. Agent setup panel: show 3 preset agents, text input for custom agent
15. "Start Simulation" button that calls the API and connects WebSocket

### Phase 5: Frontend — Live Visualization
16. Build `LivePriceChart.tsx` (real-time updating candlestick)
17. Build `AgentLeaderboard.tsx` (live P&L rankings)
18. Build `OrderBookVisualizer.tsx` (depth chart)
19. Build `TradeLog.tsx` (scrollable feed with agent reasoning)
20. Build `AICommentaryPanel.tsx` (commentator messages)

### Phase 6: Polish + LLM Agent
21. Wire up the `LLMAgent` with OpenRouter (test inside sandbox)
22. Add regime visual indicators (background color changes)
23. Add simulation complete screen (final results, winner announcement)
24. Error handling for sandbox failures, WebSocket disconnects
25. Reconnection logic for WebSocket (`reconnecting-websocket`)

### Phase 7: Demo Prep
26. Pre-configure a good demo scenario: set initial_price, volatility, crash_probability for dramatic results
27. Have a pre-written custom strategy that performs well: "Buy when RSI < 25 during a crash regime, sell when RSI > 75"
28. Test the full flow: setup → start → watch 200 ticks → see results
29. Ensure the AI commentator generates entertaining commentary
30. Have a fallback: if Modal is slow, have a "local mode" that runs the engine directly (not in sandbox)

---

## Links to All Tools & APIs (Simulation)

| Tool | URL | Notes |
|------|-----|-------|
| Modal Sandbox | https://modal.com/docs/guide/sandboxes | Simulation runtime |
| Modal Python SDK | https://pypi.org/project/modal/ | `modal.Sandbox.create()` |
| OpenRouter API | https://openrouter.ai/docs/api/reference | LLM for custom agents + commentator |
| OpenRouter Models | https://openrouter.ai/models | Use `openai/gpt-4o-mini` |
| FastAPI WebSockets | https://fastapi.tiangolo.com/advanced/websockets/ | Real-time streaming |
| lightweight-charts | https://tradingview.github.io/lightweight-charts/ | Live price chart |
| reconnecting-websocket | https://github.com/pladaria/reconnecting-websocket | Auto-reconnect |

---

## Critical Notes for the Coding Agent

1. **The simulation engine must be SELF-CONTAINED.** It runs inside a Modal Sandbox with no access to the backend's code. Everything it needs (OrderBook, Portfolio, Agents, MarketDataGenerator) must be defined in a single file or injected as a string.

2. **All output from the sandbox is via stdout JSON lines.** Use `print(json.dumps(event), flush=True)` for every event. The `flush=True` is critical for real-time streaming.

3. **The WebSocket must handle multiple simultaneous simulations.** Each simulation has its own sandbox and client list. Don't use global state that conflicts between simulations.

4. **Modal Sandbox has a default 5-minute timeout.** Set `timeout=600` (10 min) in `Sandbox.create()` for longer simulations.

5. **OpenRouter API key must be passed into the sandbox as a Modal Secret.** The sandbox environment is isolated — it can't read the backend's `.env` file.

6. **The LLM agent should fail gracefully.** If OpenRouter is slow or returns garbage, the agent should just skip that tick (hold). Never let an LLM failure crash the simulation.

7. **The AI commentator runs on the BACKEND, not in the sandbox.** It consumes events from the sandbox stream and generates commentary independently. This avoids adding LLM latency to the simulation tick loop.

8. **Keep tick_delay_seconds at 0.3 for demos.** This gives ~60 seconds for a 200-tick simulation — fast enough to be exciting, slow enough to follow.

9. **The 3 preset agents must ALWAYS be present.** They represent the real market archetypes. The user's custom agent competes against them.

10. **Information asymmetry is key to the educational value.** Make sure the quant agent always acts before the retail agent. Sort agents by reaction speed before the decision loop.

11. **The order book must be properly seeded with liquidity.** Without market maker orders, the first agent trades will fail (no counter-party). Replenish every 10 ticks.

12. **Frontend chart updates must be INCREMENTAL.** Don't re-render the entire chart on each tick. Use `candlestickSeries.update(newCandle)` from lightweight-charts for O(1) updates.

13. **The Tracker module can auto-launch simulations.** The `POST /api/simulation/create` endpoint must work when called from the Tracker's Modal Cron function (via HTTP). This is the key cross-module integration: Tracker detects crash → creates a simulation with crash conditions → agents trade the scenario → user gets a Poke notification with a link to watch.

14. **Supabase is the shared database.** Use it to optionally persist simulation results. Connection string uses `SUPABASE_URL` and `SUPABASE_KEY` from `.env`.
