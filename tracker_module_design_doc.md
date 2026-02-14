# TickerMaster — Tracker Module: Technical Implementation Spec

> **Context**: This is a hackathon project. The Tracker module is one of three core features in **TickerMaster** (Research, Simulation, Tracker). The Tracker is the "always-on" layer — users deploy persistent agents on tickers that monitor prices, detect anomalies, pull real-time context via the Research module, optionally spin up simulations, and push notifications to the user's phone via **Poke** (iMessage/SMS). This is also the glue that ties the other two modules together. Polling runs on **Modal Cron**, investigation uses **Perplexity Sonar** (via Research module), and notifications go through **Poke's webhook API**. Every decision has been made — the coding agent should execute, not deliberate.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Tracker Agent Lifecycle](#3-tracker-agent-lifecycle)
4. [Modal Cron — Polling Engine](#4-modal-cron--polling-engine)
5. [Trigger Detection System](#5-trigger-detection-system)
6. [Investigation Pipeline (Research Integration)](#6-investigation-pipeline-research-integration)
7. [Analysis & Narrative Generation](#7-analysis--narrative-generation)
8. [Poke Notification System](#8-poke-notification-system)
9. [Simulation Integration](#9-simulation-integration)
10. [Database Schema (Supabase)](#10-database-schema-supabase)
11. [Backend API & WebSocket Events](#11-backend-api--websocket-events)
12. [Frontend Implementation](#12-frontend-implementation)
13. [Data Models & Schemas](#13-data-models--schemas)
14. [Environment & Configuration](#14-environment--configuration)
15. [File Structure](#15-file-structure)
16. [Implementation Order](#16-implementation-order)

---

## 1. Project Overview

### What This Is
A deploy-and-forget agent system where users create persistent tracking agents on stock tickers. Each agent continuously monitors its ticker, detects significant events (price spikes, volume anomalies, sentiment shifts), investigates **why** something happened using the Research module's APIs (Perplexity Sonar, Reddit, prediction markets), generates a concise narrative, and pushes an actionable notification to the user's phone via Poke. Users can also configure agents to auto-launch simulations when certain conditions are met.

### Core Pipeline (Per Agent, Per Tick)

```
TRIGGER → INVESTIGATE → ANALYZE → NOTIFY
   │           │            │          │
   ▼           ▼            ▼          ▼
Modal Cron   Research     OpenAI     Poke
polls price  module APIs  synthesize  iMessage/SMS
& volume     (Perplexity, narrative   notification
every 1 min  Reddit, etc)             + deep link
```

### Core Value Proposition
"Deploy an AI agent on any ticker. It watches, thinks, and texts you when it matters." Unlike price alerts on Robinhood (which just say "$AAPL hit $190"), TickerMaster tells you **why** it moved and **what you should consider doing**.

### What Makes This Impressive (Prize Alignment)

| Prize | How We Qualify |
|-------|---------------|
| **Interaction Co. (Poke)** | Poke is the primary notification channel. Agents text users via Poke with rich context — not just "price alert" but a full narrative with the reason + a link back to the dashboard. |
| **Modal — Sandbox/Cron** | Modal Cron jobs power the polling engine. Each deployed agent is a Modal `@app.function(schedule=modal.Period(minutes=1))` that runs persistently. |
| **Perplexity Sonar** | Every triggered investigation calls the Research module's Perplexity Sonar integration to fetch real-time "why" context. |
| **Greylock — Multi-Turn Agent** | The tracker agent is a multi-step agent: Trigger → Investigate → Analyze → Notify → (optionally) Simulate. Each step feeds into the next. |
| **Neo — Product** | This is the most "product-like" feature. Persistent agents + phone notifications = real utility. |

---

## 2. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                      React Frontend                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────────┐  │
│  │ Deploy Agent │ │ Active Agent │ │ Alert History / Feed       │  │
│  │ Form         │ │ Dashboard    │ │ (notifications log)        │  │
│  └──────────────┘ └──────────────┘ └────────────────────────────┘  │
└──────────────────────┬─────────────────────────────────────────────┘
                       │ REST + WebSocket
┌──────────────────────▼─────────────────────────────────────────────┐
│                    FastAPI Backend                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│  │ Tracker      │ │ Alert        │ │ Poke         │               │
│  │ CRUD Service │ │ Evaluator    │ │ Notification │               │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘               │
│         │                │                │                        │
│  ┌──────▼────────────────▼────────────────▼──────────────────┐    │
│  │                  Orchestrator Service                       │    │
│  │  Coordinates: Research API → Analysis → Poke → Simulation  │    │
│  └──────┬────────────────────────────────────────────────────┘    │
└─────────┼──────────────────────────────────────────────────────────┘
          │
    ┌─────▼──────┐     ┌───────────────┐     ┌──────────────────┐
    │ Modal Cron │────▶│ Research      │────▶│ Poke Webhook     │
    │ (polling)  │     │ Module APIs   │     │ (notification)   │
    └────────────┘     └───────────────┘     └──────────────────┘
          │
    ┌─────▼──────┐
    │ Supabase   │
    │ (state)    │
    └────────────┘
```

### Tech Stack (Tracker-Specific)

| Layer | Technology | Notes |
|-------|-----------|-------|
| **Polling Engine** | Modal Cron (`modal.Period(minutes=1)`) | Persistent, serverless, auto-scaling |
| **Price Data** | yfinance (shared with Research) | Quick price/volume checks |
| **Investigation** | Research Module internal APIs | Perplexity Sonar, Reddit, Finviz (already built) |
| **Analysis LLM** | OpenAI GPT-4o-mini (via OpenRouter) | Synthesize raw data → narrative |
| **Notifications** | Poke Webhook API | iMessage/SMS push to user's phone |
| **State Storage** | Supabase (PostgreSQL) | Agent configs, alert history, trigger state |
| **Real-Time UI** | FastAPI WebSockets (shared) | Push alert events to frontend |
| **Frontend** | React 18 (shared with Research/Simulation) | Agent management + alert feed |

---

## 3. Tracker Agent Lifecycle

### 3.1 Agent States

```
CREATED → ACTIVE → TRIGGERED → INVESTIGATING → NOTIFIED → ACTIVE (loop)
                                                    │
                                                    └→ SIMULATING (optional)
```

### 3.2 User Creates an Agent

User fills out a form:
- **Ticker**: e.g., `NVDA`
- **Name**: e.g., "NVDA Earnings Watch"
- **Alert Conditions** (pick one or more):
  - Price change > X% in Y minutes
  - Volume spike > X× average
  - RSI crosses above/below threshold
  - Custom natural language condition (e.g., "any major news about AI chips")
- **Notification preferences**: what to include in the Poke message
- **Auto-simulate**: optionally auto-launch a simulation when triggered

### 3.3 Agent Gets Deployed to Modal

When created, a tracker agent config gets saved to Supabase. The Modal Cron job — which runs every minute — reads ALL active agents from the database and evaluates each one.

This is important: **we don't create a separate Modal function per agent**. Instead, we have ONE cron function that loops over all active agents. This is simpler for a hackathon and avoids needing to dynamically deploy/undeploy Modal functions.

---

## 4. Modal Cron — Polling Engine

### 4.1 Modal App Setup for Cron

```python
# tracker/modal_cron.py
import modal

app = modal.App("tickermaster-tracker")

tracker_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "yfinance",
    "httpx",
    "supabase",
    "numpy",
)

@app.function(
    schedule=modal.Period(minutes=1),
    image=tracker_image,
    secrets=[
        modal.Secret.from_name("tickermaster-secrets"),  # Contains all API keys
    ],
    timeout=120,  # 2 min max per poll cycle
)
def poll_all_agents():
    """
    Runs every minute via Modal Cron.
    Fetches all active tracker agents from Supabase,
    evaluates trigger conditions, and fires alerts.
    """
    import os
    import json
    import httpx
    import yfinance as yf
    from supabase import create_client
    
    # Connect to Supabase
    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
    )
    
    # Fetch all active agents
    result = supabase.table("tracker_agents").select("*").eq("status", "active").execute()
    agents = result.data
    
    if not agents:
        return
    
    # Group agents by ticker to minimize API calls
    ticker_groups: dict[str, list[dict]] = {}
    for agent in agents:
        symbol = agent["symbol"]
        if symbol not in ticker_groups:
            ticker_groups[symbol] = []
        ticker_groups[symbol].append(agent)
    
    # Fetch price data for each unique ticker
    for symbol, agent_list in ticker_groups.items():
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price + recent history for change calculations
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                continue
            
            current_price = hist["Close"].iloc[-1]
            current_volume = hist["Volume"].iloc[-1]
            
            # 5-min price change
            price_5m_ago = hist["Close"].iloc[-5] if len(hist) >= 5 else hist["Close"].iloc[0]
            change_5m_pct = ((current_price - price_5m_ago) / price_5m_ago) * 100
            
            # 15-min price change
            price_15m_ago = hist["Close"].iloc[-15] if len(hist) >= 15 else hist["Close"].iloc[0]
            change_15m_pct = ((current_price - price_15m_ago) / price_15m_ago) * 100
            
            # Average volume (for spike detection)
            avg_volume = hist["Volume"].mean() if len(hist) > 0 else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # RSI (14-period using minute data)
            close = hist["Close"]
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(close) >= 15 else 50.0
            
            market_snapshot = {
                "symbol": symbol,
                "price": round(float(current_price), 2),
                "change_5m_pct": round(float(change_5m_pct), 4),
                "change_15m_pct": round(float(change_15m_pct), 4),
                "volume": int(current_volume),
                "volume_ratio": round(float(volume_ratio), 2),
                "rsi": round(float(rsi), 2),
                "avg_volume": int(avg_volume),
            }
            
            # Evaluate each agent's triggers against this snapshot
            for agent in agent_list:
                triggered = evaluate_triggers(agent, market_snapshot)
                
                if triggered:
                    # Check cooldown (don't spam — min 15 min between alerts)
                    last_alert = agent.get("last_alert_at")
                    if last_alert:
                        from datetime import datetime, timedelta
                        last_dt = datetime.fromisoformat(last_alert)
                        if datetime.utcnow() - last_dt < timedelta(minutes=15):
                            continue  # Still in cooldown
                    
                    # FIRE THE PIPELINE: Investigate → Analyze → Notify
                    fire_alert_pipeline(
                        agent=agent,
                        market_snapshot=market_snapshot,
                        trigger_reason=triggered,
                        supabase=supabase,
                    )
        
        except Exception as e:
            print(f"Error polling {symbol}: {e}")
            continue


def evaluate_triggers(agent: dict, snapshot: dict) -> str | None:
    """
    Check if any of the agent's trigger conditions are met.
    Returns a trigger reason string, or None if no trigger.
    """
    triggers = agent.get("triggers", {})
    reasons = []
    
    # Price change trigger
    price_threshold = triggers.get("price_change_pct")
    if price_threshold:
        if abs(snapshot["change_5m_pct"]) >= price_threshold:
            direction = "up" if snapshot["change_5m_pct"] > 0 else "down"
            reasons.append(
                f"Price moved {direction} {abs(snapshot['change_5m_pct']):.2f}% in 5 min "
                f"(threshold: {price_threshold}%)"
            )
    
    # Volume spike trigger
    volume_threshold = triggers.get("volume_spike_ratio")
    if volume_threshold:
        if snapshot["volume_ratio"] >= volume_threshold:
            reasons.append(
                f"Volume spike: {snapshot['volume_ratio']:.1f}× average "
                f"(threshold: {volume_threshold}×)"
            )
    
    # RSI trigger
    rsi_upper = triggers.get("rsi_upper")
    rsi_lower = triggers.get("rsi_lower")
    if rsi_upper and snapshot["rsi"] > rsi_upper:
        reasons.append(f"RSI overbought: {snapshot['rsi']:.1f} (threshold: {rsi_upper})")
    if rsi_lower and snapshot["rsi"] < rsi_lower:
        reasons.append(f"RSI oversold: {snapshot['rsi']:.1f} (threshold: {rsi_lower})")
    
    # Price level trigger (absolute)
    price_above = triggers.get("price_above")
    price_below = triggers.get("price_below")
    if price_above and snapshot["price"] >= price_above:
        reasons.append(f"Price broke above ${price_above}")
    if price_below and snapshot["price"] <= price_below:
        reasons.append(f"Price broke below ${price_below}")
    
    return " | ".join(reasons) if reasons else None
```

### 4.2 Deploying the Cron Job

```bash
# Deploy the cron job to Modal (one-time setup)
modal deploy tracker/modal_cron.py
```

Once deployed, this function runs **every minute** automatically. It scales to zero when not running and spins up instantly when the schedule fires. No servers to manage.

---

## 5. Trigger Detection System

### 5.1 Supported Triggers

| Trigger | Config Key | Example | Description |
|---------|-----------|---------|-------------|
| **Price Change %** | `price_change_pct` | `2.0` | Fire when price moves ≥2% in 5 min |
| **Volume Spike** | `volume_spike_ratio` | `3.0` | Fire when volume ≥3× average |
| **RSI Overbought** | `rsi_upper` | `75` | Fire when RSI crosses above 75 |
| **RSI Oversold** | `rsi_lower` | `25` | Fire when RSI crosses below 25 |
| **Price Above** | `price_above` | `200.0` | Fire when price crosses above $200 |
| **Price Below** | `price_below` | `150.0` | Fire when price crosses below $150 |

### 5.2 Cooldown System

To prevent notification spam, each agent has a **15-minute cooldown** after firing. The `last_alert_at` timestamp in Supabase is checked before triggering.

---

## 6. Investigation Pipeline (Research Integration)

When a trigger fires, the agent calls the **Research module's backend APIs** to gather context. This is the key integration point — the Tracker doesn't duplicate Research logic, it calls it.

```python
def fire_alert_pipeline(
    agent: dict,
    market_snapshot: dict,
    trigger_reason: str,
    supabase,
):
    """
    The full Trigger → Investigate → Analyze → Notify pipeline.
    Runs inside the Modal Cron function.
    """
    import httpx
    import os
    import json
    from datetime import datetime
    
    symbol = agent["symbol"]
    agent_id = agent["id"]
    
    # =========================================
    # STEP 1: INVESTIGATE via Research Module
    # =========================================
    # Call the Research module's internal APIs
    # These are the same endpoints the Research frontend uses
    
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    investigation = {}
    
    try:
        with httpx.Client(timeout=25.0) as client:
            # 1a. Get AI research summary (Perplexity Sonar)
            res = client.get(f"{backend_url}/api/ticker/{symbol}/ai-research")
            if res.status_code == 200:
                investigation["ai_research"] = res.json()
            
            # 1b. Get sentiment data (Reddit + X/Twitter + composite)
            res = client.get(f"{backend_url}/api/ticker/{symbol}/sentiment")
            if res.status_code == 200:
                investigation["sentiment"] = res.json()
            
            # 1c. Get X/Twitter sentiment specifically
            res = client.get(f"{backend_url}/api/ticker/{symbol}/x-sentiment")
            if res.status_code == 200:
                investigation["x_sentiment"] = res.json()
            
            # 1d. Get prediction market data (Kalshi + Polymarket)
            res = client.get(f"{backend_url}/api/prediction-markets?query={symbol}")
            if res.status_code == 200:
                investigation["prediction_markets"] = res.json()
    
    except Exception as e:
        investigation["error"] = str(e)
    
    # =========================================
    # STEP 2: ANALYZE — synthesize into narrative
    # =========================================
    narrative = generate_alert_narrative(
        symbol=symbol,
        market_snapshot=market_snapshot,
        trigger_reason=trigger_reason,
        investigation=investigation,
    )
    
    # =========================================
    # STEP 3: NOTIFY via Poke
    # =========================================
    poke_sent = send_poke_notification(
        agent=agent,
        narrative=narrative,
        market_snapshot=market_snapshot,
    )
    
    # =========================================
    # STEP 4: SAVE alert to database
    # =========================================
    alert_record = {
        "agent_id": agent_id,
        "symbol": symbol,
        "trigger_reason": trigger_reason,
        "narrative": narrative,
        "market_snapshot": json.dumps(market_snapshot),
        "investigation_data": json.dumps(investigation),
        "poke_sent": poke_sent,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    supabase.table("tracker_alerts").insert(alert_record).execute()
    
    # Update the agent's last_alert_at
    supabase.table("tracker_agents").update({
        "last_alert_at": datetime.utcnow().isoformat(),
        "total_alerts": agent.get("total_alerts", 0) + 1,
    }).eq("id", agent_id).execute()
    
    # =========================================
    # STEP 5: (OPTIONAL) Auto-simulate
    # =========================================
    if agent.get("auto_simulate", False):
        trigger_simulation(agent, market_snapshot, narrative)
    
    # Emit event to WebSocket for live dashboard updates
    try:
        httpx.post(f"{backend_url}/api/tracker/emit-alert", json={
            "agent_id": agent_id,
            "alert": alert_record,
        }, timeout=5.0)
    except Exception:
        pass  # Non-critical
```

---

## 7. Analysis & Narrative Generation

The analysis step takes raw investigation data and synthesizes it into a concise, actionable narrative using an LLM.

```python
def generate_alert_narrative(
    symbol: str,
    market_snapshot: dict,
    trigger_reason: str,
    investigation: dict,
) -> str:
    """
    Use OpenAI (via OpenRouter) to synthesize all data into a
    concise, actionable narrative for the user.
    """
    import httpx
    import os
    
    # Extract key data from investigation
    ai_summary = ""
    if "ai_research" in investigation:
        ai_summary = investigation["ai_research"].get("summary", "")[:500]
    
    sentiment_info = ""
    if "sentiment" in investigation:
        s = investigation["sentiment"]
        composite = s.get("composite", {})
        sentiment_info = f"Sentiment: {composite.get('label', 'N/A')} ({composite.get('composite_score', 0)}/100)"
    
    prediction_info = ""
    if "prediction_markets" in investigation:
        pm = investigation["prediction_markets"]
        kalshi = pm.get("kalshi", [])[:3]
        if kalshi:
            prediction_info = "Prediction Markets: " + "; ".join(
                f"{m.get('title', 'N/A')}: {m.get('implied_probability', 'N/A')}%"
                for m in kalshi
            )
    
    prompt = f"""You are a financial alert analyst for TickerMaster. Generate a CONCISE alert narrative (max 4 sentences).

ALERT TRIGGER:
- Ticker: ${symbol} at ${market_snapshot['price']}
- Trigger: {trigger_reason}
- 5min Change: {market_snapshot['change_5m_pct']:.2f}%
- Volume: {market_snapshot['volume_ratio']:.1f}× average
- RSI: {market_snapshot['rsi']:.1f}

RESEARCH CONTEXT:
{ai_summary[:300] if ai_summary else 'No research data available'}

{sentiment_info}
{prediction_info}

Write a SHORT, actionable alert. Format:
1. What happened (1 sentence)
2. Why it likely happened (1-2 sentences based on research)
3. What to consider doing (1 sentence)

Be direct. No fluff. Think Bloomberg terminal alert, not blog post."""

    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a concise financial alert analyst. Maximum 4 sentences."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 250,
            },
            timeout=10.0,
        )
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    except Exception as e:
        # Fallback: generate a basic alert without LLM
        direction = "📈" if market_snapshot["change_5m_pct"] > 0 else "📉"
        return (
            f"{direction} ${symbol} alert: Price at ${market_snapshot['price']} "
            f"({market_snapshot['change_5m_pct']:+.2f}% in 5min). "
            f"Trigger: {trigger_reason}. "
            f"Check the dashboard for full analysis."
        )
```

---

## 8. Poke Notification System

### 8.1 Poke API Integration

Poke uses a simple webhook endpoint. Sending a message to Poke delivers it to the user via iMessage, WhatsApp, or SMS — whatever they've configured.

**API Endpoint**: `POST https://poke.com/api/v1/inbound-sms/webhook`
**Auth**: `Authorization: Bearer {POKE_API_KEY}`
**API Key**: Generated at `https://poke.com/settings/advanced`

```python
def send_poke_notification(
    agent: dict,
    narrative: str,
    market_snapshot: dict,
) -> bool:
    """
    Send a push notification to the user via Poke.
    Poke delivers via iMessage/SMS/WhatsApp.
    
    Returns True if sent successfully.
    """
    import httpx
    import os
    
    poke_api_key = os.environ.get("POKE_API_KEY")
    if not poke_api_key:
        print("POKE_API_KEY not set, skipping notification")
        return False
    
    symbol = market_snapshot["symbol"]
    price = market_snapshot["price"]
    change = market_snapshot["change_5m_pct"]
    direction = "📈" if change > 0 else "📉"
    
    # Build the Poke message
    # Keep it concise — this is a text message, not an email
    dashboard_url = f"https://tickermaster.vercel.app/research/{symbol}"
    
    message = (
        f"{direction} TickerMaster Alert: ${symbol} ${price} ({change:+.2f}%)\n\n"
        f"{narrative}\n\n"
        f"🔗 View full analysis: {dashboard_url}"
    )
    
    try:
        response = httpx.post(
            "https://poke.com/api/v1/inbound-sms/webhook",
            headers={
                "Authorization": f"Bearer {poke_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "message": message,
            },
            timeout=10.0,
        )
        
        return response.status_code == 200
    
    except Exception as e:
        print(f"Poke notification failed: {e}")
        return False
```

### 8.2 What the User Receives

The user gets a text message (via Poke on iMessage/SMS) like:

```
📉 TickerMaster Alert: $NVDA $875.30 (-3.42%)

NVDA dropped 3.4% in 5 minutes on heavy volume (4.2× average).
The sell-off appears tied to new export restrictions on AI chips
to China, which Reuters reported this morning. Reddit sentiment
has shifted sharply bearish (-62/100). Consider reviewing your
position or waiting for a stabilization before adding.

🔗 View full analysis: https://tickermaster.vercel.app/research/NVDA
```

### 8.3 Deep Link Back to Dashboard

The notification includes a link to the Research page for that ticker. When the user taps it, they see the full analysis — charts, sentiment, AI research summary, prediction markets — all from the Research module.

---

## 9. Simulation Integration

### 9.1 Auto-Simulate on Trigger

If the user enabled `auto_simulate`, the tracker agent can automatically spin up a Simulation when a trigger fires. This creates a feedback loop:

**Trigger** → "NVDA just dropped 3%" → **Auto-simulate** → "What happens if you buy the dip?" → Agent shows you how a momentum bot vs value investor would trade this scenario.

```python
def trigger_simulation(agent: dict, market_snapshot: dict, narrative: str):
    """
    Auto-launch a simulation when a tracker agent triggers.
    Creates a simulation config based on current market conditions
    and posts it to the Simulation module's API.
    """
    import httpx
    import os
    
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    
    symbol = market_snapshot["symbol"]
    price = market_snapshot["price"]
    
    # Determine initial conditions based on the trigger
    # If the trigger was a crash, start the sim in a volatile regime
    is_crash = market_snapshot["change_5m_pct"] < -2.0
    
    sim_config = {
        "total_ticks": 100,  # Shorter sim for auto-triggered
        "tick_delay_seconds": 0.2,
        "symbol": symbol,
        "initial_price": price,
        "initial_cash_per_agent": 100_000.0,
        "volatility": 0.04 if is_crash else 0.02,  # Higher vol if crash
        "crash_probability": 0.02 if is_crash else 0.005,
        "agents": [
            {
                "name": f"Dip Buyer ({symbol})",
                "strategy": (
                    f"The stock just dropped {market_snapshot['change_5m_pct']:.1f}%. "
                    f"Context: {narrative[:200]}. "
                    f"You believe this is an overreaction. Buy the dip aggressively."
                ),
                "think_interval": 3,
            }
        ],
    }
    
    try:
        response = httpx.post(
            f"{backend_url}/api/simulation/create",
            json=sim_config,
            timeout=10.0,
        )
        
        if response.status_code == 200:
            sim_data = response.json()
            sim_id = sim_data.get("simulation_id")
            
            # Notify user that a simulation was auto-launched
            sim_url = f"https://tickermaster.vercel.app/simulation/{sim_id}"
            send_poke_notification(
                agent=agent,
                narrative=f"🎮 Auto-simulation launched for ${symbol}! Watch AI agents trade this scenario: {sim_url}",
                market_snapshot=market_snapshot,
            )
    
    except Exception as e:
        print(f"Auto-simulation failed: {e}")
```

---

## 10. Database Schema (Supabase)

### 10.1 Tables

```sql
-- Tracker Agents: stores each deployed agent's config
CREATE TABLE tracker_agents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',  -- For multi-user (future)
    symbol TEXT NOT NULL,                      -- e.g., "NVDA"
    name TEXT NOT NULL,                        -- e.g., "NVDA Earnings Watch"
    status TEXT NOT NULL DEFAULT 'active',     -- active, paused, deleted
    
    -- Trigger configuration (JSON)
    triggers JSONB NOT NULL DEFAULT '{}',
    -- Example: {
    --   "price_change_pct": 2.0,
    --   "volume_spike_ratio": 3.0,
    --   "rsi_upper": 75,
    --   "rsi_lower": 25,
    --   "price_above": 200.0,
    --   "price_below": 150.0
    -- }
    
    -- Options
    auto_simulate BOOLEAN DEFAULT FALSE,
    
    -- State
    last_alert_at TIMESTAMPTZ,
    total_alerts INTEGER DEFAULT 0,
    last_checked_at TIMESTAMPTZ,
    last_price NUMERIC,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for fast active agent lookup
CREATE INDEX idx_tracker_agents_active ON tracker_agents (status) WHERE status = 'active';
CREATE INDEX idx_tracker_agents_symbol ON tracker_agents (symbol);

-- Tracker Alerts: stores every alert that was fired
CREATE TABLE tracker_alerts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    agent_id UUID REFERENCES tracker_agents(id),
    symbol TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    narrative TEXT,
    market_snapshot JSONB,
    investigation_data JSONB,
    poke_sent BOOLEAN DEFAULT FALSE,
    simulation_id TEXT,  -- If auto-simulate was triggered
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_tracker_alerts_agent ON tracker_alerts (agent_id);
CREATE INDEX idx_tracker_alerts_created ON tracker_alerts (created_at DESC);
```

### 10.2 Supabase Connection

```python
# backend/services/database.py
from supabase import create_client
import os

def get_supabase_client():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
    )
```

---

## 11. Backend API & WebSocket Events

### 11.1 REST Endpoints

```
POST /api/tracker/agents
     Body: { symbol, name, triggers, auto_simulate }
     Returns: { id, symbol, name, status, triggers }
     Creates a new tracker agent

GET  /api/tracker/agents
     Returns: [ { id, symbol, name, status, triggers, total_alerts, last_alert_at } ]
     List all tracker agents for the user

GET  /api/tracker/agents/{agent_id}
     Returns: Full agent details + recent alerts

PATCH /api/tracker/agents/{agent_id}
     Body: { status?, triggers?, name?, auto_simulate? }
     Update agent config (pause/resume/edit triggers)

DELETE /api/tracker/agents/{agent_id}
     Soft-deletes the agent (sets status = 'deleted')

GET  /api/tracker/alerts?agent_id={optional}&limit=20
     Returns: Recent alerts, newest first

POST /api/tracker/emit-alert
     Internal endpoint called by Modal Cron to push alert to WebSocket clients
```

### 11.2 WebSocket Events (Tracker → Frontend)

The backend pushes alert events to connected WebSocket clients in real-time:

```
WS /api/tracker/ws

Events (Server → Client):
{
  "type": "new_alert",
  "data": {
    "agent_id": "...",
    "symbol": "NVDA",
    "trigger_reason": "Price moved down -3.42% in 5 min",
    "narrative": "...",
    "market_snapshot": {...},
    "poke_sent": true,
    "created_at": "..."
  }
}

{
  "type": "agent_checked",
  "data": {
    "agent_id": "...",
    "symbol": "NVDA",
    "price": 875.30,
    "status": "no_trigger"
  }
}
```

---

## 12. Frontend Implementation

### 12.1 Tracker Page Layout (`/tracker`)

```
┌──────────────────────────────────────────────────────────────────┐
│ HEADER: Tracker Agents  |  [+ Deploy New Agent]                  │
├──────────────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────┐           │
│ │ ACTIVE AGENTS (grid of cards)                      │           │
│ │                                                    │           │
│ │ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │           │
│ │ │ 🟢 NVDA      │ │ 🟢 AAPL      │ │ 🟢 TSLA     │ │           │
│ │ │ $875.30      │ │ $192.44      │ │ $248.50     │ │           │
│ │ │ Price ±2%    │ │ Vol 3x       │ │ RSI < 30    │ │           │
│ │ │ 5 alerts     │ │ 2 alerts     │ │ 0 alerts    │ │           │
│ │ │ Last: 2h ago │ │ Last: 1d ago │ │ Never       │ │           │
│ │ │ [Pause][Edit]│ │ [Pause][Edit]│ │ [Pause][Edit│ │           │
│ │ └──────────────┘ └──────────────┘ └─────────────┘ │           │
│ └────────────────────────────────────────────────────┘           │
├──────────────────────────────────────────────────────────────────┤
│ ALERT FEED (real-time, scrollable)                               │
│                                                                  │
│ 📉 14:32  NVDA  -3.42%  "NVDA dropped on export restriction..." │
│    [View Research] [View Simulation]                             │
│                                                                  │
│ 📈 11:15  AAPL  Vol 4.2×  "Apple volume surged ahead of..."    │
│    [View Research]                                               │
│                                                                  │
│ 📉 09:30  NVDA  -1.85%  "Continuation of yesterday's sell..."  │
│    [View Research] [View Simulation]                             │
└──────────────────────────────────────────────────────────────────┘
```

### 12.2 Deploy Agent Modal (Form)

```
┌────────────────────────────────────────────┐
│         Deploy New Tracker Agent           │
├────────────────────────────────────────────┤
│ Ticker:  [  NVDA  ▼ ]                     │
│ Name:    [ NVDA Crash Watch            ]  │
│                                            │
│ Alert Conditions:                          │
│ ☑ Price change ≥ [2.0]% in 5 min         │
│ ☑ Volume spike ≥ [3.0]× average          │
│ ☐ RSI above: [ 75 ]                      │
│ ☑ RSI below: [ 30 ]                      │
│ ☐ Price above: $[    ]                   │
│ ☐ Price below: $[    ]                   │
│                                            │
│ Options:                                   │
│ ☑ Auto-simulate on trigger                │
│                                            │
│ [  Cancel  ]          [ 🚀 Deploy Agent ] │
└────────────────────────────────────────────┘
```

### 12.3 Key Components

```
components/
  tracker/
    AgentCard.tsx          # Card showing agent status, ticker, last alert
    DeployAgentModal.tsx   # Form to create new agent
    EditAgentModal.tsx     # Edit triggers, pause/resume
    AlertFeed.tsx          # Real-time scrolling alert list
    AlertCard.tsx          # Single alert with narrative + action links
    TrackerHeader.tsx      # Page header with agent count + deploy button
```

### 12.4 Alert Card Actions

Each alert card has action buttons that link to the other modules:
- **"View Research"** → navigates to `/research/{symbol}` (Research module)
- **"View Simulation"** → navigates to `/simulation/{sim_id}` (if auto-sim was triggered)
- **"Deploy Similar Agent"** → opens the deploy form pre-filled with this ticker

### 12.5 Color Scheme

Same dark theme as other modules:
```
Active Agent:    #22c55e border (green-500)
Paused Agent:    #f59e0b border (amber-500)
Alert (Bearish): #ef4444 background tint
Alert (Bullish): #22c55e background tint
Poke sent badge: #8b5cf6 (violet — Poke's brand vibe)
```

---

## 13. Data Models & Schemas

### Pydantic Models (Backend)

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TriggerConfig(BaseModel):
    price_change_pct: Optional[float] = None    # e.g., 2.0 = fire on ±2%
    volume_spike_ratio: Optional[float] = None  # e.g., 3.0 = fire on 3× volume
    rsi_upper: Optional[float] = None           # e.g., 75
    rsi_lower: Optional[float] = None           # e.g., 25
    price_above: Optional[float] = None         # e.g., 200.0
    price_below: Optional[float] = None         # e.g., 150.0

class CreateAgentRequest(BaseModel):
    symbol: str
    name: str
    triggers: TriggerConfig
    auto_simulate: bool = False

class UpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None           # "active", "paused"
    triggers: Optional[TriggerConfig] = None
    auto_simulate: Optional[bool] = None

class TrackerAgentResponse(BaseModel):
    id: str
    symbol: str
    name: str
    status: str
    triggers: TriggerConfig
    auto_simulate: bool
    total_alerts: int
    last_alert_at: Optional[str]
    last_price: Optional[float]
    created_at: str

class AlertResponse(BaseModel):
    id: str
    agent_id: str
    symbol: str
    trigger_reason: str
    narrative: Optional[str]
    market_snapshot: dict
    poke_sent: bool
    simulation_id: Optional[str]
    created_at: str
```

---

## 14. Environment & Configuration

### Additional API Keys (beyond Research + Simulation)

Add to `.env`:
```env
# Poke (The Interaction Company)
# Generate at https://poke.com/settings/advanced
POKE_API_KEY=pk_xxxxxxxxxxxx

# Supabase
SUPABASE_URL=https://yvofwqjdxhzvtucaygih.supabase.co
SUPABASE_KEY=your_supabase_anon_key

# Backend URL (for Modal Cron to call back to)
BACKEND_URL=https://tickermaster-api.vercel.app  # or localhost for dev

# Modal (shared with Simulation)
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
```

### Modal Secret Setup

Create a Modal secret that bundles all keys for the Cron function:

```bash
modal secret create tickermaster-secrets \
  SUPABASE_URL="https://yvofwqjdxhzvtucaygih.supabase.co" \
  SUPABASE_KEY="your_key" \
  OPENROUTER_API_KEY="sk-or-xxx" \
  POKE_API_KEY="pk_xxx" \
  BACKEND_URL="https://tickermaster-api.vercel.app"
```

### Additional Python Dependencies

Add to `requirements.txt`:
```
supabase==2.9.0
```

---

## 15. File Structure

```
project/
├── backend/
│   ├── main.py                          # FastAPI app (shared — add tracker router)
│   ├── config.py                        # Environment variables (shared)
│   ├── routers/
│   │   ├── research.py                  # Research routes
│   │   ├── simulation.py               # Simulation routes + WebSocket
│   │   └── tracker.py                  # Tracker CRUD + WebSocket + emit endpoint
│   ├── services/
│   │   ├── ... (research + simulation services)
│   │   ├── database.py                 # Supabase client helper
│   │   ├── tracker_service.py          # CRUD operations for agents/alerts
│   │   ├── alert_evaluator.py          # Trigger evaluation logic (shared w/ Modal)
│   │   ├── alert_pipeline.py           # Investigate → Analyze → Notify pipeline
│   │   └── poke.py                     # Poke notification service
│   ├── models/
│   │   ├── schemas.py                  # Shared Pydantic models
│   │   ├── simulation_schemas.py       # Simulation models
│   │   └── tracker_schemas.py          # Tracker models
│   └── ...
│
├── tracker/
│   ├── modal_cron.py                   # Modal Cron function (deployed separately)
│   └── README.md                       # How to deploy the cron job
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Router (add /tracker route)
│   │   ├── pages/
│   │   │   ├── HomePage.tsx
│   │   │   ├── TickerPage.tsx          # Research
│   │   │   ├── SimulationPage.tsx      # Simulation
│   │   │   └── TrackerPage.tsx         # Tracker dashboard
│   │   ├── components/
│   │   │   ├── ... (research + simulation components)
│   │   │   ├── tracker/
│   │   │   │   ├── AgentCard.tsx             # Agent status card
│   │   │   │   ├── DeployAgentModal.tsx      # Create agent form
│   │   │   │   ├── EditAgentModal.tsx        # Edit agent form
│   │   │   │   ├── AlertFeed.tsx             # Real-time alert list
│   │   │   │   ├── AlertCard.tsx             # Single alert display
│   │   │   │   └── TrackerHeader.tsx         # Page header
│   │   │   └── common/
│   │   │       └── ... (shared components)
│   │   ├── hooks/
│   │   │   ├── useSimulationWebSocket.ts
│   │   │   └── useTrackerWebSocket.ts        # WebSocket hook for alerts
│   │   └── ...
│   └── ...
│
├── supabase/
│   └── migrations/
│       └── 001_tracker_tables.sql       # SQL for tracker tables
│
└── README.md
```

---

## 16. Implementation Order

### Phase 1: Database + CRUD (Foundation)
1. Create Supabase tables (`tracker_agents`, `tracker_alerts`)
2. Implement `database.py` (Supabase client)
3. Implement `tracker_service.py` (CRUD for agents and alerts)
4. Create tracker router with REST endpoints
5. Test: can create/read/update/delete agents via API

### Phase 2: Modal Cron Polling
6. Write `modal_cron.py` with `poll_all_agents` function
7. Implement `evaluate_triggers` with all trigger types
8. Deploy to Modal: `modal deploy tracker/modal_cron.py`
9. Test: create an agent, verify cron function polls it every minute

### Phase 3: Investigation Pipeline
10. Wire up Research module API calls inside the cron function
11. Implement `generate_alert_narrative` (OpenRouter LLM call)
12. Test: trigger an agent manually, verify it calls Research APIs and generates a narrative

### Phase 4: Poke Notifications
13. Implement `poke.py` (Poke webhook integration)
14. Test: send a test notification, verify it arrives on phone
15. Wire Poke into the alert pipeline
16. Test end-to-end: trigger → investigate → analyze → Poke notification

### Phase 5: Frontend
17. Build `TrackerPage.tsx` with agent grid + alert feed
18. Build `DeployAgentModal.tsx` form
19. Build `AlertCard.tsx` with action links to Research/Simulation
20. Wire up WebSocket for real-time alert updates

### Phase 6: Simulation Integration
21. Implement `trigger_simulation` in the alert pipeline
22. Add "auto-simulate" toggle to the deploy form
23. Test: trigger with auto_simulate=true, verify simulation launches

### Phase 7: Polish + Demo
24. Add agent pause/resume functionality
25. Add "last checked" heartbeat indicator on agent cards
26. Pre-deploy agents for demo tickers (NVDA, TSLA, AAPL)
27. Set trigger thresholds low enough to fire during demo (e.g., 0.5% price change)
28. Have a teammate send a test Poke notification to show the judges
29. Demo flow: show active agents → trigger fires → phone buzzes → tap link → Research page

---

## Links to All Tools & APIs (Tracker)

| Tool | URL | Notes |
|------|-----|-------|
| Poke API | `https://poke.com/api/v1/inbound-sms/webhook` | Notification delivery |
| Poke API Key | `https://poke.com/settings/advanced` | Generate API key here |
| Modal Cron | https://modal.com/docs/guide/cron | Scheduled polling |
| Modal Secrets | https://modal.com/docs/guide/secrets | Store API keys for cron |
| Supabase | https://supabase.com/docs | Database for agent state |
| OpenRouter | https://openrouter.ai/docs | LLM for narrative generation |
| yfinance | https://pypi.org/project/yfinance/ | Price/volume polling |

---

## Cross-Module Integration Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    TickerMaster Architecture                  │
│                                                              │
│  TRACKER ──calls──▶ RESEARCH ◀──shared data──▶ SIMULATION   │
│    │                   │                           │         │
│    │  "What's the      │  "Here's the full        │         │
│    │   sentiment on     │   analysis for NVDA"     │         │
│    │   NVDA right now?" │                          │         │
│    │                    │                          │         │
│    ├──auto-triggers────▶────────────────────────▶──┘         │
│    │  "Launch a sim     "Simulate how agents                 │
│    │   for this crash"   trade this scenario"                │
│    │                                                         │
│    └──notifies──▶ POKE ──▶ User's phone                     │
│                   "📉 NVDA dropped 3.4%..."                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Critical Notes for the Coding Agent

1. **The Modal Cron function is deployed SEPARATELY from the FastAPI backend.** Run `modal deploy tracker/modal_cron.py` independently. It communicates with the backend via HTTP calls, not Python imports.

2. **The cron function must be self-contained.** It runs in a Modal container — it can't import from `backend/services/`. Any shared logic (like `evaluate_triggers`) must either be duplicated in the cron file or factored into a shared package that's pip-installed.

3. **Use ONE cron function for all agents, not one per agent.** Loop over all active agents from Supabase. This is simpler and avoids the complexity of dynamically deploying Modal functions.

4. **15-minute cooldown between alerts is non-negotiable.** Nobody wants their phone blowing up every minute. Check `last_alert_at` before firing.

5. **The investigation step calls the Research module's REST API.** Don't re-implement Perplexity Sonar / Reddit / sentiment logic in the Tracker. Call `/api/ticker/{symbol}/ai-research` and `/api/ticker/{symbol}/sentiment`. This ensures consistency and avoids duplicating code.

6. **Poke API is dead simple.** One POST to `https://poke.com/api/v1/inbound-sms/webhook` with `{"message": "your text"}` and a Bearer token. If it fails, log it and continue — don't let a Poke failure crash the pipeline.

7. **Always include a deep link in the Poke notification.** The link back to the Research page (`/research/{symbol}`) is what connects the phone notification to the full dashboard experience. This is the "wow" for judges.

8. **For the demo, set trigger thresholds LOW.** During a hackathon demo, you need triggers to fire. Set price_change_pct to 0.3% or even 0.1% so it fires while judges are watching. You can always explain the real-world thresholds would be higher.

9. **Supabase is the shared database for all three modules.** Research can use it for caching, Simulation for persistence, Tracker for agent state. One database, one connection string.

10. **The auto-simulate feature is the showstopper integration.** Tracker detects crash → calls Research for context → launches Simulation with crash conditions → notifies user on phone with link to watch the sim. This hits Modal, Perplexity, OpenAI, and Poke in one flow.

11. **Agent Observability is mandatory.** Every price check, trigger evaluation, investigation, and notification MUST log to `agent_activity` table via `log_agent_activity()`. Even "no trigger" results should log so the observability feed shows the agent is alive and working. Example:
```python
await log_agent_activity(
    user_id=agent["user_id"], module="tracker",
    agent_name=f"Tracker/{agent['symbol']}",
    action=f"Price check: ${snapshot['price']} — no trigger",
    details=snapshot, status="success"
)
```

12. **Poke conversational commands create tracker agents.** When a user texts Poke "track AMZN with 3% alert", the Poke inbound webhook (`/api/poke/inbound`) parses this and creates a tracker agent via the CRUD API. See the Master Build Spec Section 9 for full Poke MCP implementation.

13. **X/Twitter sentiment is an additional investigation source.** The alert pipeline now also fetches X/Twitter sentiment via `/api/ticker/{symbol}/x-sentiment` to enrich the narrative. The narrative LLM prompt should include X sentiment alongside Reddit and prediction market data.

14. **Kalshi and Polymarket data are public — no auth needed for reads.** Kalshi: `https://api.elections.kalshi.com/trade-api/v2/markets`. Polymarket: `https://gamma-api.polymarket.com/markets` and `https://clob.polymarket.com`. Both return JSON directly.

15. **User authentication is required.** All tracker agents belong to a `user_id` (from Supabase Auth). The Modal Cron function uses the Supabase service role key to read all active agents across all users. RLS policies protect per-user data on the frontend.
