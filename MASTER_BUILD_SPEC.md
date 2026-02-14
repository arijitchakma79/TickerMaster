# TickerMaster — Master Build Specification

> **This is the primary orchestration document.** The coding agent reads this FIRST, follows the build order, and references the 3 sub-module specs (Research, Simulation, Tracker) for detailed implementation. Every architectural decision is final. Execute, don't deliberate.

---

## Table of Contents

1. [Product Vision & Demo Flow](#1-product-vision--demo-flow)
2. [Tech Stack & Architecture](#2-tech-stack--architecture)
3. [Environment Setup](#3-environment-setup)
4. [Supabase Schema & Auth](#4-supabase-schema--auth)
5. [Backend — FastAPI Core](#5-backend--fastapi-core)
6. [Frontend — React/TypeScript](#6-frontend--reacttypescript)
7. [Landing Page & Onboarding Tutorial](#7-landing-page--onboarding-tutorial)
8. [Agent Observability System](#8-agent-observability-system)
9. [Poke MCP Integration — Conversational Interface](#9-poke-mcp-integration--conversational-interface)
10. [Browserbase / Stagehand — Deep Research Scraping](#10-browserbase--stagehand--deep-research-scraping)
11. [New Data Sources — X, Kalshi, Polymarket](#11-new-data-sources--x-kalshi-polymarket)
12. [Cross-Module Integration Map](#12-cross-module-integration-map)
13. [Complete File Structure](#13-complete-file-structure)
14. [Build Order (Phases 1-10)](#14-build-order-phases-1-10)
15. [Testing & Debugging Protocol](#15-testing--debugging-protocol)
16. [Deployment (Vercel + Modal)](#16-deployment-vercel--modal)
17. [Prize Alignment Matrix](#17-prize-alignment-matrix)

---

## 1. Product Vision & Demo Flow

### One-Liner

**TickerMaster** — An AI-powered financial command center where persistent agents research, simulate, and track the market for you, then text you when it matters.

### Three Pillars

| Module         | What It Does                                                       | Killer Feature                                                                 |
| -------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **Research**   | Yahoo Finance meets AI analyst. Enter a ticker, get everything.    | Perplexity Sonar AI summaries + Browserbase deep scraping + prediction markets |
| **Simulation** | Real-time arena where AI trading agents compete in Modal Sandboxes | Watch a quant bot crush a retail YOLO trader, with AI commentary               |
| **Tracker**    | Deploy persistent agents that monitor tickers 24/7 via Modal Cron  | Phone buzzes via Poke: "NVDA down 3% — here's why and what to do"              |

### Demo Script (3 Minutes for Judges)

```
0:00 — Landing page. "TickerMaster: Your AI Trading Command Center"
       Click "Sign in with Google" → Supabase Auth → Dashboard

0:15 — RESEARCH: Type "NVDA" in search bar
       → Instant price/chart loads
       → AI Research panel streams Perplexity summary
       → Sentiment gauge shows Reddit/X mood
       → Prediction markets show Kalshi/Polymarket odds
       → "Deep Research" button → Browserbase scrapes Finviz/Morningstar live
       → Show agent activity panel: "Browserbase agent navigating finviz.com..."

0:45 — SIMULATION: Click "Simulate" tab
       → Type: "Buy when RSI < 30, sell when RSI > 70"
       → Click "Launch Arena" → Modal Sandbox spins up
       → Real-time chart appears, 4 agents trading
       → Quant bot profits while Retail bot panic-sells
       → AI commentator: "The Quant just bought the dip 😂"
       → Show agent thinking panel: reasoning chains visible

1:30 — TRACKER: Click "Tracker" tab
       → Show 3 deployed agents (NVDA, TSLA, AAPL)
       → Green "active" indicators pulsing
       → Click "Deploy Agent" → form for AMZN with price ±2% trigger
       → Show alert feed with past notifications
       → Show phone receiving Poke iMessage: "📉 NVDA Alert..."
       → Show "Reply to Poke: 'simulate NVDA crash'" → triggers sim from text

2:15 — OBSERVABILITY: Click "Agents" in nav
       → Activity feed showing all agent actions across modules
       → "Tracker/NVDA: checked price $875.30 — no trigger"
       → "Research/NVDA: Perplexity fetch complete — bearish sentiment"
       → "Simulation/Arena-7: Quant bot executed BUY 50 shares at $873"
       → Live WebSocket-powered, auto-scrolling

2:45 — Wrap: "TickerMaster uses Modal Sandboxes for simulation,
       Modal Cron for tracking, Perplexity Sonar for research,
       Browserbase for deep scraping, Poke for phone alerts,
       and OpenAI for agent intelligence. It's production-ready."
```

---

## 2. Tech Stack & Architecture

### Master Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Vercel)                            │
│            React 18 + TypeScript + TailwindCSS + Shadcn/UI          │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌─────────────────────┐  │
│  │ Landing  │ │  Research   │ │Simulation│ │     Tracker         │  │
│  │ Page     │ │  Dashboard  │ │  Arena   │ │     Dashboard       │  │
│  └──────────┘ └────────────┘ └──────────┘ └─────────────────────┘  │
│                    ┌─────────────────────┐                          │
│                    │ Agent Observability  │                          │
│                    │ Panel (all modules)  │                          │
│                    └─────────────────────┘                          │
└────────────────────────┬────────────────────────────────────────────┘
                         │ REST + WebSocket
┌────────────────────────▼────────────────────────────────────────────┐
│                     BACKEND (FastAPI on Vercel/Railway)              │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Auth     │ │  Research   │ │Simulation│ │ Tracker  │            │
│  │ Router   │ │  Router     │ │  Router  │ │ Router   │            │
│  └──────────┘ └────────────┘ └──────────┘ └──────────┘            │
│  ┌──────────────────────────────────────────────────────┐          │
│  │              Shared Services Layer                    │          │
│  │  Supabase Client │ Poke Service │ Agent Logger       │          │
│  └──────────────────────────────────────────────────────┘          │
└──────┬──────────┬──────────────┬──────────────┬─────────────────────┘
       │          │              │              │
  ┌────▼────┐ ┌──▼───────┐ ┌───▼──────┐ ┌────▼────────┐
  │Supabase │ │Perplexity│ │  Modal   │ │Browserbase  │
  │ Auth+DB │ │  Sonar   │ │Sandbox+  │ │ Stagehand   │
  │         │ │          │ │  Cron    │ │             │
  └─────────┘ └──────────┘ └──────────┘ └─────────────┘
       │
  ┌────▼────────────────────────────────────┐
  │          External Data APIs              │
  │  yfinance │ Reddit │ X/Twitter │ FRED   │
  │  Kalshi   │ Polymarket │ Finviz(BB)     │
  └─────────────────────────────────────────┘
       │
  ┌────▼────┐
  │  Poke   │ ──▶ User's Phone (iMessage/SMS)
  │ Webhook │
  └─────────┘
```

### Full Tech Stack

| Layer                  | Technology                                   | Version | Why                                         |
| ---------------------- | -------------------------------------------- | ------- | ------------------------------------------- |
| **Frontend**           | React 18 + TypeScript                        | 18.3+   | Industry standard, fast                     |
| **Styling**            | TailwindCSS + Shadcn/UI                      | 3.4+    | Professional design system, rapid iteration |
| **Charts**             | Lightweight-charts (TradingView) + Recharts  | 4.2+    | Real-time financial charts                  |
| **Backend**            | FastAPI (Python)                             | 0.115+  | Async, WebSocket native, fast               |
| **Auth**               | Supabase Auth (Google OAuth + email/pass)    | —       | Zero-config, built-in RLS                   |
| **Database**           | Supabase (PostgreSQL)                        | —       | Shared state, real-time subscriptions       |
| **LLM**                | OpenAI GPT-4o-mini via OpenRouter            | —       | Cost-efficient, fast, structured output     |
| **AI Research**        | Perplexity Sonar API                         | —       | Cited web research, real-time               |
| **Deep Scraping**      | Browserbase + Stagehand (Python)             | —       | AI-powered browser automation               |
| **Simulation Runtime** | Modal Sandbox                                | —       | Isolated containers, stdout streaming       |
| **Scheduled Polling**  | Modal Cron                                   | —       | Serverless recurring jobs                   |
| **Notifications**      | Poke Webhook API                             | —       | iMessage/SMS push notifications             |
| **Social Data**        | Reddit API, X/Twitter API                    | —       | Sentiment signals                           |
| **Prediction Markets** | Kalshi API, Polymarket CLOB (public)         | —       | Implied probability signals                 |
| **Macro Data**         | FRED API                                     | —       | Economic indicators                         |
| **Deployment**         | Vercel (frontend) + Railway/Render (backend) | —       | Easy, fast, free tier                       |

---

## 3. Environment Setup

### .env File (Root of Project)

```env
# ============================================
# TickerMaster Environment Variables
# ============================================

# --- Supabase ---
SUPABASE_URL=https://yvofwqjdxhzvtucaygih.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.REPLACE_WITH_REAL_ANON_KEY
SUPABASE_SERVICE_KEY=REPLACE_WITH_SERVICE_ROLE_KEY
DATABASE_URL=postgresql://postgres:treehacks2026@db.yvofwqjdxhzvtucaygih.supabase.co:5432/postgres

# --- LLM ---
OPENAI_API_KEY=sk-proj-REPLACE
OPENROUTER_API_KEY=sk-or-v1-REPLACE

# --- Research APIs ---
PERPLEXITY_API_KEY=pplx-REPLACE
FRED_API_KEY=Cf7792e7fe29de8f3c34ee64223f8b7d

# --- Social APIs ---
REDDIT_CLIENT_ID=REPLACE
REDDIT_CLIENT_SECRET=REPLACE

X_BEARER_TOKEN=REPLACE
X_CONSUMER_KEY=5bEp3OylpNy6rkkisohLwPdMg
X_CONSUMER_SECRET=REPLACE
X_ACCESS_TOKEN=REPLACE
X_ACCESS_TOKEN_SECRET=REPLACE

# --- Prediction Markets ---
KALSHI_API_KEY=ff303393-25ed-4586-b0bf-e3435d2ae264
# Polymarket: No key needed — public CLOB endpoint
POLYMARKET_CLOB_URL=https://clob.polymarket.com

# --- Browserbase (Stagehand) ---
BROWSERBASE_API_KEY=bb_live_REPLACE
BROWSERBASE_PROJECT_ID=09785f57-278e-40c8-a9ba-0b6fa7e6692a

# --- Modal ---
MODAL_TOKEN_ID=ak-REPLACE
MODAL_TOKEN_SECRET=as-REPLACE

# --- Poke (Notification) ---
POKE_API_KEY=pk_REPLACE

# --- Frontend (prefix with NEXT_PUBLIC_ for client-side) ---
NEXT_PUBLIC_SUPABASE_URL=https://yvofwqjdxhzvtucaygih.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.REPLACE
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### .gitignore

```gitignore
# Dependencies
node_modules/
__pycache__/
*.pyc
.venv/
venv/
env/

# Environment
.env
.env.local
.env.production
.env.*.local

# Build
.next/
out/
dist/
build/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*

# Modal
.modal/

# Supabase local
supabase/.temp/

# Misc
coverage/
.vercel
```

---

## 4. Supabase Schema & Auth

### 4.1 Authentication Setup

We use **Supabase Auth** with two providers:

1. **Google OAuth** (primary — best UX)
2. **Email/Password** (fallback)

#### Setup Steps (Manual in Supabase Dashboard)

1. Go to **Authentication → Providers → Google**
2. Enable Google provider
3. Add your Google OAuth Client ID and Secret (from Google Cloud Console)
4. The callback URL is: `https://yvofwqjdxhzvtucaygih.supabase.co/auth/v1/callback`
5. In Google Cloud Console, add authorized redirect URI: `https://yvofwqjdxhzvtucaygih.supabase.co/auth/v1/callback`
6. Add `http://localhost:3000` as authorized JavaScript origin for dev

#### If Google OAuth is not set up yet (hackathon shortcut)

Use **email/password auth** with Supabase's built-in magic link support. This works out of the box with zero config. The coding agent should implement BOTH auth methods — Google button + email/password form — so whichever is configured first works.

### 4.2 Database Schema

```sql
-- ============================================
-- TickerMaster Complete Database Schema
-- Run in Supabase SQL Editor
-- ============================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. USER PROFILES (extends Supabase auth.users)
-- ============================================
CREATE TABLE public.profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email TEXT,
    display_name TEXT,
    avatar_url TEXT,
    poke_enabled BOOLEAN DEFAULT FALSE,
    tutorial_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Auto-create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, display_name, avatar_url)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.email),
        NEW.raw_user_meta_data->>'avatar_url'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================
-- 2. TRACKER AGENTS
-- ============================================
CREATE TABLE public.tracker_agents (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'deleted')),
    triggers JSONB NOT NULL DEFAULT '{}',
    auto_simulate BOOLEAN DEFAULT FALSE,
    last_alert_at TIMESTAMPTZ,
    total_alerts INTEGER DEFAULT 0,
    last_checked_at TIMESTAMPTZ,
    last_price NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_tracker_agents_user ON public.tracker_agents (user_id);
CREATE INDEX idx_tracker_agents_active ON public.tracker_agents (status) WHERE status = 'active';
CREATE INDEX idx_tracker_agents_symbol ON public.tracker_agents (symbol);

-- ============================================
-- 3. TRACKER ALERTS
-- ============================================
CREATE TABLE public.tracker_alerts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    agent_id UUID REFERENCES public.tracker_agents(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    narrative TEXT,
    market_snapshot JSONB,
    investigation_data JSONB,
    poke_sent BOOLEAN DEFAULT FALSE,
    simulation_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_tracker_alerts_agent ON public.tracker_alerts (agent_id);
CREATE INDEX idx_tracker_alerts_user ON public.tracker_alerts (user_id);
CREATE INDEX idx_tracker_alerts_created ON public.tracker_alerts (created_at DESC);

-- ============================================
-- 4. SIMULATION HISTORY
-- ============================================
CREATE TABLE public.simulations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    config JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    results JSONB,
    modal_sandbox_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_simulations_user ON public.simulations (user_id);

-- ============================================
-- 5. AGENT ACTIVITY LOG (Observability)
-- ============================================
CREATE TABLE public.agent_activity (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    module TEXT NOT NULL CHECK (module IN ('research', 'simulation', 'tracker')),
    agent_name TEXT NOT NULL,
    action TEXT NOT NULL,
    details JSONB,
    status TEXT DEFAULT 'success' CHECK (status IN ('success', 'error', 'pending', 'running')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_activity_user ON public.agent_activity (user_id);
CREATE INDEX idx_agent_activity_created ON public.agent_activity (created_at DESC);
CREATE INDEX idx_agent_activity_module ON public.agent_activity (module);

-- ============================================
-- 6. RESEARCH CACHE
-- ============================================
CREATE TABLE public.research_cache (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol TEXT NOT NULL,
    data_type TEXT NOT NULL,
    data JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_research_cache_lookup ON public.research_cache (symbol, data_type);

-- ============================================
-- 7. USER WATCHLIST
-- ============================================
CREATE TABLE public.watchlist (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    symbol TEXT NOT NULL,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, symbol)
);

-- ============================================
-- ROW LEVEL SECURITY
-- ============================================
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tracker_agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tracker_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_activity ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.watchlist ENABLE ROW LEVEL SECURITY;

-- Profiles: users can read/update their own profile
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

-- Tracker agents: users manage their own agents
CREATE POLICY "Users can CRUD own tracker agents" ON public.tracker_agents
    FOR ALL USING (auth.uid() = user_id);

-- Tracker alerts: users see their own alerts
CREATE POLICY "Users can view own alerts" ON public.tracker_alerts
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Service can insert alerts" ON public.tracker_alerts
    FOR INSERT WITH CHECK (true);  -- Service role inserts from Modal Cron

-- Simulations: users manage their own sims
CREATE POLICY "Users can CRUD own simulations" ON public.simulations
    FOR ALL USING (auth.uid() = user_id);

-- Agent activity: users see their own activity
CREATE POLICY "Users can view own activity" ON public.agent_activity
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Service can insert activity" ON public.agent_activity
    FOR INSERT WITH CHECK (true);

-- Watchlist: users manage their own watchlist
CREATE POLICY "Users can CRUD own watchlist" ON public.watchlist
    FOR ALL USING (auth.uid() = user_id);

-- Research cache: public read, service write
ALTER TABLE public.research_cache ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can read cache" ON public.research_cache
    FOR SELECT USING (true);
CREATE POLICY "Service can write cache" ON public.research_cache
    FOR INSERT WITH CHECK (true);
CREATE POLICY "Service can update cache" ON public.research_cache
    FOR UPDATE USING (true);
```

---

## 5. Backend — FastAPI Core

### 5.1 Main Application Entry Point

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routers import research, simulation, tracker, auth, agents
from services.database import init_supabase

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_supabase()
    yield
    # Shutdown

app = FastAPI(
    title="TickerMaster API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://tickermaster.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routers
app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(research.router, prefix="/api", tags=["Research"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])
app.include_router(tracker.router, prefix="/api/tracker", tags=["Tracker"])
app.include_router(agents.router, prefix="/api/agents", tags=["Agent Observability"])

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "tickermaster"}
```

### 5.2 Shared Services

```python
# backend/services/database.py
from supabase import create_client, Client
import os

_client: Client | None = None

def init_supabase():
    global _client
    _client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_SERVICE_KEY", os.environ["SUPABASE_KEY"]),
    )

def get_supabase() -> Client:
    if _client is None:
        init_supabase()
    return _client
```

```python
# backend/services/agent_logger.py
"""
Central agent activity logger.
Every agent action across all modules logs here for observability.
"""
from services.database import get_supabase
from datetime import datetime

async def log_agent_activity(
    user_id: str,
    module: str,          # "research", "simulation", "tracker"
    agent_name: str,      # e.g., "Perplexity Sonar", "Quant Bot", "NVDA Tracker"
    action: str,          # e.g., "Fetching sentiment", "Executed BUY", "Price check"
    details: dict = None,
    status: str = "success",
):
    """Log an agent action to the observability table."""
    try:
        get_supabase().table("agent_activity").insert({
            "user_id": user_id,
            "module": module,
            "agent_name": agent_name,
            "action": action,
            "details": details or {},
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        print(f"[AgentLogger] Failed to log: {e}")
```

```python
# backend/services/poke.py
"""
Poke notification service.
Sends messages via Poke's webhook API to user's iMessage/SMS.
"""
import httpx
import os

POKE_ENDPOINT = "https://poke.com/api/v1/inbound-sms/webhook"

async def send_poke_message(message: str) -> bool:
    """Send a message to the user via Poke (iMessage/SMS/WhatsApp)."""
    api_key = os.environ.get("POKE_API_KEY")
    if not api_key:
        print("[Poke] No API key configured")
        return False

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                POKE_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"message": message},
                timeout=10.0,
            )
            return response.status_code == 200
    except Exception as e:
        print(f"[Poke] Send failed: {e}")
        return False
```

### 5.3 Auth Router

```python
# backend/routers/auth.py
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from services.database import get_supabase

router = APIRouter()

class TokenVerifyRequest(BaseModel):
    access_token: str

@router.post("/verify")
async def verify_token(req: TokenVerifyRequest):
    """Verify a Supabase JWT and return user info."""
    try:
        supabase = get_supabase()
        user = supabase.auth.get_user(req.access_token)
        return {
            "user_id": user.user.id,
            "email": user.user.email,
            "name": user.user.user_metadata.get("full_name", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Get user profile from profiles table."""
    result = get_supabase().table("profiles").select("*").eq("id", user_id).single().execute()
    return result.data
```

---

## 6. Frontend — React/TypeScript

### 6.1 Core Setup

The frontend uses **Next.js 14+ (App Router)** with:

- **TailwindCSS** for styling
- **Shadcn/UI** for professional component library
- **@supabase/ssr** for server-side auth
- **Framer Motion** for smooth animations
- **lightweight-charts** for financial charts
- **Recharts** for data visualizations

### 6.2 Supabase Client Setup

```typescript
// lib/supabase/client.ts
import { createBrowserClient } from "@supabase/ssr";

export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  );
}
```

```typescript
// lib/supabase/server.ts
import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

export async function createClient() {
  const cookieStore = await cookies();
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value, options }) =>
            cookieStore.set(name, value, options),
          );
        },
      },
    },
  );
}
```

```typescript
// middleware.ts
import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";

export async function middleware(request: NextRequest) {
  let supabaseResponse = NextResponse.next({ request });

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value),
          );
          supabaseResponse = NextResponse.next({ request });
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options),
          );
        },
      },
    },
  );

  const {
    data: { user },
  } = await supabase.auth.getUser();

  // Protect dashboard routes
  const protectedPaths = ["/research", "/simulation", "/tracker", "/agents"];
  const isProtected = protectedPaths.some((path) =>
    request.nextUrl.pathname.startsWith(path),
  );

  if (isProtected && !user) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  return supabaseResponse;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico|api|auth).*)"],
};
```

### 6.3 App Layout & Navigation

```typescript
// app/layout.tsx — Root layout with persistent nav
// Design: Dark theme (slate-900 bg), glass-morphism nav bar
// Nav items: Logo | Research | Simulation | Tracker | Agents | [User Avatar]
// Active tab: cyan-400 underline + glow effect
// Mobile: hamburger → slide-out drawer

// Color system (consistent across ALL modules):
const colors = {
  bg: "#0f172a", // slate-900
  surface: "#1e293b", // slate-800
  surfaceHover: "#334155", // slate-700
  border: "#475569", // slate-600
  text: "#f8fafc", // slate-50
  textMuted: "#94a3b8", // slate-400
  accent: "#06b6d4", // cyan-500  (primary action color)
  green: "#22c55e", // green-500 (bullish/buy/active)
  red: "#ef4444", // red-500   (bearish/sell/error)
  amber: "#f59e0b", // amber-500 (hold/warning/paused)
  violet: "#8b5cf6", // violet-500 (quant agent / Poke)
  blue: "#3b82f6", // blue-500  (value agent / info)
  orange: "#f97316", // orange-500 (retail agent)
};
```

### 6.4 Page Structure

```
app/
├── layout.tsx              # Root layout: dark theme, nav bar, auth provider
├── page.tsx                # Landing page (public)
├── login/
│   └── page.tsx            # Auth page (Google + email/password)
├── auth/
│   └── callback/
│       └── route.ts        # OAuth callback handler
├── research/
│   ├── page.tsx            # Main research dashboard (search bar → ticker)
│   └── [symbol]/
│       └── page.tsx        # Individual ticker research page
├── simulation/
│   ├── page.tsx            # Simulation arena (setup + live view)
│   └── [id]/
│       └── page.tsx        # Specific simulation run
├── tracker/
│   └── page.tsx            # Tracker dashboard (agents + alerts)
├── agents/
│   └── page.tsx            # Agent observability / activity feed
└── tutorial/
    └── page.tsx            # Interactive onboarding tutorial
```

---

## 7. Landing Page & Onboarding Tutorial

### 7.1 Landing Page (`/`)

A **gorgeous, conversion-focused** landing page. Not a bland "Sign Up" form. Think Linear.app or Vercel's landing page — dark theme, subtle animations, clear value prop.

**Sections:**

1. **Hero**: Full-width dark gradient. Headline: "Your AI Trading Command Center". Subhead: "Research stocks, simulate strategies, deploy persistent agents — all powered by AI." CTA: "Get Started Free" button (glow animation).
2. **Three Pillars**: Cards for Research, Simulation, Tracker with animated icons and one-line descriptions.
3. **Live Demo Preview**: Embedded screenshot/animation showing the dashboard in action. Or a short looping video clip.
4. **Tech Stack Badges**: Small logos/badges showing integrations: "Powered by Perplexity, Modal, OpenAI, Poke"
5. **Footer CTA**: "Start researching in 30 seconds" → Sign in button

**Design Details:**

- Background: subtle animated gradient mesh (CSS, not a library)
- Hero text: gradient text (cyan → violet)
- Cards: glass-morphism (backdrop-blur, subtle border)
- Animations: Framer Motion fade-in-up on scroll
- Responsive: mobile-first

### 7.2 Onboarding Tutorial (`/tutorial`)

After first login, if `profile.tutorial_completed === false`, redirect to `/tutorial`.

**Interactive walkthrough (3-step stepper):**

**Step 1 — Research**: "Enter any stock ticker to get instant AI-powered research. Try typing NVDA." → Shows mini search bar that auto-navigates to /research/NVDA on submit.

**Step 2 — Simulation**: "Watch AI agents trade against each other. Hit 'Launch Arena' to start a simulation." → Shows preview of simulation with a play button.

**Step 3 — Tracker**: "Deploy an agent to watch any stock 24/7. It'll text you when something happens." → Shows mini agent deploy form.

**Completion**: "You're all set! 🎉" → Button "Go to Dashboard" → sets `tutorial_completed = true` in profiles table → redirect to /research.

---

## 8. Agent Observability System

### What This Is

A **real-time activity feed** showing what every AI agent across all three modules is doing, right now. This is the "show, don't tell" of the agentic experience. Users see:

- "🔍 Perplexity Agent: Fetching real-time analysis for NVDA..."
- "🤖 Quant Bot: Executed BUY 50 shares at $873.20 (RSI: 28.4)"
- "📡 Tracker/NVDA: Price check — $875.30, no trigger (5m change: +0.12%)"
- "🌐 Browserbase Agent: Navigating finviz.com/quote.ashx?t=NVDA..."
- "📱 Poke: Alert sent to user — NVDA dropped 3.4%"

### Frontend Component

```
┌──────────────────────────────────────────────────────────────────┐
│  🤖 Agent Activity                              [Filter ▼] [⟳] │
├──────────────────────────────────────────────────────────────────┤
│  ● LIVE                                                         │
│                                                                  │
│  14:32:05  🔍 Research/Perplexity   NVDA analysis complete      │
│            ├─ "Bearish sentiment on export restrictions"          │
│            └─ 3 sources cited | 1.2s                            │
│                                                                  │
│  14:32:03  🤖 Simulation/Quant     BUY 50 @ $873.20            │
│            ├─ Reasoning: "RSI oversold at 28.4, SMA crossover"   │
│            └─ Portfolio: $97,340 | +2.1% return                 │
│                                                                  │
│  14:32:01  📡 Tracker/NVDA-Watch   Price check: $875.30        │
│            ├─ 5m change: +0.12% | Volume: 1.2× avg             │
│            └─ Status: No trigger (threshold: ±2%)               │
│                                                                  │
│  14:31:58  🌐 Research/Browserbase  Scraping finviz.com...      │
│            ├─ Agent navigating analyst ratings page               │
│            └─ Status: Running... (est. 5s)                      │
│                                                                  │
│  14:31:45  📱 Tracker/Poke         Alert delivered              │
│            ├─ "NVDA dropped 3.4% — export restrictions"         │
│            └─ Delivery: iMessage ✓                              │
└──────────────────────────────────────────────────────────────────┘
```

### Backend: WebSocket Event Stream

```python
# backend/routers/agents.py
from fastapi import APIRouter, WebSocket
from services.database import get_supabase

router = APIRouter()

# In-memory list of connected WebSocket clients
_ws_clients: list[WebSocket] = []

@router.websocket("/ws")
async def agent_activity_ws(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except:
        _ws_clients.remove(websocket)

async def broadcast_activity(event: dict):
    """Broadcast an agent activity event to all connected clients."""
    for ws in _ws_clients.copy():
        try:
            await ws.send_json(event)
        except:
            _ws_clients.remove(ws)

@router.get("/activity")
async def get_recent_activity(user_id: str, limit: int = 50):
    """Get recent agent activity from database."""
    result = get_supabase().table("agent_activity") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    return result.data
```

### Integration Points

Every service that performs an agent action calls `log_agent_activity()` AND `broadcast_activity()`:

- Research: Perplexity fetch, Browserbase scrape, sentiment analysis
- Simulation: agent decisions, trades, commentary
- Tracker: price checks, trigger evaluations, Poke notifications

---

## 9. Poke MCP Integration — Conversational Interface

### What This Is

Beyond simple notifications, Poke becomes a **conversational interface** for TickerMaster. Users can text Poke to:

- "research NVDA" → triggers Research module, sends back a summary
- "simulate crash for TSLA" → launches a simulation, sends back a link
- "track AMZN with 3% alert" → deploys a tracker agent
- "what are my agents doing?" → returns current agent status

### How It Works

Poke receives user texts via its platform and can forward them to our backend via webhook. We expose an endpoint that Poke calls when the user texts a command.

```python
# backend/routers/poke_inbound.py
from fastapi import APIRouter, Request
from services.poke import send_poke_message
from services.agent_logger import log_agent_activity
import httpx
import os

router = APIRouter()

@router.post("/api/poke/inbound")
async def handle_poke_inbound(request: Request):
    """
    Handle inbound messages from the user via Poke.
    Poke forwards the user's text message to this webhook.
    We parse the intent and execute the appropriate action.
    """
    body = await request.json()
    user_message = body.get("message", "").strip().lower()

    # Parse intent using simple keyword matching (fast, no LLM needed)
    if user_message.startswith("research "):
        symbol = user_message.replace("research ", "").upper().strip()
        return await handle_research_command(symbol)

    elif user_message.startswith("simulate "):
        query = user_message.replace("simulate ", "").strip()
        return await handle_simulate_command(query)

    elif user_message.startswith("track "):
        query = user_message.replace("track ", "").strip()
        return await handle_track_command(query)

    elif "status" in user_message or "agents" in user_message:
        return await handle_status_command()

    else:
        # Use LLM to interpret freeform message
        return await handle_freeform_command(user_message)

async def handle_research_command(symbol: str):
    """Fetch quick research and send back via Poke."""
    api_url = os.environ.get("BACKEND_URL", "http://localhost:8000")

    async with httpx.AsyncClient(timeout=20.0) as client:
        res = await client.get(f"{api_url}/api/ticker/{symbol}/ai-research")
        if res.status_code == 200:
            data = res.json()
            summary = data.get("summary", "No data available")[:500]
            await send_poke_message(
                f"📊 Research: ${symbol}\n\n{summary}\n\n"
                f"🔗 Full analysis: https://tickermaster.vercel.app/research/{symbol}"
            )
            return {"status": "sent"}

    await send_poke_message(f"❌ Couldn't fetch research for ${symbol}. Try again?")
    return {"status": "error"}

async def handle_simulate_command(query: str):
    """Parse a simulation request and launch it."""
    # Use OpenRouter to extract intent
    # "crash for TSLA" → {symbol: "TSLA", scenario: "crash"}
    await send_poke_message(
        f"🎮 Launching simulation for: {query}\n"
        f"Watch live: https://tickermaster.vercel.app/simulation"
    )
    # Trigger simulation creation via internal API
    return {"status": "launched"}

async def handle_track_command(query: str):
    """Deploy a tracker agent from text."""
    await send_poke_message(
        f"📡 Deploying tracker agent for: {query}\n"
        f"Manage agents: https://tickermaster.vercel.app/tracker"
    )
    return {"status": "deployed"}

async def handle_status_command():
    """Report current agent status."""
    # Query active agents from Supabase
    await send_poke_message(
        "🤖 Your agents:\n"
        "• NVDA Tracker: Active (last check 30s ago)\n"
        "• TSLA Tracker: Active (0 alerts today)\n"
        "• Simulation #7: Completed (+4.2% return)\n"
    )
    return {"status": "sent"}
```

---

## 10. Browserbase / Stagehand — Deep Research Scraping

### What This Is

For data that isn't available via simple APIs (analyst ratings, institutional research, detailed financial statements), we use **Browserbase + Stagehand** to run AI-powered browser agents that navigate real financial websites.

### Implementation

```python
# backend/services/browserbase_scraper.py
"""
Browserbase/Stagehand integration for deep financial research.
AI-powered browser agents that scrape analyst ratings, financial data,
and institutional research from sites like Finviz, Morningstar, and Reuters.
"""
from stagehand import StagehandConfig, Stagehand
import os
import json

async def scrape_finviz_analysis(symbol: str) -> dict:
    """
    Use Stagehand to scrape detailed analyst data from Finviz.
    Returns: analyst ratings, price targets, insider trading, news headlines
    """
    config = StagehandConfig(
        env="BROWSERBASE",
        api_key=os.environ["BROWSERBASE_API_KEY"],
        project_id=os.environ["BROWSERBASE_PROJECT_ID"],
        model_name="gpt-4o-mini",
        model_client_options={"apiKey": os.environ["OPENAI_API_KEY"]},
        headless=True,
    )

    async with Stagehand(config) as stagehand:
        page = stagehand.page

        # Navigate to Finviz stock page
        await page.goto(f"https://finviz.com/quote.ashx?t={symbol}")

        # Extract analyst ratings
        ratings = await page.extract(
            "Extract all analyst ratings including: "
            "firm name, rating (buy/hold/sell), price target, date. "
            "Return as JSON array."
        )

        # Extract key financial metrics
        metrics = await page.extract(
            "Extract the key financial metrics table including: "
            "P/E, EPS, Market Cap, Dividend, Beta, 52W High, 52W Low, "
            "RSI, SMA 20, SMA 50, SMA 200, Short Float, Target Price. "
            "Return as JSON object."
        )

        # Extract recent insider trading
        insider = await page.extract(
            "Extract recent insider trading activity including: "
            "name, title, trade type (buy/sell), shares, value, date. "
            "Return as JSON array. Max 5 entries."
        )

        # Extract recent news headlines
        news = await page.extract(
            "Extract the recent news headlines and their timestamps. "
            "Return as JSON array with fields: headline, timestamp, source."
        )

        return {
            "symbol": symbol,
            "analyst_ratings": ratings,
            "financial_metrics": metrics,
            "insider_trading": insider,
            "news_headlines": news,
            "source": "finviz.com",
        }

async def scrape_reddit_deep_analysis(symbol: str) -> dict:
    """
    Use Stagehand to scrape in-depth Reddit analysis threads.
    Goes beyond the API — finds DD posts, reads comments, extracts bull/bear cases.
    """
    config = StagehandConfig(
        env="BROWSERBASE",
        api_key=os.environ["BROWSERBASE_API_KEY"],
        project_id=os.environ["BROWSERBASE_PROJECT_ID"],
        model_name="gpt-4o-mini",
        model_client_options={"apiKey": os.environ["OPENAI_API_KEY"]},
        headless=True,
    )

    async with Stagehand(config) as stagehand:
        page = stagehand.page

        # Search for DD posts on r/wallstreetbets and r/stocks
        await page.goto(
            f"https://www.reddit.com/r/wallstreetbets+stocks+investing/search/"
            f"?q={symbol}+DD&sort=new&t=week"
        )

        # Extract top analysis posts
        posts = await page.extract(
            f"Extract the top 5 analysis/DD posts about {symbol}. "
            "For each post, extract: title, author, upvotes, "
            "comment_count, a 2-sentence summary of the thesis, "
            "and whether it's bullish or bearish. Return as JSON array."
        )

        return {
            "symbol": symbol,
            "reddit_analysis": posts,
            "source": "reddit.com (r/wallstreetbets, r/stocks, r/investing)",
        }
```

### Endpoint

```python
# In backend/routers/research.py
@router.get("/ticker/{symbol}/deep-research")
async def get_deep_research(symbol: str, background_tasks: BackgroundTasks):
    """
    Trigger Browserbase deep research scraping.
    This is slower (5-15s) so it runs in background and returns a task ID.
    The frontend polls or uses WebSocket for completion.
    """
    # Log the activity
    await log_agent_activity(
        user_id="system",
        module="research",
        agent_name="Browserbase Agent",
        action=f"Starting deep research scrape for {symbol}",
        status="running",
    )

    # Run scraping in background
    background_tasks.add_task(run_deep_research, symbol)

    return {"status": "started", "message": f"Deep research agent deployed for {symbol}"}
```

---

## 11. New Data Sources — X, Kalshi, Polymarket

### 11.1 X/Twitter Integration

```python
# backend/services/x_service.py
"""X (Twitter) sentiment analysis via API v2."""
import httpx
import os

async def get_x_sentiment(symbol: str, count: int = 20) -> dict:
    """Search recent tweets about a stock and analyze sentiment."""
    bearer_token = os.environ.get("X_BEARER_TOKEN")
    if not bearer_token:
        return {"error": "X API not configured"}

    query = f"${symbol} OR #{symbol} lang:en -is:retweet"
    url = "https://api.twitter.com/2/tweets/search/recent"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={"Authorization": f"Bearer {bearer_token}"},
            params={
                "query": query,
                "max_results": min(count, 100),
                "tweet.fields": "created_at,public_metrics,text",
            },
            timeout=10.0,
        )

        if response.status_code != 200:
            return {"error": f"X API error: {response.status_code}"}

        data = response.json()
        tweets = data.get("data", [])

        # Simple keyword-based sentiment (fast, no LLM needed)
        bullish_words = {"buy", "bull", "long", "moon", "calls", "breakout", "undervalued", "strong"}
        bearish_words = {"sell", "bear", "short", "puts", "crash", "overvalued", "dump", "weak"}

        bullish = 0
        bearish = 0
        for tweet in tweets:
            text = tweet["text"].lower()
            if any(w in text for w in bullish_words): bullish += 1
            if any(w in text for w in bearish_words): bearish += 1

        total = bullish + bearish or 1

        return {
            "symbol": symbol,
            "tweet_count": len(tweets),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "sentiment_score": round((bullish / total) * 100, 1),
            "label": "Bullish" if bullish > bearish else "Bearish" if bearish > bullish else "Neutral",
            "sample_tweets": [
                {"text": t["text"][:200], "metrics": t.get("public_metrics", {})}
                for t in tweets[:5]
            ],
            "source": "X/Twitter",
        }
```

### 11.2 Kalshi Integration

```python
# backend/services/kalshi_service.py
"""Kalshi prediction market data — public endpoints, no auth for market data."""
import httpx

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

async def search_kalshi_markets(query: str, limit: int = 10) -> list[dict]:
    """Search Kalshi markets relevant to a query (e.g., stock ticker, topic)."""
    async with httpx.AsyncClient() as client:
        # Search for events related to the query
        response = await client.get(
            f"{KALSHI_BASE}/events",
            params={"status": "open", "limit": limit},
            timeout=10.0,
        )

        if response.status_code != 200:
            return []

        events = response.json().get("events", [])

        # Filter events that match the query
        query_lower = query.lower()
        matching = []
        for event in events:
            title = event.get("title", "").lower()
            category = event.get("category", "").lower()
            if query_lower in title or query_lower in category or \
               any(query_lower in tag.lower() for tag in event.get("tags", [])):
                matching.append({
                    "event_ticker": event.get("event_ticker"),
                    "title": event.get("title"),
                    "category": event.get("category"),
                    "status": event.get("status"),
                    "markets_count": len(event.get("markets", [])),
                })

        # Also fetch finance/economics events
        response2 = await client.get(
            f"{KALSHI_BASE}/events",
            params={"status": "open", "limit": 20},
            timeout=10.0,
        )

        if response2.status_code == 200:
            for event in response2.json().get("events", []):
                cat = event.get("category", "").lower()
                if cat in ("economics", "finance", "financial", "tech"):
                    matching.append({
                        "event_ticker": event.get("event_ticker"),
                        "title": event.get("title"),
                        "category": event.get("category"),
                    })

        return matching[:limit]

async def get_kalshi_market_details(market_ticker: str) -> dict:
    """Get detailed info for a specific Kalshi market."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{KALSHI_BASE}/markets/{market_ticker}",
            timeout=10.0,
        )
        if response.status_code == 200:
            market = response.json().get("market", {})
            return {
                "ticker": market.get("ticker"),
                "title": market.get("title"),
                "yes_price": market.get("yes_price"),
                "no_price": market.get("no_price"),
                "volume": market.get("volume"),
                "open_interest": market.get("open_interest"),
                "implied_probability": market.get("yes_price", 0),
            }
        return {}
```

### 11.3 Polymarket Integration (Public, No Auth)

```python
# backend/services/polymarket_service.py
"""Polymarket data via public CLOB endpoint — no API key needed."""
import httpx

POLYMARKET_CLOB = "https://clob.polymarket.com"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"

async def search_polymarket_events(query: str, limit: int = 10) -> list[dict]:
    """Search Polymarket for prediction markets related to a query."""
    async with httpx.AsyncClient() as client:
        # Use Gamma API to search markets
        response = await client.get(
            f"{POLYMARKET_GAMMA}/markets",
            params={
                "closed": "false",
                "limit": limit,
            },
            timeout=10.0,
        )

        if response.status_code != 200:
            return []

        markets = response.json()
        if not isinstance(markets, list):
            markets = markets.get("data", markets.get("markets", []))

        query_lower = query.lower()
        matching = []
        for market in markets:
            question = market.get("question", "").lower()
            description = market.get("description", "").lower()
            if query_lower in question or query_lower in description:
                tokens = market.get("tokens", [])
                yes_price = None
                if tokens and len(tokens) > 0:
                    yes_price = tokens[0].get("price")

                matching.append({
                    "question": market.get("question"),
                    "slug": market.get("slug"),
                    "yes_price": yes_price,
                    "volume": market.get("volume"),
                    "liquidity": market.get("liquidity"),
                    "end_date": market.get("end_date_iso"),
                    "url": f"https://polymarket.com/event/{market.get('slug', '')}",
                    "source": "polymarket",
                })

        return matching[:limit]

async def get_polymarket_prices(token_id: str) -> dict:
    """Get live price from Polymarket CLOB."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{POLYMARKET_CLOB}/midpoint",
            params={"token_id": token_id},
            timeout=5.0,
        )
        if response.status_code == 200:
            return response.json()
        return {}
```

---

## 12. Cross-Module Integration Map

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TICKERMASTER DATA FLOW                           │
│                                                                          │
│  USER ──text──▶ POKE ──webhook──▶ BACKEND ──parse──▶ EXECUTE            │
│    │                                                     │               │
│    │  ┌─────────────────────────────────────────────────┘               │
│    │  │                                                                  │
│    │  ▼                                                                  │
│    │  RESEARCH MODULE                                                    │
│    │  ├─ Perplexity Sonar → AI summaries with citations                 │
│    │  ├─ Browserbase/Stagehand → Deep scraping (Finviz, Reddit)        │
│    │  ├─ yfinance → Price, fundamentals, technicals                     │
│    │  ├─ Reddit API → Post sentiment                                     │
│    │  ├─ X/Twitter API → Tweet sentiment                                │
│    │  ├─ Kalshi API → Prediction market odds                            │
│    │  ├─ Polymarket CLOB → Prediction market odds                       │
│    │  └─ FRED API → Macro economic data                                 │
│    │       │                                                             │
│    │       │ Research APIs consumed by:                                   │
│    │       ├──────────────────────────────▶ TRACKER (investigation)      │
│    │       └──────────────────────────────▶ SIMULATION (agent context)   │
│    │                                                                     │
│    │  SIMULATION MODULE                                                  │
│    │  ├─ Modal Sandbox → Isolated simulation containers                 │
│    │  ├─ Order Book Engine → Price-time priority matching               │
│    │  ├─ 4 Agent Archetypes → Quant, Value, Retail, Custom LLM         │
│    │  ├─ GBM Market Generator → Realistic price movements              │
│    │  ├─ OpenRouter GPT-4o-mini → Agent decisions + reasoning           │
│    │  └─ AI Commentator → Witty play-by-play                           │
│    │       │                                                             │
│    │       │ Auto-launched by:                                           │
│    │       └──────────────────────────────── TRACKER (auto-simulate)     │
│    │                                                                     │
│    │  TRACKER MODULE                                                     │
│    │  ├─ Modal Cron → Every-minute polling                              │
│    │  ├─ yfinance → Real-time price/volume                              │
│    │  ├─ Trigger Evaluator → Price %, volume, RSI conditions            │
│    │  ├─ Research Module APIs → Investigation context                    │
│    │  ├─ OpenRouter → Narrative synthesis                               │
│    │  └─ Poke Webhook → Phone notification with deep link              │
│    │                                                                     │
│    │  ALL MODULES ──log──▶ AGENT ACTIVITY TABLE ──ws──▶ OBSERVABILITY   │
│    │                                                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Complete File Structure

```
tickermaster/
├── .env                              # All API keys (gitignored)
├── .gitignore
├── README.md
├── MASTER_BUILD_SPEC.md             # This file
├── RESEARCH_SPEC.md                 # Research module spec
├── SIMULATION_SPEC.md               # Simulation module spec
├── TRACKER_SPEC.md                  # Tracker module spec
│
├── backend/
│   ├── main.py                      # FastAPI app entry point
│   ├── config.py                    # Environment config loader
│   ├── requirements.txt             # Python dependencies
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py                  # Auth routes (verify, profile)
│   │   ├── research.py              # Research endpoints
│   │   ├── simulation.py            # Simulation REST + WebSocket
│   │   ├── tracker.py               # Tracker CRUD + WebSocket
│   │   ├── agents.py                # Agent observability + WebSocket
│   │   └── poke_inbound.py          # Poke conversational webhook
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── database.py              # Supabase client
│   │   ├── agent_logger.py          # Central activity logger
│   │   ├── poke.py                  # Poke notification service
│   │   │
│   │   ├── # Research services
│   │   ├── perplexity_service.py    # Perplexity Sonar AI research
│   │   ├── yfinance_service.py      # Stock data (shared)
│   │   ├── reddit_service.py        # Reddit API sentiment
│   │   ├── x_service.py             # X/Twitter sentiment
│   │   ├── kalshi_service.py        # Kalshi prediction markets
│   │   ├── polymarket_service.py    # Polymarket prediction markets
│   │   ├── fred_service.py          # FRED macro data
│   │   ├── browserbase_scraper.py   # Browserbase/Stagehand deep scraping
│   │   ├── sentiment_engine.py      # Composite sentiment scoring
│   │   │
│   │   ├── # Simulation services
│   │   ├── modal_manager.py         # Modal Sandbox creation
│   │   ├── simulation_engine.py     # Self-contained engine code
│   │   ├── commentator.py           # AI market commentator
│   │   │
│   │   ├── # Tracker services
│   │   ├── tracker_service.py       # CRUD for agents/alerts
│   │   ├── alert_evaluator.py       # Trigger evaluation
│   │   └── alert_pipeline.py        # Investigate → Analyze → Notify
│   │
│   └── models/
│       ├── __init__.py
│       ├── schemas.py               # Shared Pydantic models
│       ├── research_schemas.py      # Research models
│       ├── simulation_schemas.py    # Simulation models
│       └── tracker_schemas.py       # Tracker models
│
├── tracker/
│   ├── modal_cron.py                # Modal Cron function (deployed separately)
│   └── README.md
│
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── next.config.ts
│   ├── middleware.ts                 # Auth middleware (route protection)
│   │
│   ├── lib/
│   │   ├── supabase/
│   │   │   ├── client.ts            # Browser Supabase client
│   │   │   └── server.ts            # Server Supabase client
│   │   ├── api.ts                   # Backend API helper
│   │   └── utils.ts                 # Shared utilities
│   │
│   ├── app/
│   │   ├── layout.tsx               # Root layout (nav, theme, auth)
│   │   ├── page.tsx                 # Landing page
│   │   ├── globals.css              # Global styles + Tailwind
│   │   │
│   │   ├── login/
│   │   │   └── page.tsx             # Auth page
│   │   ├── auth/
│   │   │   └── callback/
│   │   │       └── route.ts         # OAuth callback
│   │   │
│   │   ├── research/
│   │   │   ├── page.tsx             # Research search/home
│   │   │   └── [symbol]/
│   │   │       └── page.tsx         # Ticker research page
│   │   │
│   │   ├── simulation/
│   │   │   ├── page.tsx             # Simulation arena
│   │   │   └── [id]/
│   │   │       └── page.tsx         # Specific sim run
│   │   │
│   │   ├── tracker/
│   │   │   └── page.tsx             # Tracker dashboard
│   │   │
│   │   ├── agents/
│   │   │   └── page.tsx             # Agent observability
│   │   │
│   │   └── tutorial/
│   │       └── page.tsx             # Onboarding tutorial
│   │
│   └── components/
│       ├── ui/                      # Shadcn/UI components
│       ├── nav/
│       │   ├── Navbar.tsx           # Main navigation
│       │   └── UserMenu.tsx         # Auth state + dropdown
│       ├── landing/
│       │   ├── Hero.tsx
│       │   ├── Features.tsx
│       │   └── Footer.tsx
│       ├── research/
│       │   ├── SearchBar.tsx
│       │   ├── PriceChart.tsx
│       │   ├── AIResearchPanel.tsx
│       │   ├── SentimentGauge.tsx
│       │   ├── PredictionMarkets.tsx
│       │   ├── DeepResearchButton.tsx
│       │   └── FundamentalsTable.tsx
│       ├── simulation/
│       │   ├── AgentSetupPanel.tsx
│       │   ├── LivePriceChart.tsx
│       │   ├── OrderBookViz.tsx
│       │   ├── AgentLeaderboard.tsx
│       │   ├── TradeLog.tsx
│       │   └── CommentaryPanel.tsx
│       ├── tracker/
│       │   ├── AgentCard.tsx
│       │   ├── DeployAgentModal.tsx
│       │   ├── AlertFeed.tsx
│       │   └── AlertCard.tsx
│       ├── agents/
│       │   └── ActivityFeed.tsx
│       └── common/
│           ├── LoadingSpinner.tsx
│           ├── ErrorBoundary.tsx
│           └── GlassCard.tsx
│
└── supabase/
    └── migrations/
        └── 001_initial_schema.sql   # Complete schema from Section 4
```

---

## 14. Build Order (Phases 1-10)

> **CRITICAL**: Follow this order. Each phase builds on the previous. Test after each phase.

### Phase 1: Project Scaffolding (30 min)

1. Initialize Next.js project with TypeScript + TailwindCSS
2. Install all frontend deps: `@supabase/ssr`, `@supabase/supabase-js`, `recharts`, `lightweight-charts`, `framer-motion`, `lucide-react`
3. Set up Shadcn/UI: `npx shadcn@latest init`
4. Create `.env` and `.gitignore`
5. Create `backend/` directory with FastAPI skeleton + `requirements.txt`
6. **Test**: Frontend runs on localhost:3000, backend on localhost:8000

### Phase 2: Auth & Database (45 min)

7. Run SQL schema in Supabase SQL Editor
8. Set up Supabase Auth providers (Google + email/password)
9. Implement frontend auth: login page, middleware, Supabase clients
10. Implement backend auth router (verify token, get profile)
11. **Test**: Can sign in, see profile, protected routes redirect to login

### Phase 3: Landing Page & Layout (30 min)

12. Build landing page (hero, features, CTA)
13. Build main app layout (dark theme, nav bar, responsive)
14. Build tutorial page
15. **Test**: Beautiful landing page, smooth navigation, tutorial flow

### Phase 4: Research Module (2 hours)

16. Implement all backend research services (see Research Spec):
    - yfinance, Perplexity Sonar, Reddit API, X/Twitter API
    - Kalshi, Polymarket, FRED
    - Browserbase/Stagehand scraper
    - Composite sentiment engine
17. Implement research router with all endpoints
18. Build frontend research page: search bar, price chart, AI panel, sentiment gauge, prediction markets, deep research button
19. Wire agent logger into all research services
20. **Test**: Enter NVDA → full research page loads with real data from all sources

### Phase 5: Simulation Module (2 hours)

21. Write self-contained simulation engine (see Simulation Spec)
22. Implement Modal Sandbox integration
23. Implement WebSocket for real-time streaming
24. Build frontend: setup panel, live chart, order book, leaderboard, trade log, commentary
25. Wire agent logger into simulation events
26. **Test**: Create custom agent → launch sim → watch real-time → see results

### Phase 6: Tracker Module (1.5 hours)

27. Implement tracker CRUD service + router
28. Write Modal Cron function with trigger evaluation
29. Implement alert pipeline (Research API → LLM narrative → Poke notification)
30. Build frontend: agent cards, deploy modal, alert feed
31. Wire agent logger into tracker events
32. **Test**: Deploy agent → wait for trigger → phone buzzes with alert

### Phase 7: Poke Conversational Interface (45 min)

33. Implement Poke inbound webhook handler
34. Implement intent parsing for research/simulate/track/status commands
35. **Test**: Text Poke "research NVDA" → get summary back

### Phase 8: Agent Observability (30 min)

36. Build agents page with real-time activity feed
37. Connect WebSocket to broadcast agent events
38. Add filter/search to activity feed
39. **Test**: Use all 3 modules → see all agent activity in one feed

### Phase 9: Polish & Integration (1 hour)

40. Add loading states, error boundaries, skeleton loaders
41. Add smooth animations (Framer Motion transitions)
42. Ensure responsive design (mobile-friendly)
43. Add toast notifications for async operations
44. Cross-check all module integrations (Tracker → Research, Tracker → Simulation)
45. Run through full demo script end-to-end

### Phase 10: Deploy & Test (30 min)

46. Deploy frontend to Vercel
47. Deploy backend to Railway/Render
48. Deploy Modal Cron: `modal deploy tracker/modal_cron.py`
49. Create Modal secret with all API keys
50. Update `.env` with production URLs
51. **Final test**: Full demo flow on production

---

## 15. Testing & Debugging Protocol

### After EVERY Phase, Run These Checks:

```bash
# Backend health
curl http://localhost:8000/api/health

# Auth works
# (manual) Sign in via Google or email → verify redirect to dashboard

# Research endpoint
curl http://localhost:8000/api/ticker/NVDA/quote
curl http://localhost:8000/api/ticker/NVDA/ai-research
curl http://localhost:8000/api/ticker/NVDA/sentiment

# Simulation
# (manual) Create simulation via frontend, verify WebSocket streams events

# Tracker
curl -X POST http://localhost:8000/api/tracker/agents \
  -H "Content-Type: application/json" \
  -d '{"symbol":"NVDA","name":"Test","triggers":{"price_change_pct":0.1}}'

# Poke notification
curl -X POST https://poke.com/api/v1/inbound-sms/webhook \
  -H "Authorization: Bearer $POKE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message":"TickerMaster test notification 🎉"}'

# Agent activity
curl http://localhost:8000/api/agents/activity?user_id=test&limit=10
```

### Common Issues & Fixes

- **CORS errors**: Check FastAPI `allow_origins` includes frontend URL
- **Supabase RLS blocking**: Use service role key for backend, anon key for frontend
- **Modal Sandbox timeout**: Increase `timeout=600`, check stdout buffering
- **WebSocket disconnects**: Implement reconnection logic with exponential backoff
- **Poke not delivering**: Verify API key at poke.com/settings/advanced
- **yfinance rate limits**: Add 1-second delay between requests, cache aggressively
- **Browserbase timeout**: Increase `dom_settle_timeout_ms`, check headless mode

---

## 16. Deployment (Vercel + Modal)

### Frontend → Vercel

```bash
cd frontend
vercel --prod
# Set env vars in Vercel dashboard: NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY, NEXT_PUBLIC_API_URL
```

### Backend → Railway or Render

```bash
# railway.app or render.com
# Point to backend/ directory
# Set all env vars from .env
# Start command: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Modal Cron → Modal Cloud

```bash
cd tracker
modal deploy modal_cron.py
# Verify: modal app list (should show tickermaster-tracker)
```

### Post-Deploy Checklist

- [ ] Frontend loads, auth works
- [ ] Research page fetches real data
- [ ] Simulation WebSocket connects
- [ ] Tracker agents visible
- [ ] Poke notifications deliver
- [ ] Agent observability feed shows events
- [ ] Deep research (Browserbase) completes
- [ ] Tutorial flow works for new users

---

## 17. Prize Alignment Matrix

| Prize                         | Primary Module       | Integration Point                                                                 | Confidence |
| ----------------------------- | -------------------- | --------------------------------------------------------------------------------- | ---------- |
| **Modal Sandbox Challenge**   | Simulation           | Entire sim engine in Modal Sandbox                                                | 🟢 High    |
| **Modal Cron**                | Tracker              | Polling engine runs on Modal Cron                                                 | 🟢 High    |
| **Perplexity Sonar**          | Research             | AI research summaries with citations                                              | 🟢 High    |
| **Poke / Interaction Co.**    | Tracker + Poke MCP   | Phone alerts + conversational commands                                            | 🟢 High    |
| **Greylock Multi-Turn Agent** | Simulation + Tracker | Sim agents: analyze→trade→react loop. Tracker: trigger→investigate→analyze→notify | 🟢 High    |
| **OpenAI**                    | All                  | LLM-powered agents throughout                                                     | 🟢 High    |
| **Browserbase**               | Research             | AI browser agents scraping Finviz, Reddit                                         | 🟢 High    |
| **Neo Product Potential**     | All                  | Persistent agents + phone alerts = real product                                   | 🟢 High    |
| **YC Reimagine**              | All                  | Reimagine Bloomberg Terminal for retail                                           | 🟡 Medium  |
| **Visa / Commerce**           | Tracker              | Could extend to execute real trades                                               | 🟡 Medium  |
| **Google / Vercel**           | Infra                | Deployed on Vercel, Supabase (GCP-backed)                                         | 🟡 Medium  |
| **Human Capital**             | All                  | Fellowship-worthy product with real market potential                              | 🟡 Medium  |
| **Fetch.ai**                  | Simulation           | AI agent trading arena                                                            | 🟡 Medium  |
| **Elastic**                   | Research             | Search functionality across tickers/data                                          | 🟡 Medium  |

---

## Critical Reminders for the Coding Agent

1. **Read the sub-module specs before implementing each module.** This master doc gives the skeleton; the sub-specs have full implementation details.

2. **Agent Observability is NOT optional.** Every service call should log to `agent_activity`. This is what makes the app feel "agentic" to users and judges.

3. **The dark theme must be consistent.** Use the color system from Section 6.3 everywhere. No white backgrounds, no mismatched grays.

4. **Test with real API keys.** Don't mock data — the APIs are live. Use real yfinance, real Perplexity, real Kalshi. The demo needs real data.

5. **Poke is the wow factor.** Getting a text message notification during the demo is what will make judges remember this project. Set low trigger thresholds for demo day.

6. **The landing page matters.** First impressions count. Invest time in making it beautiful — gradient text, glass cards, smooth animations.

7. **Modal Sandbox is the prize play.** Make sure simulation actually runs inside a Modal Sandbox, not just locally. Judges will check.

8. **Polymarket has NO API key requirement for read-only market data.** Use the public CLOB endpoint at `https://clob.polymarket.com` and Gamma API at `https://gamma-api.polymarket.com`.

9. **Kalshi's market data endpoints are public too.** Use `https://api.elections.kalshi.com/trade-api/v2/markets` — no auth needed for reading market data.

10. **Every page should have content within 2 seconds.** Use skeleton loaders, progressive loading, and cache aggressively. Slow is death at a hackathon demo.
