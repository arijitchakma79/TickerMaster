# TickerMaster Gap Analysis (Phase 0c)

## Already Built (Keep + Polish)
- FastAPI backend with working routers for research, simulation, tracker, system, chat.
- Real-time WebSocket manager and simulation/tracker streaming infrastructure.
- yfinance integration for quote/candles and S&P return window.
- Core simulation engine with agent policies, order-book/slippage, and event broadcasting.
- Tracker polling loop with Perplexity-based investigation and alert generation.
- Frontend dashboard UI with dark/light theme, tabbed modules, charts, and event rail.

## Partially Built (Complete per spec)
- Research module: has Perplexity/X/Reddit/Kalshi/Polymarket basics but lacks FRED, deep research, cache table integration, weighted composite, and full endpoint contract.
- Tracker: has watchlist/alerts polling but no spec-style CRUD for persistent tracker agents and alert feed API contract.
- Observability: WebSocket event rail exists, but no dedicated `/agents` activity API/page and no guaranteed `agent_activity` DB logging for all actions.
- Environment and infra: required keys and full Supabase schema were not committed as runnable project artifacts.

## Missing (New Build)
- Next.js App Router frontend architecture with protected routes and Supabase SSR auth.
- Dedicated landing page + tutorial route flow as specified.
- Poke conversational inbound webhook endpoint and command routing.
- Full Browserbase/Stagehand deep scraping implementation.
- Modal cron deployment artifact for tracker polling (`tracker/modal_cron.py`) and full production deployment wiring.

## Architectural Mismatch Found
- Current frontend is **Vite React SPA**, but spec requires **Next.js App Router + @supabase/ssr middleware auth**.
- Existing backend routes are mostly under `/research`, `/simulation`, `/tracker`; spec validation expects `/api/...` route namespace.
