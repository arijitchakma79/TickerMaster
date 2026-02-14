# TickerMaster MVP (TreeHacks 2026)

TickerMaster is a real-time sandbox for learning trading dynamics through AI agents and market intelligence feeds.

Core product surfaces:
1. `Research`: Perplexity Sonar + X + Reddit + prediction-market context.
2. `Simulation`: Multi-agent arena with order-book impact, slippage, delayed news propagation, and crash regimes.
3. `Tracker`: Yahoo-style watchlist with valuation metrics, spike detection, and alert pipeline.

## Stack
- Backend: `FastAPI` + `WebSockets` + `yfinance`
- Frontend: `React` + `TypeScript` + `Vite` + `Recharts`
- Agent models: `OpenRouter` (open-source model default: `meta-llama/llama-3.1-8b-instruct`)
- Commentary model: `OpenAI`

## Monorepo Layout
```text
TickerMaster/
  backend/
    app/
      main.py
      schemas.py
      routers/
      services/
    requirements.txt
    .env.example
  frontend/
    src/
      components/
      hooks/
      lib/
    package.json
    .env.example
  .env.example
  .gitignore
```

## Quick Start

### 1) Configure environment
Copy `.env.example` to `.env` in the project root or into `backend/.env` and set keys.

### 2) Run backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3) Run frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173`  
Backend: `http://localhost:8000`

## API Highlights
- `POST /research/analyze`
- `GET /research/candles/{ticker}`
- `POST /simulation/start`
- `POST /simulation/stop/{session_id}`
- `GET /simulation/sessions`
- `POST /simulation/modal/sandbox`
- `GET /simulation/modal/cron-health`
- `GET /tracker/snapshot`
- `POST /tracker/watchlist`
- `POST /tracker/alerts`
- `POST /tracker/poll`
- `POST /chat/commentary`
- `GET /integrations`
- `WS /ws/stream?channels=global,simulation,tracker`

## How the MVP Maps to Sponsor Tool Requirements

### Research
- Perplexity Sonar API for catalyst synthesis.
- X API and Reddit API ingestion for public sentiment flow.
- Kalshi + Polymarket adapters for prediction-market context.
- Finance graphing via Yahoo-style candles and metric tables.
- Tool links exposed in UI for Morningstar / Reuters / J.P. Morgan / Yahoo Finance.

### Simulation
- Natural-language sandbox trigger endpoint for Modal (`/simulation/modal/sandbox`).
- OpenRouter-powered agents with user-defined parameters:
  - personality
  - model
  - aggressiveness
  - risk limit
  - trade size
- Realism mechanics:
  - order book spread + market impact
  - execution slippage
  - delayed news diffusion (quant first, retail lag)
  - crash regimes sampled from S&P 500 return distribution

### Tracker Pipeline
- `Trigger`: periodic polling loop for price/volume anomalies (cron-ready for Modal).
- `Investigate`: Perplexity Sonar explains likely catalysts.
- `Analyze`: Cerebras or NVIDIA NIM synthesizes high-signal narrative.
- `Notify`: Poke Recipe/MCP handoff payload (`npx poke` workflow, no direct Poke HTTP API dependency).

### Poke Setup (TreeHacks)
Run this once from repo root to wrap TickerMaster as a Poke MCP Recipe:
```bash
npx poke
```
Then open Kitchen to test/deploy your Recipe and wire alert payloads:
- Kitchen: https://poke.com/kitchen
- Recipes docs: https://poke.com/docs/recipes

## Required External Links
- OpenAI: https://platform.openai.com/
- OpenRouter: https://openrouter.ai/
- Perplexity Sonar: https://docs.perplexity.ai/
- X API: https://developer.x.com/en/docs
- Reddit API: https://www.reddit.com/dev/api/
- Kalshi API: https://docs.kalshi.com/
- Polymarket: https://docs.polymarket.com/
- Modal Sandbox: https://modal.com/docs/guide/sandbox
- Modal Cron: https://modal.com/docs/guide/cron
- Poke Docs: https://poke.com/docs/recipes
- Poke Kitchen: https://poke.com/kitchen
- Poke npm package: https://www.npmjs.com/package/poke
- Interaction Company: https://interaction.co/
- Cerebras API: https://inference-docs.cerebras.ai/
- NVIDIA NIM: https://build.nvidia.com/
- Morningstar: https://www.morningstar.com/
- Reuters Markets: https://www.reuters.com/markets/
- J.P. Morgan Insights: https://www.jpmorgan.com/insights
- Yahoo Finance: https://finance.yahoo.com/

## Notes
- This MVP is educational and not investment advice.
- Most integrations gracefully fall back to synthetic/demo responses if keys are missing.
- For production: persist state, secure auth, rate-limit providers, and harden retry/backoff logic.
