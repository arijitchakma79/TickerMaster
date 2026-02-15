# TickerMaster MVP (TreeHacks 2026)

TickerMaster is a real-time sandbox for learning trading dynamics through AI agents and market intelligence feeds.

Core product surfaces:
1. `Research`: Perplexity Sonar + X + Reddit + prediction-market context.
2. `Simulation`: Multi-agent arena with order-book impact, slippage, delayed news propagation, and crash regimes.
3. `Tracker`: Real-time watchlist with valuation metrics, spike detection, and alert pipeline.

## Stack
- Backend: `FastAPI` + `WebSockets`
- Market Data: `Alpaca` (primary) + `Finnhub` (fallback)
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

## Step-by-Step Startup

### 1) Create `.env` in repo root
Create `/TickerMaster/.env` and include at minimum:

```env
# Supabase
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_KEY=<your-publishable-key>
SUPABASE_SERVICE_KEY=<your-secret-service-role-key>
DATABASE_URL=postgresql://postgres:<password>@db.<project>.supabase.co:5432/postgres

# Backend URL
BACKEND_URL=http://localhost:8000
```

Add your API keys for Alpaca / Finnhub / Perplexity / OpenAI / OpenRouter / X / Browserbase / Modal as needed.

For Modal sandbox runtime, also set:
- `MODAL_SIMULATION_APP_NAME` (default `tickermaster-simulation`)
- `MODAL_SANDBOX_TIMEOUT_SECONDS` (default `600`)
- `MODAL_SANDBOX_IDLE_TIMEOUT_SECONDS` (default `120`)
- `MODAL_INFERENCE_FUNCTION_NAME` (default `agent_inference`)
- `MODAL_INFERENCE_TIMEOUT_SECONDS` (default `15`)

To enable Modal inference function:
```bash
modal secret create tickermaster-secrets OPENROUTER_API_KEY=<your-openrouter-key>
modal deploy simulation/modal_inference.py
```

For frontend auth, add these in `frontend/.env`:
```bash
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=https://<your-project>.supabase.co
VITE_SUPABASE_ANON_KEY=<your-publishable-key>
```

### 2) Apply database schema in Supabase
In Supabase Dashboard:
1. Open `SQL Editor`.
2. Paste contents of `supabase/schema.sql`.
3. Run it once.

This creates tables like `research_cache`, `agent_activity`, `simulations`, `tracker_agents`, `tracker_alerts`, `watchlist`, and `favorite_stocks`.

### 3) Start backend (Terminal A)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend URL: `http://localhost:8000`

### 4) Start frontend (Terminal B)
```bash
cd frontend
npm install
npm run dev
```

Frontend URL: `http://localhost:5173`

### 5) Verify backend is healthy
```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/ticker/NVDA/quote
curl http://localhost:8000/api/ticker/NVDA/ai-research
curl http://localhost:8000/api/ticker/NVDA/sentiment
curl "http://localhost:8000/api/prediction-markets?query=fed"
curl http://localhost:8000/api/ticker/NVDA/x-sentiment
```

### 6) Optional: verify Supabase writes
Use Supabase SQL Editor or REST to confirm rows are being inserted into:
- `research_cache`
- `agent_activity`
- `simulations`
- `tracker_alerts`

If cache writes appear but activity/alerts do not, confirm backend is using `SUPABASE_SERVICE_KEY` (not only publishable key).

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
- Finance graphing via Alpaca/Finnhub candles and metric tables.
- Tool links exposed in UI for Morningstar / Reuters / J.P. Morgan / Alpaca / Finnhub.

### Simulation
- Natural-language sandbox trigger endpoint for Modal (`/simulation/modal/sandbox`).
- Backend now launches Modal sandboxes through the Modal Python SDK (`modal==1.3.3`) when credentials are present.
- Simulation sessions started with `inference_runtime=modal` first call a Modal function for agent decisions, then fall back to direct OpenRouter if Modal inference is unavailable.
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
- Alpaca Market Data: https://docs.alpaca.markets/docs/about-market-data-api
- Finnhub: https://finnhub.io/

## Notes
- This MVP is educational and not investment advice.
- Most integrations gracefully fall back to synthetic/demo responses if keys are missing.
- For production: persist state, secure auth, rate-limit providers, and harden retry/backoff logic.
