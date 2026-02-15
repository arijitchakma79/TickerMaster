export type SentimentTag = "bullish" | "neutral" | "bearish";

export interface SourceLink {
  source: string;
  title: string;
  url: string;
}

export interface SentimentBreakdown {
  source: string;
  sentiment: SentimentTag;
  score: number;
  summary: string;
  links: SourceLink[];
}

export interface ResearchResponse {
  ticker: string;
  generated_at: string;
  aggregate_sentiment: number;
  recommendation: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell";
  narratives: string[];
  source_breakdown: SentimentBreakdown[];
  prediction_markets: Record<string, unknown>[];
  tool_links: SourceLink[];
}

export interface DeepResearchResponse {
  symbol: string;
  source: string;
  analyst_ratings?: string;
  insider_trading?: string;
  reddit_dd_summary?: string;
  recommendation_timeline?: Array<{
    period: string;
    strong_buy: number;
    buy: number;
    hold: number;
    sell: number;
    strong_sell: number;
  }>;
  price_target?: {
    last_updated?: string;
    target_high?: number | null;
    target_low?: number | null;
    target_mean?: number | null;
    target_median?: number | null;
  };
  insider_highlights?: Array<{
    date?: string;
    name?: string;
    code?: string;
    shares?: number | null;
    price?: number | null;
    value_estimate?: number | null;
  }>;
  reddit_highlights?: Array<{
    subreddit?: string;
    title?: string;
    url?: string;
    score?: number | null;
    comments?: number | null;
  }>;
  recent_news?: Array<{
    headline?: string;
    source?: string;
    url?: string;
    datetime?: string | number;
    summary?: string;
  }>;
  deep_bullets?: string[];
  sources?: string[];
  notes?: string;
}

export interface ResearchChatResponse {
  ticker: string;
  response: string;
  model: string;
  generated_at: string;
  context_refreshed: boolean;
  sources: string[];
}

export interface CandlePoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndicatorSnapshot {
  ticker: string;
  period: string;
  interval: string;
  latest: {
    sma20?: number | null;
    sma50?: number | null;
    sma200?: number | null;
    ema21?: number | null;
    ema50?: number | null;
    vwap?: number | null;
    rsi14?: number | null;
    macd_line?: number | null;
    macd_signal?: number | null;
    macd_hist?: number | null;
    bb_upper?: number | null;
    bb_mid?: number | null;
    bb_lower?: number | null;
    atr14?: number | null;
  };
  available: string[];
}

export interface InsiderTransaction {
  start_date?: string | null;
  filer_name?: string | null;
  filer_relation?: string | null;
  money_text?: string | null;
  shares?: number | null;
  value?: number | null;
  ownership?: string | null;
}

export interface AdvancedStockData {
  ticker: string;
  current_price?: number | null;
  change_percent?: number | null;
  company_name?: string | null;
  exchange?: string | null;
  sector?: string | null;
  industry?: string | null;
  website?: string | null;
  description?: string | null;
  market_cap?: number | null;
  beta?: number | null;
  trailing_pe?: number | null;
  forward_pe?: number | null;
  eps_trailing?: number | null;
  eps_forward?: number | null;
  dividend_yield?: number | null;
  fifty_two_week_high?: number | null;
  fifty_two_week_low?: number | null;
  avg_volume?: number | null;
  volume?: number | null;
  recommendation?: string | null;
  target_mean_price?: number | null;
  insider_transactions: InsiderTransaction[];
}

export interface MarketMetric {
  ticker: string;
  price: number;
  change_percent: number;
  pe_ratio?: number | null;
  beta?: number | null;
  volume?: number | null;
  market_cap?: number | null;
}

export interface MarketMoversResponse {
  generated_at: string;
  universe_size: number;
  winners: MarketMetric[];
  losers: MarketMetric[];
  tickers: MarketMetric[];
}

export interface TickerLookup {
  ticker: string;
  name: string;
  exchange?: string | null;
  instrument_type?: string | null;
}

export interface AgentActivity {
  module?: string;
  agent_name?: string;
  action?: string;
  status?: string;
  details?: Record<string, unknown>;
  created_at?: string;
  timestamp?: string;
  channel?: string;
  type?: string;
}

export interface TrackerSnapshot {
  generated_at: string;
  tickers: MarketMetric[];
  alerts_triggered: Array<Record<string, unknown>>;
}

export interface TrackerAgent {
  id: string;
  symbol: string;
  name: string;
  status: "active" | "paused" | "deleted";
  triggers: Record<string, unknown>;
  auto_simulate: boolean;
  total_alerts?: number;
  last_alert_at?: string | null;
  created_at?: string;
}

export interface TrackerAgentDetail {
  agent: TrackerAgent;
  market: MarketMetric | null;
  recent_alerts: Array<Record<string, unknown>>;
  recent_actions: Array<Record<string, unknown>>;
}

export interface TrackerAgentInteractResponse {
  ok: boolean;
  agent: TrackerAgent;
  reply: { response: string; model: string; generated_at: string };
  parsed_intent: Record<string, unknown>;
  market_state: Record<string, unknown>;
  research_state: Record<string, unknown>;
  tool_outputs?: {
    chart?: { period: string; interval: string; points: CandlePoint[] };
    research?: Record<string, unknown>;
    simulation?: { session_id: string; ticker: string };
  };
}

export interface AgentConfig {
  name: string;
  personality: "quant_momentum" | "fundamental_value" | "retail_reactive";
  model: string;
  strategy_prompt?: string;
  aggressiveness: number;
  risk_limit: number;
  trade_size: number;
  active: boolean;
}

export interface TradeRecord {
  timestamp: string;
  session_id: string;
  ticker: string;
  side: "buy" | "sell";
  agent: string;
  quantity: number;
  price: number;
  slippage_bps: number;
  rationale: string;
}

export interface PortfolioPosition {
  holdings: number;
  avg_cost: number;
  market_price: number;
  market_value: number;
  realized_pnl: number;
  unrealized_pnl: number;
  net_gain: number;
}

export interface PortfolioSnapshot {
  cash: number;
  holdings: number;
  avg_cost: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl?: number;
  equity: number;
  positions?: Record<string, PortfolioPosition>;
}

export interface SimulationState {
  session_id: string;
  ticker: string;
  tickers: string[];
  running: boolean;
  paused: boolean;
  tick: number;
  current_price: number;
  market_prices: Record<string, number>;
  volatility: number;
  started_at: string;
  ends_at: string;
  crash_mode: boolean;
  recent_news: string[];
  trades: TradeRecord[];
  portfolios: Record<string, PortfolioSnapshot>;
  order_book: {
    bids: Array<{ price: number; size: number }>;
    asks: Array<{ price: number; size: number }>;
  };
}

export interface ModalSandboxResponse {
  status: "started" | "failed" | "stub" | string;
  message?: string;
  error?: string;
  hint?: string;
  session_id: string;
  sandbox_id?: string;
  app_id?: string;
  app_name?: string;
  dashboard_url?: string;
  app_dashboard_url?: string;
  sandbox_dashboard_url?: string;
  prompt_preview?: string;
  stdout_preview?: string[];
  metadata?: Record<string, unknown>;
  link?: string;
}

export interface ModalCronHealthResponse {
  status: "configured" | "stub" | string;
  message: string;
  polling_interval_seconds: number;
  app_name?: string;
  sdk_available?: boolean;
  sandbox_timeout_seconds?: number;
  sandbox_idle_timeout_seconds?: number;
  inference_function_name?: string;
  inference_timeout_seconds?: number;
  install_hint?: string;
}

export interface WSMessage {
  channel?: string;
  type: string;
  module?: "research" | "simulation" | "tracker" | string;
  agent_name?: string;
  action?: string;
  status?: string;
  details?: Record<string, unknown>;
  created_at?: string;
  timestamp?: string;
  [key: string]: unknown;
}
