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
  notes?: string;
}

export interface CandlePoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
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

export interface SimulationState {
  session_id: string;
  ticker: string;
  running: boolean;
  tick: number;
  current_price: number;
  volatility: number;
  started_at: string;
  ends_at: string;
  crash_mode: boolean;
  recent_news: string[];
  trades: TradeRecord[];
  portfolios: Record<string, Record<string, number>>;
  order_book: {
    bids: Array<{ price: number; size: number }>;
    asks: Array<{ price: number; size: number }>;
  };
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
