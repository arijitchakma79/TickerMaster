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

export interface CandlePoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
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
  [key: string]: unknown;
}
