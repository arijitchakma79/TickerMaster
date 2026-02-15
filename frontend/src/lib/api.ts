import axios from "axios";
import { searchTickerDirectory as searchTickerDirectoryOnline } from "./tickerDirectory";
import type {
  AgentActivity,
  AdvancedStockData,
  AgentConfig,
  CandlePoint,
  DeepResearchResponse,
  MarketMetric,
  ModalCronHealthResponse,
  ModalSandboxResponse,
  ResearchResponse,
  TrackerAgent,
  TrackerAgentDetail,
  TrackerAgentInteractResponse,
  SimulationState,
  TickerLookup,
  TrackerSnapshot
} from "./types";

function trimTrailingSlashes(value: string) {
  return value.replace(/\/+$/, "");
}

const API_URL = trimTrailingSlashes(import.meta.env.VITE_API_URL ?? "http://localhost:8000");

const client = axios.create({
  baseURL: API_URL,
  timeout: 30000
});

export const getApiUrl = () => API_URL;

export const getWsUrl = () => {
  const explicit = import.meta.env.VITE_WS_URL;
  if (explicit) return trimTrailingSlashes(explicit);
  const fromHttp = API_URL.replace(/^https?/i, (prefix) => (prefix.toLowerCase() === "https" ? "wss" : "ws"));
  return `${fromHttp}/ws/stream?channels=global,simulation,tracker,agents`;
};

export async function runResearch(ticker: string, timeframe = "7d"): Promise<ResearchResponse> {
  const { data } = await client.post<ResearchResponse>("/research/analyze", {
    ticker,
    timeframe,
    include_prediction_markets: true
  });
  return data;
}

export async function fetchCandles(ticker: string, period = "3mo", interval = "1d") {
  const { data } = await client.get<{ ticker: string; points: CandlePoint[] }>(
    `/research/candles/${ticker}`,
    { params: { period, interval } }
  );
  return data.points;
}

export async function fetchAdvancedStockData(ticker: string): Promise<AdvancedStockData> {
  const { data } = await client.get<AdvancedStockData>(`/research/advanced/${ticker}`);
  return data;
}

export async function fetchRealtimeQuote(ticker: string): Promise<MarketMetric> {
  const { data } = await client.get<MarketMetric>(`/research/quote/${ticker}`);
  return data;
}

export async function searchTickerDirectory(query: string, limit = 8): Promise<TickerLookup[]> {
  return searchTickerDirectoryOnline(query, limit);
}
export async function startSimulation(payload: {
  ticker: string;
  target_tickers?: string[];
  duration_seconds: number;
  initial_price: number;
  starting_cash: number;
  volatility: number;
  inference_runtime?: "direct" | "modal";
  agents: AgentConfig[];
}): Promise<SimulationState> {
  const { data } = await client.post<SimulationState>("/simulation/start", payload);
  return data;
}

export async function stopSimulation(sessionId: string) {
  const { data } = await client.post(`/simulation/stop/${sessionId}`);
  return data;
}

export async function pauseSimulation(sessionId: string): Promise<SimulationState> {
  const { data } = await client.post<SimulationState>(`/simulation/pause/${sessionId}`);
  return data;
}

export async function resumeSimulation(sessionId: string): Promise<SimulationState> {
  const { data } = await client.post<SimulationState>(`/simulation/resume/${sessionId}`);
  return data;
}

export async function getSimulationState(sessionId: string): Promise<SimulationState> {
  const { data } = await client.get<SimulationState>(`/simulation/state/${sessionId}`);
  return data;
}

export async function getSimulationSessions() {
  const { data } = await client.get<{ sessions: SimulationState[] }>("/simulation/sessions");
  return data.sessions;
}

export async function triggerTrackerPoll(): Promise<TrackerSnapshot> {
  const { data } = await client.post<TrackerSnapshot>("/tracker/poll");
  return data;
}

export async function getTrackerSnapshot(): Promise<TrackerSnapshot> {
  const { data } = await client.get<TrackerSnapshot>("/tracker/snapshot");
  return data;
}

export async function setWatchlist(tickers: string[]) {
  const { data } = await client.post<{ watchlist: string[] }>("/tracker/watchlist", { tickers });
  return data.watchlist;
}

export async function getWatchlist() {
  const { data } = await client.get<{ watchlist: string[] }>("/tracker/watchlist");
  return data.watchlist;
}

export async function addAlert(payload: { ticker: string; threshold_percent: number; direction: "up" | "down" | "either" }) {
  const { data } = await client.post("/tracker/alerts", payload);
  return data;
}

export async function fetchIntegrations() {
  const { data } = await client.get<Record<string, boolean>>("/integrations");
  return data;
}

export async function getAgentActivity(limit = 80, module?: string): Promise<AgentActivity[]> {
  const params: { limit: number; module?: string } = { limit };
  if (module) params.module = module;
  const { data } = await client.get<{ items?: AgentActivity[] }>("/api/agents/activity", { params });
  return Array.isArray(data.items) ? data.items : [];
}

export async function requestCommentary(prompt: string, context?: Record<string, unknown>) {
  const { data } = await client.post<{ response: string; model: string; generated_at: string }>(
    "/chat/commentary",
    { prompt, context }
  );
  return data;
}

export async function spinModalSandbox(prompt: string, session_id: string): Promise<ModalSandboxResponse> {
  const { data } = await client.post<ModalSandboxResponse>("/simulation/modal/sandbox", { prompt, session_id });
  return data;
}

export async function getModalCronHealth(): Promise<ModalCronHealthResponse> {
  const { data } = await client.get<ModalCronHealthResponse>("/simulation/modal/cron-health");
  return data;
}

export async function runDeepResearch(ticker: string): Promise<DeepResearchResponse> {
  const { data } = await client.post<DeepResearchResponse>(`/research/deep/${ticker}`);
  return data;
}

export async function createTrackerAgent(payload: {
  symbol: string;
  name: string;
  triggers: Record<string, unknown>;
  auto_simulate?: boolean;
}): Promise<TrackerAgent> {
  const { data } = await client.post<TrackerAgent>("/api/tracker/agents", payload);
  return data;
}

export async function listTrackerAgents(): Promise<TrackerAgent[]> {
  const { data } = await client.get<TrackerAgent[]>("/api/tracker/agents");
  return data;
}

export async function deleteTrackerAgent(agentId: string): Promise<{ ok: boolean }> {
  const { data } = await client.delete<{ ok: boolean }>(`/api/tracker/agents/${agentId}`);
  return data;
}

export async function getTrackerAgentDetail(agentId: string): Promise<TrackerAgentDetail> {
  const { data } = await client.get<TrackerAgentDetail>(`/api/tracker/agents/${agentId}/detail`);
  return data;
}

export async function createTrackerAgentByPrompt(prompt: string, userId?: string) {
  const { data } = await client.post<{ ok: boolean; agent: TrackerAgent; parsed: Record<string, unknown> }>(
    "/api/tracker/agents/nl-create",
    { prompt, user_id: userId }
  );
  return data;
}

export async function interactWithTrackerAgent(agentId: string, message: string, userId?: string): Promise<TrackerAgentInteractResponse> {
  const { data } = await client.post<TrackerAgentInteractResponse>(`/api/tracker/agents/${agentId}/interact`, {
    message,
    user_id: userId
  });
  return data;
}
