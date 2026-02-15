import axios from "axios";
import { searchTickerDirectory as searchTickerDirectoryOnline } from "./tickerDirectory";
import type {
  AgentActivity,
  AdvancedStockData,
  AgentConfig,
  CandlePoint,
  DeepResearchResponse,
  IndicatorSnapshot,
  MarketMetric,
  MarketMoversResponse,
  ModalCronHealthResponse,
  ModalSandboxResponse,
  ResearchChatResponse,
  ResearchResponse,
  TrackerAgent,
  TrackerAgentDetail,
  TrackerAgentInteractResponse,
  SimulationState,
  TickerLookup,
  TrackerSnapshot
} from "./types";

type AuthUser = {
  id: string;
  email?: string;
};

export type AuthSession = {
  access_token: string;
  refresh_token?: string;
  user: AuthUser;
};

const AUTH_STORAGE_KEY = "tickermaster-auth-session";
const SUPABASE_URL = trimTrailingSlashes(import.meta.env.VITE_SUPABASE_URL ?? "");
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY ?? "";

function trimTrailingSlashes(value: string) {
  return value.replace(/\/+$/, "");
}

const API_URL = trimTrailingSlashes(import.meta.env.VITE_API_URL ?? "http://localhost:8000");

const client = axios.create({
  baseURL: API_URL,
  timeout: 30000
});

function normalizeResearchResponse(input: unknown, ticker: string): ResearchResponse {
  const raw = (input ?? {}) as Record<string, unknown>;
  const recommendationRaw = String(raw.recommendation ?? "hold");
  const recommendation =
    recommendationRaw === "strong_buy" ||
    recommendationRaw === "buy" ||
    recommendationRaw === "hold" ||
    recommendationRaw === "sell" ||
    recommendationRaw === "strong_sell"
      ? recommendationRaw
      : "hold";

  const narratives = Array.isArray(raw.narratives)
    ? raw.narratives.map((item) => String(item)).filter(Boolean)
    : typeof raw.summary === "string"
      ? raw.summary.split(/\n+/).map((item) => item.trim()).filter(Boolean)
      : [];

  const source_breakdown = Array.isArray(raw.source_breakdown)
    ? (raw.source_breakdown as ResearchResponse["source_breakdown"])
    : [];

  const prediction_markets = Array.isArray(raw.prediction_markets) ? raw.prediction_markets : [];

  const tool_links = Array.isArray(raw.tool_links)
    ? (raw.tool_links as ResearchResponse["tool_links"])
    : Array.isArray(raw.citations)
      ? (raw.citations as ResearchResponse["tool_links"])
      : [];

  return {
    ticker: String(raw.ticker ?? ticker).toUpperCase(),
    generated_at: String(raw.generated_at ?? new Date().toISOString()),
    aggregate_sentiment: Number(raw.aggregate_sentiment ?? 0),
    recommendation,
    narratives,
    source_breakdown,
    prediction_markets,
    tool_links
  };
}

function networkErrorMessage(feature: string): Error {
  return new Error(
    `${feature} could not reach backend at ${API_URL}. Ensure backend is running and FRONTEND_ORIGINS includes your frontend URL.`
  );
}

function parseJwtUserId(token?: string): string | null {
  if (!token) return null;
  const chunks = token.split(".");
  if (chunks.length < 2) return null;
  try {
    const payloadRaw = atob(chunks[1].replace(/-/g, "+").replace(/_/g, "/"));
    const payload = JSON.parse(payloadRaw) as { sub?: string };
    return typeof payload.sub === "string" ? payload.sub : null;
  } catch {
    return null;
  }
}

function readSessionFromStorage(): AuthSession | null {
  try {
    const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AuthSession;
    if (!parsed?.access_token || !parsed?.user?.id) return null;
    return parsed;
  } catch {
    return null;
  }
}

let authSession: AuthSession | null = readSessionFromStorage();
const authSubscribers = new Set<(session: AuthSession | null) => void>();

function notifyAuthSubscribers() {
  for (const subscriber of authSubscribers) {
    try {
      subscriber(authSession);
    } catch {
      // no-op
    }
  }
}

function setAuthSession(next: AuthSession | null) {
  authSession = next;
  try {
    if (next) window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(next));
    else window.localStorage.removeItem(AUTH_STORAGE_KEY);
  } catch {
    // no-op
  }
  notifyAuthSubscribers();
}

client.interceptors.request.use((config) => {
  const token = authSession?.access_token;
  const userId = authSession?.user?.id ?? parseJwtUserId(token ?? undefined);

  if (typeof (config.headers as { set?: unknown } | undefined)?.set === "function") {
    const headers = config.headers as { set: (key: string, value: string) => void };
    if (token) headers.set("Authorization", `Bearer ${token}`);
    if (userId) headers.set("x-user-id", userId);
    return config;
  }

  const nextHeaders: Record<string, string> = {
    ...((config.headers as Record<string, string>) ?? {})
  };
  if (token) {
    nextHeaders.Authorization = `Bearer ${token}`;
  }
  if (userId) {
    nextHeaders["x-user-id"] = userId;
  }
  config.headers = nextHeaders as typeof config.headers;
  return config;
});

export const getApiUrl = () => API_URL;
export const isAuthConfigured = () => Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);
export const getAuthSession = () => authSession;
export const subscribeAuthSession = (callback: (session: AuthSession | null) => void) => {
  authSubscribers.add(callback);
  return () => {
    authSubscribers.delete(callback);
  };
};

async function callSupabaseAuth(path: string, body: Record<string, unknown>) {
  if (!isAuthConfigured()) {
    throw new Error("Supabase auth is not configured in frontend env.");
  }
  const response = await fetch(`${SUPABASE_URL}/auth/v1/${path}`, {
    method: "POST",
    headers: {
      apikey: SUPABASE_ANON_KEY,
      Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message =
      (typeof payload?.msg === "string" && payload.msg) ||
      (typeof payload?.error_description === "string" && payload.error_description) ||
      (typeof payload?.error === "string" && payload.error) ||
      "Authentication request failed.";
    throw new Error(message);
  }
  return payload as Record<string, unknown>;
}

function toAuthSession(payload: Record<string, unknown>): AuthSession | null {
  const accessToken = typeof payload.access_token === "string" ? payload.access_token : "";
  const refreshToken = typeof payload.refresh_token === "string" ? payload.refresh_token : undefined;
  const userPayload = payload.user;
  const userId =
    (typeof (userPayload as { id?: unknown })?.id === "string" && (userPayload as { id: string }).id) ||
    parseJwtUserId(accessToken) ||
    "";
  if (!accessToken || !userId) return null;
  const email =
    typeof (userPayload as { email?: unknown })?.email === "string"
      ? (userPayload as { email: string }).email
      : undefined;
  return {
    access_token: accessToken,
    refresh_token: refreshToken,
    user: { id: userId, email }
  };
}

export async function signInWithPassword(email: string, password: string): Promise<AuthSession> {
  const payload = await callSupabaseAuth("token?grant_type=password", { email, password });
  const session = toAuthSession(payload);
  if (!session) throw new Error("Sign-in succeeded but no usable session was returned.");
  setAuthSession(session);
  return session;
}

export async function signUpWithPassword(email: string, password: string): Promise<AuthSession | null> {
  const payload = await callSupabaseAuth("signup", { email, password });
  const session = toAuthSession(payload);
  if (session) setAuthSession(session);
  return session;
}

export async function signOut(): Promise<void> {
  const token = authSession?.access_token;
  if (token && isAuthConfigured()) {
    try {
      await fetch(`${SUPABASE_URL}/auth/v1/logout`, {
        method: "POST",
        headers: {
          apikey: SUPABASE_ANON_KEY,
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });
    } catch {
      // no-op
    }
  }
  setAuthSession(null);
}

export const getWsUrl = () => {
  const explicit = import.meta.env.VITE_WS_URL;
  if (explicit) return trimTrailingSlashes(explicit);
  const fromHttp = API_URL.replace(/^https?/i, (prefix) => (prefix.toLowerCase() === "https" ? "wss" : "ws"));
  return `${fromHttp}/ws/stream?channels=global,simulation,tracker,agents`;
};

export async function runResearch(ticker: string, timeframe = "7d"): Promise<ResearchResponse> {
  const symbol = ticker.trim().toUpperCase();
  const payload = { ticker: symbol, timeframe, include_prediction_markets: true };
  const endpoints = ["/research/analyze", "/api/research/analyze"];
  let lastError: unknown = null;

  for (const endpoint of endpoints) {
    try {
      const { data } = await client.post<ResearchResponse>(endpoint, payload);
      return normalizeResearchResponse(data, symbol);
    } catch (error) {
      lastError = error;
      if (axios.isAxiosError(error) && error.response?.status === 404) continue;
      if (axios.isAxiosError(error) && !error.response) continue;
      break;
    }
  }

  try {
    const { data } = await client.get<{ research?: ResearchResponse }>(`/api/ticker/${symbol}`, { params: { timeframe } });
    if (data?.research) {
      return normalizeResearchResponse(data.research, symbol);
    }
  } catch (error) {
    lastError = error;
  }

  try {
    const { data } = await client.get(`/api/ticker/${symbol}/ai-research`, { params: { timeframe } });
    return normalizeResearchResponse(data, symbol);
  } catch (error) {
    lastError = error;
  }

  if (axios.isAxiosError(lastError) && !lastError.response) {
    throw networkErrorMessage("Research");
  }
  throw lastError instanceof Error ? lastError : new Error("Research request failed.");
}

export async function fetchCandles(ticker: string, period = "3mo", interval = "1d", refresh = false) {
  const { data } = await client.get<{ ticker: string; points: CandlePoint[] }>(
    `/research/candles/${ticker}`,
    { params: { period, interval, refresh } }
  );
  return data.points;
}

export async function fetchIndicatorSnapshot(ticker: string, period = "6mo", interval = "1d"): Promise<IndicatorSnapshot> {
  const { data } = await client.get<IndicatorSnapshot>(`/research/indicators/${ticker}`, {
    params: { period, interval }
  });
  return data;
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

export async function getFavoriteStocks(): Promise<string[]> {
  const { data } = await client.get<{ favorites?: string[] }>("/api/user/favorites");
  return Array.isArray(data.favorites) ? data.favorites : [];
}

export async function setFavoriteStocks(symbols: string[]): Promise<string[]> {
  const { data } = await client.put<{ favorites?: string[] }>("/api/user/favorites", { symbols });
  return Array.isArray(data.favorites) ? data.favorites : [];
}

export type UserProfilePayload = {
  id?: string;
  email?: string;
  display_name?: string;
  avatar_url?: string;
  poke_enabled?: boolean;
  tutorial_completed?: boolean;
};

export async function getUserProfile(): Promise<{
  user_id: string | null;
  profile: UserProfilePayload | null;
  require_username_setup?: boolean;
  username_locked?: boolean;
}> {
  const { data } = await client.get<{
    user_id: string | null;
    profile: UserProfilePayload | null;
    require_username_setup?: boolean;
    username_locked?: boolean;
  }>("/api/user/profile");
  return data;
}

export async function updateUserPreferences(payload: {
  display_name?: string;
  avatar_data_url?: string;
  poke_enabled?: boolean;
  tutorial_completed?: boolean;
  watchlist?: string[];
  favorites?: string[];
}): Promise<{ ok: boolean; profile?: UserProfilePayload | null; require_username_setup?: boolean }> {
  try {
    const { data } = await client.patch<{ ok: boolean; profile?: UserProfilePayload | null; require_username_setup?: boolean }>(
      "/api/user/preferences",
      payload,
    );
    return data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const detail =
        (error.response?.data as { detail?: unknown; error?: unknown } | undefined)?.detail ??
        (error.response?.data as { detail?: unknown; error?: unknown } | undefined)?.error;
      if (typeof detail === "string" && detail.trim()) {
        throw new Error(detail.trim());
      }
    }
    throw error;
  }
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
  const symbol = ticker.trim().toUpperCase();
  const endpoints = [`/research/deep/${symbol}`, `/api/research/deep/${symbol}`];
  let lastError: unknown = null;

  for (const endpoint of endpoints) {
    try {
      const { data } = await client.post<DeepResearchResponse>(endpoint);
      return data;
    } catch (error) {
      lastError = error;
      if (axios.isAxiosError(error) && error.response?.status === 404) continue;
      if (axios.isAxiosError(error) && !error.response) continue;
      break;
    }
  }

  if (axios.isAxiosError(lastError) && !lastError.response) {
    throw networkErrorMessage("Deep research");
  }
  throw lastError instanceof Error ? lastError : new Error("Deep research request failed.");
}

export async function getMarketMovers(limit = 5): Promise<MarketMoversResponse> {
  const endpoints = ["/research/movers", "/api/research/movers"];
  let lastError: unknown = null;

  for (const endpoint of endpoints) {
    try {
      const { data } = await client.get<MarketMoversResponse>(endpoint, { params: { limit } });
      return data;
    } catch (error) {
      lastError = error;
      if (axios.isAxiosError(error) && error.response?.status === 404) continue;
      if (axios.isAxiosError(error) && !error.response) continue;
      break;
    }
  }

  throw lastError instanceof Error ? lastError : new Error("Market movers request failed.");
}

export async function askResearchQuery(payload: {
  prompt: string;
  ticker?: string;
  timeframe?: string;
  include_deep?: boolean;
  auto_fetch_if_missing?: boolean;
}): Promise<ResearchChatResponse> {
  const candidates = new Set<string>();
  candidates.add("/api/research/chat");
  candidates.add("/research/chat");
  candidates.add("/chat/research-query");
  candidates.add("/api/chat/research-query");

  try {
    const url = new URL(API_URL);
    const basePath = trimTrailingSlashes(url.pathname || "");
    const origin = url.origin;

    if (basePath && basePath !== "/") {
      candidates.add(`${origin}${basePath}/api/research/chat`);
      candidates.add(`${origin}${basePath}/research/chat`);
      candidates.add(`${origin}${basePath}/chat/research-query`);
      candidates.add(`${origin}${basePath}/api/chat/research-query`);
    }
    candidates.add(`${origin}/api/research/chat`);
    candidates.add(`${origin}/research/chat`);
    candidates.add(`${origin}/chat/research-query`);
    candidates.add(`${origin}/api/chat/research-query`);
  } catch {
    // API_URL can be relative in some setups.
  }

  const attempted: string[] = [];
  let last404: unknown = null;
  let lastNetwork: unknown = null;
  let lastError: unknown = null;

  for (const endpoint of candidates) {
    attempted.push(endpoint);
    try {
      const { data } = await client.post<ResearchChatResponse>(endpoint, payload);
      return data;
    } catch (err) {
      lastError = err;
      if (axios.isAxiosError(err)) {
        if (err.response?.status === 404 || err.response?.status === 405) {
          last404 = err;
          continue;
        }
        if (!err.response) {
          lastNetwork = err;
          continue;
        }
      }
      throw err;
    }
  }

  // Fallback: synthesize a chat answer from the standard research pipeline.
  const ticker = (payload.ticker || "AAPL").toUpperCase().trim();
  const timeframe = payload.timeframe || "7d";
  try {
    const analysis = await runResearch(ticker, timeframe);
    const recommendation = analysis.recommendation ? `Recommendation: ${analysis.recommendation}.` : "";
    const narratives = analysis.narratives
      .map((line) => line.trim())
      .filter(Boolean)
      .slice(0, 6)
      .map((line) => `- ${line}`)
      .join("\n");
    const sources = analysis.source_breakdown
      .map((entry) => `${entry.source}: ${entry.summary}`.trim())
      .filter(Boolean)
      .slice(0, 3)
      .map((line) => `- ${line}`)
      .join("\n");
    const fallbackBody = narratives || sources || "No additional structured context available.";
    return {
      ticker,
      response:
        `Dedicated chat endpoint is unavailable, but I pulled the latest ${ticker} research context.\n` +
        `${recommendation ? `${recommendation}\n` : ""}` +
        `${fallbackBody}`,
      model: "research-fallback",
      generated_at: new Date().toISOString(),
      context_refreshed: true,
      sources: ["fallback_run_research"],
    };
  } catch (fallbackError) {
    if (last404) {
      throw new Error(
        `Research chat route mismatch. Tried: ${attempted.join(", ")}. ` +
        "No compatible chat endpoint is available on this backend."
      );
    }
    if (lastNetwork || (axios.isAxiosError(fallbackError) && !fallbackError.response)) {
      throw networkErrorMessage("Research chat");
    }
    throw fallbackError instanceof Error
      ? fallbackError
      : lastError instanceof Error
        ? lastError
        : new Error("Research chat request failed.");
  }
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
