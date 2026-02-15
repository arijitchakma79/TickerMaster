import { useEffect, useMemo, useRef, useState } from "react";
import {
  askResearchQuery,
  fetchAdvancedStockData,
  fetchCandles,
  fetchIndicatorSnapshot,
  getAgentActivity,
  runDeepResearch,
  runResearch,
  searchTickerDirectory
} from "../lib/api";
import { formatCompactNumber, formatPercent, formatToCents } from "../lib/format";
import { resolveTickerCandidate } from "../lib/tickerInput";
import type { AgentActivity, AdvancedStockData, CandlePoint, DeepResearchResponse, IndicatorSnapshot, ResearchResponse, TickerLookup, WSMessage } from "../lib/types";
import StockChart, { type ChartOverlayIndicator } from "./StockChart";

interface Props {
  activeTicker: string;
  onTickerChange: (ticker: string) => void;
  connected: boolean;
  events: WSMessage[];
}

function normalizeSymbol(value: string) {
  return value.trim().toUpperCase().replace(/\./g, "-");
}

function containsTickerToken(action: string, ticker: string) {
  if (!ticker) return true;
  const tokens = action
    .toUpperCase()
    .replace(/\./g, "-")
    .split(/[^A-Z0-9-]+/)
    .filter(Boolean);
  return tokens.includes(ticker);
}

function activityTime(item: AgentActivity) {
  const raw = item.created_at ?? item.timestamp;
  if (!raw) return 0;
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? parsed : 0;
}

function normalizeActivity(raw: AgentActivity | WSMessage): AgentActivity | null {
  if (!raw || typeof raw !== "object") return null;
  const action = typeof raw.action === "string" ? raw.action : "";
  const agentName = typeof raw.agent_name === "string" ? raw.agent_name : "";
  const channel = typeof raw.channel === "string" ? raw.channel : undefined;
  const type = typeof raw.type === "string" ? raw.type : undefined;
  const isAgentPayload = Boolean(action || agentName || type === "agent_activity" || channel === "agents");
  if (!isAgentPayload) return null;
  return {
    module: typeof raw.module === "string" ? raw.module : undefined,
    agent_name: agentName || undefined,
    action: action || undefined,
    status: typeof raw.status === "string" ? raw.status : undefined,
    details: raw.details && typeof raw.details === "object" ? (raw.details as Record<string, unknown>) : undefined,
    created_at: typeof raw.created_at === "string" ? raw.created_at : undefined,
    timestamp: typeof raw.timestamp === "string" ? raw.timestamp : undefined,
    channel,
    type,
  };
}

function activityMatchesTicker(item: AgentActivity, ticker: string) {
  if (!ticker) return true;
  const details = item.details;
  if (details && typeof details === "object") {
    const detailSymbol = (details as Record<string, unknown>).symbol;
    if (typeof detailSymbol === "string" && normalizeSymbol(detailSymbol) === ticker) return true;
  }
  if (typeof item.action === "string" && containsTickerToken(item.action, ticker)) return true;
  return false;
}

function activityKey(item: AgentActivity) {
  return [
    item.created_at ?? item.timestamp ?? "",
    item.agent_name ?? "",
    item.action ?? "",
    item.status ?? "",
  ].join("|");
}

function statusClass(status: string | undefined) {
  const value = (status ?? "running").toLowerCase();
  if (value === "success" || value === "error" || value === "pending") return value;
  return "running";
}

function sentimentScore100(score: number) {
  const raw = ((score + 1) / 2) * 100;
  return Math.max(0, Math.min(100, Math.round(raw)));
}

function sentimentLabelDetailed(score: number) {
  const value = sentimentScore100(score);
  if (value <= 15) return "Strong Bearish";
  if (value <= 35) return "Bearish";
  if (value <= 45) return "Slight Bearish";
  if (value <= 55) return "Neutral";
  if (value <= 65) return "Slight Bullish";
  if (value <= 85) return "Bullish";
  return "Strong Bullish";
}

function sentimentToneClass(score: number): "bullish" | "neutral" | "bearish" {
  if (score >= 0.2) return "bullish";
  if (score <= -0.2) return "bearish";
  return "neutral";
}

type SummarySection = {
  heading: string;
  bullets: string[];
  text: string;
};

type IndicatorKey =
  | ChartOverlayIndicator
  | "rsi14"
  | "macd_line"
  | "macd_signal"
  | "macd_hist"
  | "atr14";

const INDICATOR_OPTIONS: Array<{ key: IndicatorKey; label: string; group: "trend" | "momentum" | "volatility" }> = [
  { key: "sma20", label: "SMA 20", group: "trend" },
  { key: "sma50", label: "SMA 50", group: "trend" },
  { key: "sma200", label: "SMA 200", group: "trend" },
  { key: "ema21", label: "EMA 21", group: "trend" },
  { key: "ema50", label: "EMA 50", group: "trend" },
  { key: "vwap", label: "VWAP", group: "trend" },
  { key: "bbands", label: "Bollinger Bands", group: "volatility" },
  { key: "rsi14", label: "RSI 14", group: "momentum" },
  { key: "macd_line", label: "MACD Line", group: "momentum" },
  { key: "macd_signal", label: "MACD Signal", group: "momentum" },
  { key: "macd_hist", label: "MACD Histogram", group: "momentum" },
  { key: "atr14", label: "ATR 14", group: "volatility" },
];

const DEFAULT_INDICATORS: Record<IndicatorKey, boolean> = {
  sma20: false,
  sma50: false,
  sma200: false,
  ema21: false,
  ema50: false,
  vwap: false,
  bbands: false,
  rsi14: false,
  macd_line: false,
  macd_signal: false,
  macd_hist: false,
  atr14: false,
};

function parseSummarySections(raw: string): SummarySection[] {
  const summary = raw.trim();
  if (!summary) return [];

  const headingRegex = /\*\*([^*]+)\*\*:\s*/g;
  const matches = Array.from(summary.matchAll(headingRegex));
  const normalize = (value: string) =>
    value
      .replace(/\[\[(.*?)\]\]/g, "[$1]")
      .replace(/\s+/g, " ")
      .trim();

  const extractBullets = (block: string) => {
    const normalized = normalize(block);
    const chunks = normalized.split(/\s+-\s+/).map((item) => item.trim()).filter(Boolean);
    if (chunks.length <= 1) return [];
    return chunks.map((item, idx) => (idx === 0 ? item.replace(/^[^-]+?:\s*/, "") : item));
  };

  if (matches.length === 0) {
    const bullets = extractBullets(summary);
    return [
      {
        heading: "Summary",
        bullets,
        text: bullets.length > 0 ? "" : normalize(summary.replace(/\*\*/g, "")),
      },
    ];
  }

  const sections: SummarySection[] = [];
  for (let idx = 0; idx < matches.length; idx += 1) {
    const current = matches[idx];
    const next = matches[idx + 1];
    const heading = (current[1] ?? "Summary").trim();
    const start = (current.index ?? 0) + current[0].length;
    const end = next?.index ?? summary.length;
    const block = summary.slice(start, end).trim().replace(/\*\*/g, "");
    const bullets = extractBullets(block);
    const cleanedHeading = heading.replace(/^summary\s*:/i, "").replace(/^summary$/i, "Summary").trim();
    sections.push({
      heading: cleanedHeading || "Summary",
      bullets,
      text: bullets.length > 0 ? "" : normalize(block),
    });
  }
  return sections;
}

function formatIndicatorValue(snapshot: IndicatorSnapshot, key: IndicatorKey): string {
  if (key === "bbands") {
    const upper = snapshot.latest.bb_upper;
    const mid = snapshot.latest.bb_mid;
    const lower = snapshot.latest.bb_lower;
    if ([upper, mid, lower].every((val) => typeof val === "number")) {
      return `${(upper as number).toFixed(2)} / ${(mid as number).toFixed(2)} / ${(lower as number).toFixed(2)}`;
    }
    return "-";
  }
  const value = snapshot.latest[key as keyof IndicatorSnapshot["latest"]];
  return typeof value === "number" ? value.toFixed(2) : "-";
}

export default function ResearchPanel({ activeTicker, onTickerChange, connected, events }: Props) {
  const [infoOpen, setInfoOpen] = useState(false);
  const [indicatorInfoOpen, setIndicatorInfoOpen] = useState(false);
  const [insiderPage, setInsiderPage] = useState(1);
  const [tickerInput, setTickerInput] = useState(activeTicker);
  const [tickerInputFocused, setTickerInputFocused] = useState(false);
  const [tickerSuggestions, setTickerSuggestions] = useState<TickerLookup[]>([]);
  const [tickerSearchLoading, setTickerSearchLoading] = useState(false);
  const [timeframe, setTimeframe] = useState("7d");
  const [loading, setLoading] = useState(false);
  const [research, setResearch] = useState<ResearchResponse | null>(null);
  const [candles, setCandles] = useState<CandlePoint[]>([]);
  const [advanced, setAdvanced] = useState<AdvancedStockData | null>(null);
  const [deepResearch, setDeepResearch] = useState<DeepResearchResponse | null>(null);
  const [deepLoading, setDeepLoading] = useState(false);
  const [agentProgress, setAgentProgress] = useState(0);
  const [chartMode, setChartMode] = useState<"candles" | "line">("candles");
  const [chartPeriod, setChartPeriod] = useState("6mo");
  const [chartInterval, setChartInterval] = useState("1d");
  const [selectedIndicators, setSelectedIndicators] = useState<Record<IndicatorKey, boolean>>(DEFAULT_INDICATORS);
  const [indicatorSnapshot, setIndicatorSnapshot] = useState<IndicatorSnapshot | null>(null);
  const [agentHistory, setAgentHistory] = useState<AgentActivity[]>([]);
  const [error, setError] = useState("");
  const [chatModalOpen, setChatModalOpen] = useState(false);
  const [chatPrompt, setChatPrompt] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatLog, setChatLog] = useState<Array<{
    role: "user" | "assistant";
    text: string;
    meta?: string;
  }>>([]);
  const chatLogRef = useRef<HTMLDivElement | null>(null);
  const chatSubmitLockRef = useRef(false);
  const normalizedTicker = useMemo(() => normalizeSymbol(activeTicker), [activeTicker]);
  const hasSelectedTicker = activeTicker.trim().length > 0;

  const sentimentLabel = useMemo(() => {
    if (!research) return "-";
    const value = sentimentScore100(research.aggregate_sentiment);
    return `${sentimentLabelDetailed(research.aggregate_sentiment)} · ${value}/100`;
  }, [research]);

  const insiderRows = advanced?.insider_transactions ?? [];
  const insiderPageSize = 8;
  const insiderTotalPages = Math.max(1, Math.ceil(insiderRows.length / insiderPageSize));
  const insiderPageSafe = Math.min(insiderPage, insiderTotalPages);
  const insiderSliceStart = (insiderPageSafe - 1) * insiderPageSize;
  const insiderPageRows = insiderRows.slice(insiderSliceStart, insiderSliceStart + insiderPageSize);

  const liveWireActivity = useMemo(() => {
    const merged = new Map<string, AgentActivity>();

    for (const candidate of [...events, ...agentHistory]) {
      const normalized = normalizeActivity(candidate);
      if (!normalized) continue;
      if (!activityMatchesTicker(normalized, normalizedTicker)) continue;
      const key = activityKey(normalized);
      if (!merged.has(key)) {
        merged.set(key, normalized);
      }
    }

    return Array.from(merged.values())
      .sort((a, b) => activityTime(b) - activityTime(a))
      .slice(0, 24);
  }, [agentHistory, events, normalizedTicker]);

  useEffect(() => {
    const query = tickerInput.trim();
    if (!query) {
      setTickerSuggestions([]);
      setTickerSearchLoading(false);
      return;
    }

    let active = true;
    const timer = window.setTimeout(() => {
      setTickerSearchLoading(true);
      void searchTickerDirectory(query, 8)
        .then((results) => {
          if (!active) return;
          setTickerSuggestions(results);
        })
        .catch(() => {
          if (!active) return;
          setTickerSuggestions([]);
        })
        .finally(() => {
          if (!active) return;
          setTickerSearchLoading(false);
        });
    }, 180);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [tickerInput]);

  useEffect(() => {
    const normalized = activeTicker.trim().toUpperCase();
    if (!normalized) return;
    setTickerInput((prev) => (prev === normalized ? prev : normalized));
    setInsiderPage(1);
  }, [activeTicker]);

  useEffect(() => {
    const ticker = activeTicker.trim().toUpperCase();
    if (!ticker) return;
    let active = true;

    void fetchCandles(ticker, chartPeriod, chartInterval)
      .then((chartPoints) => {
        if (!active) return;
        setCandles(chartPoints);
      })
      .catch(() => {
        if (!active) return;
      });

    return () => {
      active = false;
    };
  }, [activeTicker, chartPeriod, chartInterval]);

  useEffect(() => {
    const ticker = activeTicker.trim().toUpperCase();
    if (!ticker) {
      setIndicatorSnapshot(null);
      return;
    }
    let active = true;
    void fetchIndicatorSnapshot(ticker, chartPeriod, chartInterval)
      .then((snapshot) => {
        if (!active) return;
        setIndicatorSnapshot(snapshot);
      })
      .catch(() => {
        if (!active) return;
        setIndicatorSnapshot(null);
      });

    return () => {
      active = false;
    };
  }, [activeTicker, chartPeriod, chartInterval]);

  useEffect(() => {
    const ticker = activeTicker.trim().toUpperCase();
    if (!ticker) return;
    let active = true;

    void fetchAdvancedStockData(ticker)
      .then((snapshot) => {
        if (!active) return;
        setAdvanced(snapshot);
      })
      .catch(() => {
        if (!active) return;
      });

    return () => {
      active = false;
    };
  }, [activeTicker]);

  useEffect(() => {
    let active = true;

    const loadActivity = async () => {
      try {
        const items = await getAgentActivity(120);
        if (!active) return;
        setAgentHistory(items);
      } catch {
        if (!active) return;
        setAgentHistory([]);
      }
    };

    void loadActivity();
    const poll = window.setInterval(() => {
      void loadActivity();
    }, 15000);

    return () => {
      active = false;
      window.clearInterval(poll);
    };
  }, [normalizedTicker]);

  async function handleAnalyze(rawInput?: string) {
    setLoading(true);
    setError("");
    const seed = rawInput ?? tickerInput;
    const ticker = resolveTickerCandidate(seed, tickerSuggestions);
    if (!ticker) {
      setError("Enter a ticker or company name.");
      setLoading(false);
      return;
    }
    setTickerInput(ticker);
    setTickerInputFocused(false);
    onTickerChange(ticker);
    const runNonce = runNonceRef.current + 1;
    runNonceRef.current = runNonce;
    try {
      const [analysisResult, chartResult, advancedResult, indicatorResult] = await Promise.allSettled([
        runResearch(ticker, timeframe),
        fetchCandles(ticker, chartPeriod, chartInterval, true),
        fetchAdvancedStockData(ticker),
        fetchIndicatorSnapshot(ticker, chartPeriod, chartInterval),
      ]);
      if (runNonce !== runNonceRef.current) return;
      if (analysisResult.status === "rejected") {
        throw analysisResult.reason;
      }
      setResearch(analysisResult.value);
      if (chartResult.status === "fulfilled") setCandles(chartResult.value);
      if (advancedResult.status === "fulfilled") setAdvanced(advancedResult.value);
      if (indicatorResult.status === "fulfilled") setIndicatorSnapshot(indicatorResult.value);
    } catch (err) {
      if (runNonce !== runNonceRef.current) return;
      setError(err instanceof Error ? err.message : "Research request failed");
    } finally {
      if (runNonce !== runNonceRef.current) return;
      setAgentProgress(100);
      setLoading(false);
    }
  }

  async function handleDeepResearch() {
    setLoading(true);
    setDeepLoading(true);
    setError("");
    const ticker = resolveTickerCandidate(tickerInput, tickerSuggestions);
    if (!ticker) {
      setError("Enter a ticker or company name.");
      setLoading(false);
      setDeepLoading(false);
      return;
    }
    setTickerInput(ticker);
    setTickerInputFocused(false);
    onTickerChange(ticker);
    const runNonce = runNonceRef.current + 1;
    runNonceRef.current = runNonce;
    try {
      const [analysisResult, chartResult, advancedResult, indicatorResult, deepResult] = await Promise.allSettled([
        runResearch(ticker, timeframe),
        fetchCandles(ticker, chartPeriod, chartInterval, true),
        fetchAdvancedStockData(ticker),
        fetchIndicatorSnapshot(ticker, chartPeriod, chartInterval),
        runDeepResearch(ticker),
      ]);
      if (runNonce !== runNonceRef.current) return;
      if (analysisResult.status === "rejected") {
        throw analysisResult.reason;
      }
      setResearch(analysisResult.value);
      if (chartResult.status === "fulfilled") setCandles(chartResult.value);
      if (advancedResult.status === "fulfilled") setAdvanced(advancedResult.value);
      if (indicatorResult.status === "fulfilled") setIndicatorSnapshot(indicatorResult.value);
      if (deepResult.status === "fulfilled") {
        setDeepResearch(deepResult.value);
      } else {
        setDeepResearch(null);
      }
    } catch (err) {
      if (runNonce !== runNonceRef.current) return;
      setError(err instanceof Error ? err.message : "Deep research request failed");
    } finally {
      if (runNonce !== runNonceRef.current) return;
      setAgentProgress(100);
      setLoading(false);
      setDeepLoading(false);
    }
  }

  const sourceRows = research?.source_breakdown ?? [];
  const perplexityEntry = sourceRows.find((entry) => entry.source === "Perplexity Sonar") ?? null;
  const xEntry = sourceRows.find((entry) => entry.source === "X API") ?? null;
  const redditEntry = sourceRows.find((entry) => entry.source === "Reddit Subreddit Scrape") ?? null;
  const agentRunning = loading || deepLoading;
  const runNonceRef = useRef(0);

  useEffect(() => {
    if (!agentRunning) {
      setAgentProgress(0);
      return;
    }
    setAgentProgress((prev) => (prev > 6 ? prev : 6));
    const timer = window.setInterval(() => {
      setAgentProgress((prev) => {
        if (prev >= 92) return prev;
        const delta = Math.max(0.35, (92 - prev) * 0.06);
        return Math.min(92, prev + delta);
      });
    }, 250);
    return () => window.clearInterval(timer);
  }, [agentRunning]);

  function handleForceQuitRun() {
    runNonceRef.current += 1;
    setLoading(false);
    setDeepLoading(false);
    setAgentProgress(0);
    setError("Run stopped by user.");
  }

  function handleClearChat() {
    setChatLog([]);
    setChatPrompt("");
  }

  async function handleResearchChatSubmit(promptOverride?: string) {
    if (!hasSelectedTicker) return;
    const prompt = (promptOverride ?? chatPrompt).trim();
    if (!prompt || chatLoading || chatSubmitLockRef.current) return;
    chatSubmitLockRef.current = true;
    const seedTicker = activeTicker.trim().toUpperCase() || resolveTickerCandidate(tickerInput, tickerSuggestions);
    setChatModalOpen(true);
    setChatLog((prev) => [...prev, { role: "user", text: prompt }]);
    setChatPrompt("");
    setChatLoading(true);
    try {
      const out = await askResearchQuery({
        prompt,
        ticker: seedTicker || undefined,
        timeframe,
        include_deep: true,
        auto_fetch_if_missing: true,
      });
      setChatLog((prev) => [
        ...prev,
        {
          role: "assistant",
          text: out.response,
        },
      ]);
    } catch (err) {
      setChatLog((prev) => [
        ...prev,
        {
          role: "assistant",
          text: err instanceof Error ? err.message : "Research chat request failed.",
          meta: "error",
        },
      ]);
    } finally {
      chatSubmitLockRef.current = false;
      setChatLoading(false);
    }
  }

  useEffect(() => {
    if (!chatModalOpen) return;
    const frame = window.requestAnimationFrame(() => {
      if (!chatLogRef.current) return;
      chatLogRef.current.scrollTop = chatLogRef.current.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [chatModalOpen, chatLog, chatLoading, liveWireActivity.length]);

  return (
    <section className="panel stack stagger">
      <header className="panel-header">
        <div className="panel-title-row">
          <h2>Research</h2>
          <button type="button" className="panel-info-btn" onClick={() => setInfoOpen(true)} aria-label="Research panel info">
            i
          </button>
        </div>
        <div className="research-stack-badges" aria-label="Research data stack">
          <span className="research-stack-badge">Perplexity Sonar</span>
          <span className="research-stack-badge">X Signals</span>
          <span className="research-stack-badge">Reddit Flow</span>
          <span className="research-stack-badge">Prediction Markets</span>
        </div>
      </header>
      {infoOpen ? (
        <div className="panel-info-backdrop" onClick={() => setInfoOpen(false)}>
          <div className="panel-info-modal" onClick={(event) => event.stopPropagation()}>
            <h3>Research Workbench</h3>
            <p>Use this panel to run AI synthesis across Perplexity, X, Reddit, and prediction markets.</p>
            <ul>
              <li>Enter ticker or company, then run research.</li>
              <li>Review source breakdown, market signals, and insider data.</li>
              <li>Use Deep Research when you need extended context.</li>
            </ul>
            <button className="secondary" onClick={() => setInfoOpen(false)}>Close</button>
          </div>
        </div>
      ) : null}
      {indicatorInfoOpen ? (
        <div className="panel-info-backdrop" onClick={() => setIndicatorInfoOpen(false)}>
          <div className="panel-info-modal" onClick={(event) => event.stopPropagation()}>
            <div className="panel-header">
              <h3>Pro Indicator Snapshot</h3>
              <span className="muted">
                {indicatorSnapshot ? `${indicatorSnapshot.period} / ${indicatorSnapshot.interval}` : `${chartPeriod} / ${chartInterval}`}
              </span>
            </div>
            <div className="kpi-grid">
              {indicatorSnapshot ? (
                INDICATOR_OPTIONS.filter((item) => selectedIndicators[item.key]).map((item) => (
                  <div key={`modal-ind-${item.key}`}>
                    <p className="muted">{item.label}</p>
                    <h3>{formatIndicatorValue(indicatorSnapshot, item.key)}</h3>
                  </div>
                ))
              ) : null}
              {INDICATOR_OPTIONS.every((item) => !selectedIndicators[item.key]) ? (
                <div>
                  <p className="muted">No indicators selected</p>
                  <h3>-</h3>
                </div>
              ) : null}
            </div>
            <p className="muted">
              Commonly used by professional desks: trend (SMA/EMA/VWAP), momentum (RSI/MACD), and volatility (ATR/Bollinger).
            </p>
            <div className="source-ai-summary">
              <section className="source-summary-section">
                <h5 className="source-summary-heading">Trend</h5>
                <ul className="source-summary-list">
                  <li>SMA 20/50/200: short, medium, and long trend baselines.</li>
                  <li>EMA 21/50: reacts faster to trend changes.</li>
                  <li>VWAP: institutional execution and mean-reversion anchor.</li>
                </ul>
              </section>
              <section className="source-summary-section">
                <h5 className="source-summary-heading">Momentum</h5>
                <ul className="source-summary-list">
                  <li>RSI 14: overbought/oversold and momentum shifts.</li>
                  <li>MACD line/signal/hist: direction and acceleration.</li>
                </ul>
              </section>
              <section className="source-summary-section">
                <h5 className="source-summary-heading">Volatility</h5>
                <ul className="source-summary-list">
                  <li>Bollinger Bands: compression/expansion and breakouts.</li>
                  <li>ATR 14: normalized volatility and stop-sizing context.</li>
                </ul>
              </section>
            </div>
            <button className="secondary" onClick={() => setIndicatorInfoOpen(false)}>Close</button>
          </div>
        </div>
      ) : null}
      {agentRunning ? (
        <div className="panel-info-backdrop live-wire-backdrop">
          <div className="panel-info-modal live-wire-modal" onClick={(event) => event.stopPropagation()}>
            <div className="panel-header">
              <h3>Research Agent Running</h3>
              <span className="live-wire-running-chip">
                <span className="live-wire-running-dot" />
                {deepLoading ? "Deep Research" : "Research"}
              </span>
            </div>
            <div className="run-progress-wrap" aria-label="Research run progress">
              <div className="run-progress-track">
                <div className="run-progress-fill" style={{ width: `${Math.max(6, Math.round(agentProgress))}%` }} />
              </div>
              <span className="muted run-progress-label">{Math.max(6, Math.round(agentProgress))}%</span>
            </div>
            <p className="muted">Live Wire is streaming agent steps. Screen unlocks automatically when run completes.</p>
            <div className="panel-header">
              <h4>Live Wire</h4>
              <span className={connected ? "dot dot-live" : "dot dot-offline"}>
                {connected ? "Socket Live" : "Socket Offline"}
              </span>
            </div>
            <div className="event-list live-wire-overlay-list">
              {liveWireActivity.length === 0 ? <p className="muted">Waiting for agent activity...</p> : null}
              {liveWireActivity.map((event, idx) => (
                <article key={`${activityKey(event)}-overlay-${idx}`} className="event-item">
                  <div className="event-meta">
                    <strong>{event.agent_name ?? "AI Agent"}</strong>
                    <span>{event.module ?? "research"}</span>
                  </div>
                  <p className="live-wire-action">{event.action ?? "Processing request..."}</p>
                  <div className="live-wire-footer">
                    <span className={`live-wire-status ${statusClass(event.status)}`}>{(event.status ?? "running").toLowerCase()}</span>
                    {event.created_at || event.timestamp ? (
                      <time>{new Date(event.created_at ?? event.timestamp ?? "").toLocaleTimeString()}</time>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>
            <div className="run-overlay-actions">
              <button type="button" className="secondary force-quit-btn" onClick={handleForceQuitRun}>
                Force Quit
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {chatModalOpen ? (
        <div className="panel-info-backdrop research-chat-backdrop" onClick={() => setChatModalOpen(false)}>
          <div className="panel-info-modal research-chat-modal" onClick={(event) => event.stopPropagation()}>
            <div className="panel-header">
              <h3>Research Chat</h3>
              <div className="row-actions">
                <button type="button" className="secondary" onClick={handleClearChat} disabled={chatLoading || chatLog.length === 0}>
                  Clear chat
                </button>
                <button type="button" className="secondary" onClick={() => setChatModalOpen(false)}>Close</button>
              </div>
            </div>
            <div className="research-chat-layout">
              <div className="research-chat-log research-chat-modal-log" ref={chatLogRef}>
                {chatLog.length === 0 ? <p className="muted">{hasSelectedTicker ? "Ask a stock-specific question about this ticker." : "Select a stock first to use Research Chat."}</p> : null}
                {chatLog.map((item, idx) => (
                  <article key={`research-chat-modal-${idx}`} className={item.role === "user" ? "research-chat-item user" : "research-chat-item assistant"}>
                    <p>{item.text}</p>
                    {item.meta ? <span className="muted">{item.meta}</span> : null}
                  </article>
                ))}
                {chatLoading ? (
                  <article className="research-chat-item assistant research-chat-thinking">
                    <p>Running analysis and backend tools...</p>
                  </article>
                ) : null}
              </div>
              <aside className="research-chat-toolstream">
                <div className="panel-header">
                  <h4>Live Wire</h4>
                  <span className={connected ? "dot dot-live" : "dot dot-offline"}>
                    {connected ? "Socket Live" : "Socket Offline"}
                  </span>
                </div>
                <div className="research-chat-toolstream-list">
                  {chatLoading && liveWireActivity.length === 0 ? <p className="muted">Spinning up agents...</p> : null}
                  {!chatLoading && liveWireActivity.length === 0 ? <p className="muted">Activity appears here when tools run.</p> : null}
                  {liveWireActivity.slice(0, 12).map((event, idx) => (
                    <article key={`chat-tool-${activityKey(event)}-${idx}`} className="event-item">
                      <div className="event-meta">
                        <strong>{event.agent_name ?? "AI Agent"}</strong>
                        <span>{event.module ?? "research"}</span>
                      </div>
                      <p className="live-wire-action">{event.action ?? "Processing..."}</p>
                      <div className="live-wire-footer">
                        <span className={`live-wire-status ${statusClass(event.status)}`}>{(event.status ?? "running").toLowerCase()}</span>
                      </div>
                    </article>
                  ))}
                </div>
              </aside>
            </div>
            <div className="research-chat-compose">
              <textarea
                value={chatPrompt}
                rows={3}
                placeholder={hasSelectedTicker ? `Ask about ${activeTicker}...` : "Select a stock first..."}
                disabled={!hasSelectedTicker}
                onChange={(event) => setChatPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key !== "Enter" || event.shiftKey) return;
                  event.preventDefault();
                  void handleResearchChatSubmit();
                }}
              />
              <button onClick={() => void handleResearchChatSubmit()} disabled={!hasSelectedTicker || chatLoading || chatPrompt.trim().length === 0}>
                {chatLoading ? "Querying…" : "Ask"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <div className="research-controls-grid research-actions-card">
        <label className="research-ticker-field research-control">
          Ticker
          <div className="ticker-autocomplete">
            <input
              value={tickerInput}
              onFocus={() => setTickerInputFocused(true)}
              onBlur={() => setTimeout(() => setTickerInputFocused(false), 120)}
              onChange={(event) => setTickerInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key !== "Enter") return;
                event.preventDefault();
                const topSuggestion = tickerSuggestions[0]?.ticker;
                void handleAnalyze(topSuggestion ?? tickerInput);
              }}
              maxLength={48}
              placeholder={`Ticker or company name (${activeTicker.toUpperCase()})`}
            />
            {tickerInputFocused && tickerInput.trim() ? (
              <div className="ticker-suggestions" role="listbox" aria-label="Ticker suggestions">
                {tickerSuggestions.length > 0 ? (
                  tickerSuggestions.map((entry) => (
                    <button
                      key={`research-${entry.ticker}-${entry.name}`}
                      type="button"
                      className="ticker-suggestion"
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => {
                        setTickerInput(entry.ticker);
                        setTickerInputFocused(false);
                      }}
                    >
                      <span className="ticker-suggestion-symbol">{entry.ticker}</span>
                      <span className="ticker-suggestion-name">{entry.name}</span>
                    </button>
                  ))
                ) : (
                  <p className="ticker-suggestion-empty">{tickerSearchLoading ? "Searching…" : "No matches found"}</p>
                )}
              </div>
            ) : null}
          </div>
        </label>
        <label className="research-control">
          Timeframe
          <select value={timeframe} onChange={(event) => setTimeframe(event.target.value)}>
            <option value="24h">24h</option>
            <option value="7d">7d</option>
            <option value="30d">30d</option>
            <option value="60d">60d</option>
            <option value="90d">90d</option>
            <option value="180d">180d</option>
            <option value="1y">1y</option>
            <option value="2y">2y</option>
            <option value="5y">5y</option>
            <option value="10y">10y</option>
            <option value="max">max</option>
          </select>
        </label>
        <label className="research-control">
          Chart Type
          <select value={chartMode} onChange={(event) => setChartMode(event.target.value as "candles" | "line")}>
            <option value="candles">Candlesticks</option>
            <option value="line">Line</option>
          </select>
        </label>
        <label className="research-control">
          Chart Range
          <select value={chartPeriod} onChange={(event) => setChartPeriod(event.target.value)}>
            <option value="5d">5d</option>
            <option value="1mo">1mo</option>
            <option value="3mo">3mo</option>
            <option value="6mo">6mo</option>
            <option value="1y">1y</option>
            <option value="2y">2y</option>
            <option value="5y">5y</option>
            <option value="10y">10y</option>
            <option value="max">max</option>
          </select>
        </label>
        <label className="research-control">
          Interval
          <select value={chartInterval} onChange={(event) => setChartInterval(event.target.value)}>
            <option value="1d">1d</option>
            <option value="1wk">1wk</option>
            <option value="1mo">1mo</option>
          </select>
        </label>
        <label className="research-control research-control-indicators">
          <span className="indicator-label-row">
            Indicators
            <button
              type="button"
              className="panel-info-btn indicator-info-btn"
              onClick={() => setIndicatorInfoOpen(true)}
              aria-label="Indicators guide"
            >
              i
            </button>
          </span>
          <div className="indicator-chip-grid">
            {INDICATOR_OPTIONS.map((item) => (
              <button
                key={item.key}
                type="button"
                className={selectedIndicators[item.key] ? "indicator-chip active" : "indicator-chip"}
                onClick={() =>
                  setSelectedIndicators((prev) => ({
                    ...prev,
                    [item.key]: !prev[item.key],
                  }))
                }
              >
                {item.label}
              </button>
            ))}
          </div>
        </label>
        <div className="research-action-buttons">
          <button onClick={() => void handleAnalyze()} disabled={loading}>
            {loading ? "Analyzing…" : "Run Research"}
          </button>
          <button className="secondary" onClick={handleDeepResearch} disabled={deepLoading}>
            {deepLoading ? "Deep researching…" : "Deep Research"}
          </button>
        </div>
      </div>

      {error ? <p className="error">{error}</p> : null}

      <div className="glass-card stack">
        <div className="panel-header">
          <h3>Research Chat</h3>
          <span className="muted">Popup assistant with live backend tool activity.</span>
        </div>
        <div className="research-chat-compose">
          <textarea
            value={chatPrompt}
            rows={2}
            placeholder={hasSelectedTicker ? `Ask about ${activeTicker}...` : "Select a stock first..."}
            disabled={!hasSelectedTicker}
            onFocus={() => {
              if (!hasSelectedTicker) return;
              setChatModalOpen(true);
            }}
            onChange={(event) => setChatPrompt(event.target.value)}
            onKeyDown={(event) => {
              if (event.key !== "Enter" || event.shiftKey) return;
              event.preventDefault();
              void handleResearchChatSubmit();
            }}
          />
          <button onClick={() => setChatModalOpen(true)} disabled={!hasSelectedTicker || chatLoading}>
            {chatLoading ? "Querying…" : "Open Chat"}
          </button>
        </div>
      </div>

      <div className="glass-card kpi-grid">
        <div>
          <p className="muted">Aggregate Signal</p>
          <h3>{sentimentLabel}</h3>
        </div>
        <div>
          <p className="muted">Narratives</p>
          <h3>{research?.narratives.length ?? 0}</h3>
        </div>
        <div>
          <p className="muted">Prediction Markets</p>
          <h3>{research?.prediction_markets.length ?? 0}</h3>
        </div>
      </div>

      <div className="glass-card">
        <div className="panel-header">
          <h3>{activeTicker} Price Action</h3>
          <span className="muted">{chartPeriod} / {chartInterval}</span>
        </div>
        <StockChart
          points={candles}
          mode={chartMode}
          indicators={{
            sma20: selectedIndicators.sma20,
            sma50: selectedIndicators.sma50,
            sma200: selectedIndicators.sma200,
            ema21: selectedIndicators.ema21,
            ema50: selectedIndicators.ema50,
            vwap: selectedIndicators.vwap,
            bbands: selectedIndicators.bbands,
          }}
        />
      </div>

      {advanced ? (
        <div className="glass-card">
          <div className="panel-header">
            <h3>{advanced.company_name ?? activeTicker} Snapshot</h3>
            <span className="muted">{advanced.exchange ?? "-"}</span>
          </div>
          <div className="kpi-grid">
            <div><p className="muted">Market Cap</p><h3>{formatCompactNumber(advanced.market_cap)}</h3></div>
            <div><p className="muted">Trailing P/E</p><h3>{formatToCents(advanced.trailing_pe)}</h3></div>
            <div><p className="muted">Forward P/E</p><h3>{formatToCents(advanced.forward_pe)}</h3></div>
            <div><p className="muted">EPS (TTM)</p><h3>{formatToCents(advanced.eps_trailing)}</h3></div>
            <div><p className="muted">Target Price</p><h3>{formatToCents(advanced.target_mean_price)}</h3></div>
            <div><p className="muted">Recommendation</p><h3>{advanced.recommendation ?? "-"}</h3></div>
          </div>
          <p className="muted" style={{ marginTop: "0.75rem" }}>
            {advanced.sector ?? "-"} / {advanced.industry ?? "-"}
          </p>
          {advanced.description ? <p>{advanced.description}</p> : null}
        </div>
      ) : null}

      <div className="card-row card-row-split research-resource-row">
        <div className="glass-card">
          <h3>Perplexity Sonar</h3>
          <div className="stack small-gap">
            {perplexityEntry ? (
              <article className="source-item">
                <div className="source-title source-title-top">
                  <strong>{perplexityEntry.source}</strong>
                  <span className={`pill ${sentimentToneClass(perplexityEntry.score)}`}>
                    {sentimentLabelDetailed(perplexityEntry.score)} · {sentimentScore100(perplexityEntry.score)}/100
                  </span>
                </div>
                <div className="source-ai-summary">
                  {parseSummarySections(perplexityEntry.summary).map((section, sectionIdx) => (
                    <section key={`${perplexityEntry.source}-${section.heading}-${sectionIdx}`} className="source-summary-section">
                      <h5 className="source-summary-heading">{section.heading}</h5>
                      {section.bullets.length > 0 ? (
                        <ul className="source-summary-list">
                          {section.bullets.map((point, pointIdx) => (
                            <li key={`${perplexityEntry.source}-${section.heading}-${pointIdx}`}>{point}</li>
                          ))}
                        </ul>
                      ) : (
                        <p>{section.text}</p>
                      )}
                    </section>
                  ))}
                </div>
                {perplexityEntry.links.length > 0 ? (
                  <div className="source-citations">
                    <span className="muted">Sources:</span>
                    {perplexityEntry.links.slice(0, 8).map((link) => (
                      <a key={`${perplexityEntry.source}-cite-${link.url}`} href={link.url} target="_blank" rel="noreferrer">
                        {link.title}
                      </a>
                    ))}
                  </div>
                ) : null}
              </article>
            ) : (
              <p className="muted">No Perplexity summary yet. Run research to load it.</p>
            )}
          </div>
        </div>

        <div className="glass-card">
          <h3>Other Resources</h3>
          <div className="stack small-gap">
            {[xEntry, redditEntry].filter((item): item is NonNullable<typeof item> => Boolean(item)).map((entry) => (
              <article key={entry.source} className="source-item">
                <div className="source-title source-title-top">
                  <strong>{entry.source}</strong>
                  <span className={`pill ${sentimentToneClass(entry.score)}`}>
                    {sentimentLabelDetailed(entry.score)} · {sentimentScore100(entry.score)}/100
                  </span>
                </div>
                <p>{entry.summary}</p>
                {entry.links.length > 0 ? (
                  <div className="source-citations">
                    {entry.links.slice(0, 4).map((link) => (
                      <a key={`${entry.source}-right-${link.url}`} href={link.url} target="_blank" rel="noreferrer">
                        {link.title}
                      </a>
                    ))}
                  </div>
                ) : null}
              </article>
            ))}
            {!xEntry && !redditEntry ? <p className="muted">No X/Reddit summaries yet.</p> : null}
          </div>
          <div className="source-item prediction-resource-box">
            <div className="panel-header">
              <h4>Prediction Markets</h4>
            </div>
            <div className="table-wrap prediction-table-wrap">
              <table className="prediction-table">
                <thead>
                  <tr>
                    <th>Source</th>
                    <th>Market</th>
                    <th>Signal</th>
                  </tr>
                </thead>
                <tbody>
                  {(research?.prediction_markets ?? []).slice(0, 3).map((market, idx) => (
                    <tr key={`market-${idx}`}>
                      <td>{String(market.source ?? "-")}{idx === 0 ? " (most relevant)" : ""}</td>
                      <td>{String(market.market ?? "-")}</td>
                      <td>{String(market.probability ?? market.yes_price ?? "-")}</td>
                    </tr>
                  ))}
                  {(research?.prediction_markets ?? []).length === 0 ? (
                    <tr>
                      <td colSpan={3} className="muted">No relevant prediction market found for this ticker.</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
            {(research?.prediction_markets ?? []).length > 0 ? (
              <div className="inline-links wrap">
                {(research?.prediction_markets ?? []).slice(0, 3).map((market, idx) => (
                  typeof market.link === "string" && market.link ? (
                    <a key={`prediction-link-${idx}`} href={market.link} target="_blank" rel="noreferrer">
                      Open market {idx + 1}
                    </a>
                  ) : null
                ))}
              </div>
            ) : null}
          </div>
        </div>
      </div>

      <div className="glass-card">
        <h3>Research Destinations</h3>
        <div className="inline-links wrap">
          {(research?.tool_links ?? []).map((link) => (
            <a key={`${link.source}-${link.url}`} href={link.url} target="_blank" rel="noreferrer">
              {link.source}
            </a>
          ))}
        </div>
      </div>

      {deepResearch ? (
        <div className="glass-card">
          <div className="panel-header">
            <h3>Deep Research</h3>
            <span className="muted">{deepResearch.symbol} · {deepResearch.source}</span>
          </div>
          {(deepResearch.deep_bullets ?? []).length > 0 ? (
            <ul className="source-summary-list">
              {(deepResearch.deep_bullets ?? []).map((point, idx) => (
                <li key={`deep-bullet-${idx}`}>{point}</li>
              ))}
            </ul>
          ) : null}

          <div className="card-row card-row-split" style={{ marginTop: "10px" }}>
            <article className="source-item">
              <h4>Analyst View</h4>
              <p>{deepResearch.analyst_ratings ?? "-"}</p>
              {(deepResearch.recommendation_timeline ?? []).length > 0 ? (
                <div className="table-wrap insider-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Period</th>
                        <th>Strong Buy</th>
                        <th>Buy</th>
                        <th>Hold</th>
                        <th>Sell</th>
                        <th>Strong Sell</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(deepResearch.recommendation_timeline ?? []).map((row, idx) => (
                        <tr key={`rec-${idx}`}>
                          <td>{row.period}</td>
                          <td>{row.strong_buy}</td>
                          <td>{row.buy}</td>
                          <td>{row.hold}</td>
                          <td>{row.sell}</td>
                          <td>{row.strong_sell}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </article>

            <article className="source-item">
              <h4>Price Targets</h4>
              <div className="kpi-grid">
                <div><p className="muted">Mean</p><h3>{formatToCents(deepResearch.price_target?.target_mean)}</h3></div>
                <div><p className="muted">High</p><h3>{formatToCents(deepResearch.price_target?.target_high)}</h3></div>
                <div><p className="muted">Low</p><h3>{formatToCents(deepResearch.price_target?.target_low)}</h3></div>
              </div>
              {deepResearch.price_target?.last_updated ? (
                <p className="muted">Updated: {deepResearch.price_target.last_updated}</p>
              ) : null}
              <h4 style={{ marginTop: "12px" }}>Reddit DD</h4>
              <p>{deepResearch.reddit_dd_summary ?? "-"}</p>
            </article>
          </div>

          <div className="card-row card-row-split" style={{ marginTop: "10px" }}>
            <article className="source-item">
              <h4>Insider Highlights</h4>
              <p>{deepResearch.insider_trading ?? "-"}</p>
              {(deepResearch.insider_highlights ?? []).length > 0 ? (
                <div className="table-wrap insider-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Name</th>
                        <th>Code</th>
                        <th>Shares</th>
                        <th>Price</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(deepResearch.insider_highlights ?? []).map((row, idx) => (
                        <tr key={`deep-insider-${idx}`}>
                          <td>{row.date ?? "-"}</td>
                          <td>{row.name ?? "-"}</td>
                          <td>{row.code ?? "-"}</td>
                          <td>{formatCompactNumber(row.shares)}</td>
                          <td>{formatToCents(row.price)}</td>
                          <td>{formatCompactNumber(row.value_estimate)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </article>

            <article className="source-item">
              <h4>Reddit Highlights</h4>
              <div className="stack small-gap">
                {(deepResearch.reddit_highlights ?? []).map((row, idx) => (
                  <div key={`deep-reddit-${idx}`}>
                    <p>
                      <strong>r/{row.subreddit}</strong>: {row.title}
                    </p>
                    <div className="source-citations">
                      {row.url ? <a href={row.url} target="_blank" rel="noreferrer">Open thread</a> : null}
                      <span className="muted">score {row.score ?? "-"} · comments {row.comments ?? "-"}</span>
                    </div>
                  </div>
                ))}
                {(deepResearch.reddit_highlights ?? []).length === 0 ? <p className="muted">No subreddit highlights available.</p> : null}
              </div>
            </article>
          </div>

          <article className="source-item" style={{ marginTop: "10px" }}>
            <h4>Recent News</h4>
            <div className="stack small-gap">
              {(deepResearch.recent_news ?? []).map((item, idx) => (
                <div key={`deep-news-${idx}`}>
                  <p><strong>{item.headline ?? "-"}</strong></p>
                  <div className="source-citations">
                    {item.url ? <a href={item.url} target="_blank" rel="noreferrer">Open source</a> : null}
                    <span className="muted">{item.source ?? "unknown source"}</span>
                  </div>
                  {item.summary ? <p className="muted">{item.summary}</p> : null}
                </div>
              ))}
              {(deepResearch.recent_news ?? []).length === 0 ? <p className="muted">No recent company news returned.</p> : null}
            </div>
          </article>
          {deepResearch.sources && deepResearch.sources.length > 0 ? (
            <p className="muted" style={{ marginTop: "8px" }}>Sources: {deepResearch.sources.join(", ")}</p>
          ) : null}
        </div>
      ) : null}

      <div className="glass-card">
        <div className="panel-header">
          <h3>Insider Trading</h3>
          <span className="muted">
            {insiderRows.length} filings
            {insiderRows.length > 0 ? ` · Page ${insiderPageSafe}/${insiderTotalPages}` : ""}
          </span>
        </div>
        <div className="table-wrap insider-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Name</th>
                <th>Role</th>
                <th>Action</th>
                <th>Shares</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {insiderPageRows.map((row, idx) => (
                <tr key={`insider-${insiderSliceStart + idx}`}>
                  <td>{row.start_date ?? "-"}</td>
                  <td>{row.filer_name ?? "-"}</td>
                  <td>{row.filer_relation ?? "-"}</td>
                  <td>{row.money_text ?? "-"}</td>
                  <td>{formatCompactNumber(row.shares)}</td>
                  <td>{formatCompactNumber(row.value)}</td>
                </tr>
              ))}
              {insiderRows.length === 0 ? (
                <tr>
                  <td colSpan={6} className="muted">No insider filings returned for this symbol.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        {insiderRows.length > insiderPageSize ? (
          <div className="insider-pagination">
            <button
              type="button"
              className="secondary"
              onClick={() => setInsiderPage((page) => Math.max(1, page - 1))}
              disabled={insiderPageSafe <= 1}
            >
              Previous
            </button>
            <span className="muted">
              Showing {insiderSliceStart + 1}-{Math.min(insiderSliceStart + insiderPageSize, insiderRows.length)} of {insiderRows.length}
            </span>
            <button
              type="button"
              className="secondary"
              onClick={() => setInsiderPage((page) => Math.min(insiderTotalPages, page + 1))}
              disabled={insiderPageSafe >= insiderTotalPages}
            >
              Next
            </button>
          </div>
        ) : null}
      </div>

      <div className="glass-card live-wire-log">
        <div className="panel-header">
          <h3>Live Wire</h3>
          <span className={connected ? "dot dot-live" : "dot dot-offline"}>
            {connected ? "Socket Live" : "Socket Offline"}
          </span>
        </div>
        <div className="event-list">
          {liveWireActivity.length === 0 ? <p className="muted">No AI agent activity for {normalizedTicker} yet.</p> : null}
          {liveWireActivity.map((event, idx) => (
            <article key={`${activityKey(event)}-${idx}`} className="event-item">
              <div className="event-meta">
                <strong>{event.agent_name ?? "AI Agent"}</strong>
                <span>{event.module ?? "research"}</span>
              </div>
              <p className="live-wire-action">{event.action ?? "Processing request..."}</p>
              <div className="live-wire-footer">
                <span className={`live-wire-status ${statusClass(event.status)}`}>{(event.status ?? "running").toLowerCase()}</span>
                {event.created_at || event.timestamp ? (
                  <time>{new Date(event.created_at ?? event.timestamp ?? "").toLocaleTimeString()}</time>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      </div>

    </section>
  );
}
