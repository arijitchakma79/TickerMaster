import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { jsPDF } from "jspdf";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import {
  fetchRealtimeQuote,
  getModalCronHealth,
  pauseSimulation,
  requestCommentary,
  resumeSimulation,
  spinModalSandbox,
  startSimulation,
  stopSimulation
} from "../lib/api";
import EventRail from "./EventRail";
import { formatCurrency } from "../lib/format";
import type {
  AgentConfig,
  MarketMetric,
  ModalCronHealthResponse,
  ModalSandboxResponse,
  SimulationState,
  TradeRecord,
  WSMessage
} from "../lib/types";
import blackrockLogo from "../images/blackrock.png";
import citadelLogo from "../images/citadel.png";
import crowdLogo from "../images/crowd.png";
import janeStreetLogo from "../images/jane street.png";
import nancyLogo from "../images/nancy.png";
import vanguardLogo from "../images/vanguard.png";

interface Props {
  activeTicker: string;
  onTickerChange: (ticker: string) => void;
  watchlist: string[];
  connected: boolean;
  simulationEvent?: WSMessage;
  simulationLifecycleEvent?: WSMessage;
}

interface UserAgentEntry {
  config: AgentConfig;
  iconEmoji?: string;
  editor?: {
    risk: number;
    tempo: number;
    style: number;
    news: number;
  };
}

const SELF_AGENT_NAME = "My Trading Agent";

const DEFAULT_SETTINGS = {
  duration: 180,
  fallbackInitialPrice: 185,
  startingCash: 100000,
  baseVolatility: 0.02
};

const DEFAULT_CUSTOM = {
  name: SELF_AGENT_NAME,
  prompt: "Buy gradual pullbacks in strong trends, reduce size during crash headlines.",
  emoji: "🧠",
  risk: 52,
  tempo: 52,
  style: 56,
  news: 42
};

const STRATEGY_TEMPLATES = [
  {
    label: "Momentum",
    prompt: "I am a momentum trader. Buy upside breakouts with rising volume and cut losses quickly when trend structure fails."
  },
  {
    label: "Contrarian",
    prompt: "I am a contrarian trader. Buy oversold flushes below short-term fair value and trim into rapid mean reversion bounces."
  },
  {
    label: "Risk-Off",
    prompt: "I prioritize drawdown control. Trade smaller in high-volatility regimes, keep cash high during crash headlines, and avoid revenge trades."
  },
  {
    label: "News Reversal",
    prompt: "I trade post-news dislocations. Fade initial overreaction after catalyst headlines and only size up when follow-through confirms."
  }
] as const;

const ALL_MARKET_PROMPT_PATTERN =
  /\b(all stocks|all tickers|entire market|whole market|market-wide|across the market)\b/i;

const TICKER_PROMPT_STOPWORDS = new Set([
  "ABOUT",
  "A",
  "AI",
  "ALL",
  "AM",
  "AND",
  "AS",
  "AT",
  "BE",
  "BUY",
  "CASH",
  "CRASH",
  "DO",
  "FOR",
  "FROM",
  "HOLD",
  "IF",
  "IN",
  "IS",
  "IT",
  "LONG",
  "MARKET",
  "MY",
  "NEWS",
  "OF",
  "ON",
  "OR",
  "ONLY",
  "PRICE",
  "RISK",
  "SELL",
  "SHORT",
  "SO",
  "STOCK",
  "STOCKS",
  "STRATEGY",
  "THAT",
  "THE",
  "THIS",
  "TO",
  "TRADE",
  "TRADING",
  "USE",
  "VOL",
  "VOLATILITY",
  "WORRY",
  "WHEN",
  "WITH"
]);

const POSITIVE_NEWS_TERMS = new Set([
  "beat",
  "beats",
  "upgrade",
  "upgraded",
  "growth",
  "surge",
  "strong",
  "bullish",
  "profit",
  "record",
  "buyback",
  "outperform",
  "raise",
  "raised",
  "expansion",
  "rebound",
  "rally"
]);

const NEGATIVE_NEWS_TERMS = new Set([
  "miss",
  "missed",
  "downgrade",
  "downgraded",
  "warning",
  "lawsuit",
  "probe",
  "weak",
  "bearish",
  "decline",
  "selloff",
  "cut",
  "cuts",
  "slump",
  "recession",
  "layoff",
  "layoffs",
  "delay",
  "delays",
  "pressure",
  "crash",
  "drop",
  "drops"
]);

const INSTITUTIONAL_AGENTS: AgentConfig[] = [
  {
    name: "Citadel Execution Desk",
    personality: "quant_momentum",
    strategy_prompt: "Exploit short-term trend continuation with strict inventory controls.",
    model: "meta-llama/llama-3.1-8b-instruct",
    aggressiveness: 0.78,
    risk_limit: 0.66,
    trade_size: 36,
    active: true
  },
  {
    name: "Jane Street Microflow",
    personality: "quant_momentum",
    strategy_prompt: "React to order-flow imbalance and fast reversals.",
    model: "meta-llama/llama-3.1-8b-instruct",
    aggressiveness: 0.72,
    risk_limit: 0.62,
    trade_size: 30,
    active: true
  },
  {
    name: "BlackRock Macro Core",
    personality: "fundamental_value",
    strategy_prompt: "Favor quality assets when price dislocates below fair value.",
    model: "meta-llama/llama-3.1-8b-instruct",
    aggressiveness: 0.48,
    risk_limit: 0.55,
    trade_size: 24,
    active: true
  },
  {
    name: "Vanguard Index Sentinel",
    personality: "fundamental_value",
    strategy_prompt: "Trade gradually with drawdown control and low turnover.",
    model: "meta-llama/llama-3.1-8b-instruct",
    aggressiveness: 0.39,
    risk_limit: 0.52,
    trade_size: 20,
    active: true
  },
  {
    name: "Nancy Pelosi Tracker",
    personality: "retail_reactive",
    strategy_prompt: "React quickly to policy and disclosure-style headline shifts.",
    model: "meta-llama/llama-3.1-8b-instruct",
    aggressiveness: 0.61,
    risk_limit: 0.5,
    trade_size: 18,
    active: true
  },
  {
    name: "Retail Crowd Pulse",
    personality: "retail_reactive",
    strategy_prompt: "Follow sentiment shifts after delayed news diffusion.",
    model: "meta-llama/llama-3.1-8b-instruct",
    aggressiveness: 0.58,
    risk_limit: 0.49,
    trade_size: 16,
    active: true
  }
];

const AGENT_LOGOS: Record<string, string> = {
  "Citadel Execution Desk": citadelLogo,
  "Jane Street Microflow": janeStreetLogo,
  "BlackRock Macro Core": blackrockLogo,
  "Vanguard Index Sentinel": vanguardLogo,
  "Nancy Pelosi Tracker": nancyLogo,
  "Retail Crowd Pulse": crowdLogo
};

function clamp(min: number, value: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function derivePersonality(style: number, news: number): AgentConfig["personality"] {
  if (news >= 68) return "retail_reactive";
  if (style >= 58) return "quant_momentum";
  return "fundamental_value";
}

function personalityLabel(personality: AgentConfig["personality"]) {
  if (personality === "quant_momentum") return "Fast Trend";
  if (personality === "fundamental_value") return "Deep Value";
  return "News Reactive";
}

function styleLabel(style: number) {
  if (style < 34) return "Value-first";
  if (style < 67) return "Balanced";
  return "Momentum-first";
}

function buildCustomAgent(input: {
  name: string;
  prompt: string;
  risk: number;
  tempo: number;
  style: number;
  news: number;
}): AgentConfig {
  const personality = derivePersonality(input.style, input.news);
  return {
    name: input.name.trim() || SELF_AGENT_NAME,
    personality,
    strategy_prompt: input.prompt.trim(),
    model: "meta-llama/llama-3.1-8b-instruct",
    aggressiveness: Number(clamp(0.2, 0.22 + input.tempo / 100 * 0.7, 1).toFixed(2)),
    risk_limit: Number(clamp(0.2, 0.25 + input.risk / 100 * 0.65, 1).toFixed(2)),
    trade_size: Math.round(clamp(6, 8 + input.tempo / 100 * 42, 1000)),
    active: true
  };
}

function inferEditorStateFromAgent(agent: AgentConfig) {
  const tempo = Math.round(clamp(0, ((agent.aggressiveness - 0.22) / 0.7) * 100, 100));
  const risk = Math.round(clamp(0, ((agent.risk_limit - 0.25) / 0.65) * 100, 100));

  if (agent.personality === "quant_momentum") {
    return { risk, tempo, style: 76, news: 48 };
  }
  if (agent.personality === "retail_reactive") {
    return { risk, tempo, style: 50, news: 78 };
  }
  return { risk, tempo, style: 24, news: 36 };
}

function buildAgentStatus(
  session: SimulationState | null,
  latestTrade: TradeRecord | undefined,
  isPreset: boolean
): string {
  if (!session || session.tick === 0) return isPreset ? "Locked" : "Waiting";
  if (session.paused) return "Paused";
  if (session.running) {
    if (latestTrade) return latestTrade.side === "buy" ? "Buying" : "Selling";
    return "Analyzing";
  }
  if (latestTrade) return `Last ${latestTrade.side.toUpperCase()}`;
  return isPreset ? "Locked" : "Holding";
}

function statusTone(status: string) {
  if (status === "Buying" || status === "Selling" || status === "Analyzing") return "live";
  if (status === "Paused") return "paused";
  if (status === "Locked") return "locked";
  if (status.startsWith("Last")) return "ended";
  return "idle";
}

function positionRowsForCard(portfolio: SimulationState["portfolios"][string] | undefined) {
  if (!portfolio?.positions) return [];
  return Object.entries(portfolio.positions)
    .filter(([, position]) => (position.holdings ?? 0) > 0 || Math.abs(position.net_gain ?? 0) >= 0.01)
    .sort((a, b) => Math.abs((b[1].net_gain ?? 0)) - Math.abs((a[1].net_gain ?? 0)))
    .slice(0, 4)
    .map(([ticker, position]) => ({
      ticker,
      holdings: position.holdings ?? 0,
      netGain: position.net_gain ?? 0
    }));
}

function makeUniqueName(baseName: string, takenNames: Set<string>) {
  const seed = baseName.trim() || SELF_AGENT_NAME;
  let candidate = seed;
  let index = 2;
  while (takenNames.has(candidate)) {
    candidate = `${seed} ${index}`;
    index += 1;
  }
  return candidate;
}

function normalizeEmoji(input: string) {
  const trimmed = input.trim();
  return trimmed.length > 0 ? trimmed.slice(0, 4) : DEFAULT_CUSTOM.emoji;
}

function deriveVolatilityFromQuote(changePercent?: number | null) {
  if (changePercent == null || Number.isNaN(changePercent)) {
    return DEFAULT_SETTINGS.baseVolatility;
  }
  return Number(clamp(0.008, 0.012 + Math.abs(changePercent) / 140, 0.08).toFixed(4));
}

function normalizeSymbol(value: string) {
  return value.trim().toUpperCase().replace(/\./g, "-");
}

function parseSimulationNewsItem(raw: string) {
  const fallback = raw.replace(/\s\|\s/g, " • ").trim();
  if (!raw) {
    return { ticker: "", source: "", headline: fallback };
  }

  const [metaRaw, ...headlineParts] = raw.split(":");
  if (headlineParts.length === 0) {
    return { ticker: "", source: "", headline: fallback };
  }

  const headline = headlineParts.join(":").trim().replace(/\s\|\s/g, " • ");
  const metaParts = metaRaw
    .split("|")
    .map((part) => part.trim())
    .filter(Boolean);

  if (metaParts.length >= 2) {
    return {
      ticker: metaParts[0],
      source: metaParts.slice(1).join(" • "),
      headline
    };
  }

  if (metaParts.length === 1) {
    const token = metaParts[0];
    if (/^[A-Z][A-Z0-9.-]{0,9}$/.test(token)) {
      return { ticker: token, source: "", headline };
    }
    return { ticker: "", source: token, headline };
  }

  return { ticker: "", source: "", headline };
}

function inferNewsTone(headline: string): "positive" | "negative" | "neutral" {
  const text = headline.toLowerCase();
  const tokens = text.match(/[a-z]+/g) ?? [];
  let positive = 0;
  let negative = 0;

  for (const token of tokens) {
    if (POSITIVE_NEWS_TERMS.has(token)) positive += 1;
    if (NEGATIVE_NEWS_TERMS.has(token)) negative += 1;
  }

  if (text.includes("risk-off")) negative += 1;
  if (text.includes("risk on")) positive += 1;

  if (positive > negative && positive > 0) return "positive";
  if (negative > positive && negative > 0) return "negative";
  return "neutral";
}

function extractTickerCandidatesFromPrompt(prompt: string, watchlist: string[]): string[] {
  const matches = prompt.match(/\b\$?[A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?\b/g) ?? [];
  const normalizedWatchlist = new Set(watchlist.map(normalizeSymbol).filter(Boolean));
  const seen = new Set<string>();
  const out: string[] = [];

  for (const token of matches) {
    const hasDollarPrefix = token.startsWith("$");
    const rawToken = hasDollarPrefix ? token.slice(1) : token;
    const symbol = normalizeSymbol(rawToken);
    if (!symbol) continue;
    if (TICKER_PROMPT_STOPWORDS.has(symbol)) continue;
    if (!/^[A-Z][A-Z0-9-]{0,9}$/.test(symbol)) continue;
    const isExplicitUppercase = rawToken === rawToken.toUpperCase();
    const inWatchlist = normalizedWatchlist.has(symbol);
    // Avoid turning normal sentence words into fake tickers (e.g. Play -> PLAY).
    if (!hasDollarPrefix && !isExplicitUppercase && !inWatchlist) continue;
    if (symbol.length === 1 && !hasDollarPrefix && !inWatchlist) continue;
    if (!seen.has(symbol)) {
      seen.add(symbol);
      out.push(symbol);
    }
  }

  return out;
}

function resolveTickersFromPrompt(prompt: string, activeTicker: string, watchlist: string[]): string[] {
  const inferred = extractTickerCandidatesFromPrompt(prompt, watchlist);
  const normalizedWatchlist = watchlist.map(normalizeSymbol).filter(Boolean);
  const dedupedWatchlist = Array.from(new Set(normalizedWatchlist));
  const primary = normalizeSymbol(activeTicker) || dedupedWatchlist[0] || "AAPL";

  if (inferred.length > 0) {
    const merged = [...inferred, ...dedupedWatchlist.filter((symbol) => !inferred.includes(symbol))];
    return merged.slice(0, 12);
  }

  if (ALL_MARKET_PROMPT_PATTERN.test(prompt)) {
    if (dedupedWatchlist.length > 0) {
      if (dedupedWatchlist.includes("SPY")) {
        return ["SPY", ...dedupedWatchlist.filter((symbol) => symbol !== "SPY")];
      }
      return dedupedWatchlist;
    }
  }

  if (dedupedWatchlist.length > 1) {
    if (dedupedWatchlist.includes(primary)) {
      return [primary, ...dedupedWatchlist.filter((symbol) => symbol !== primary)];
    }
    return [primary, ...dedupedWatchlist];
  }

  return [primary];
}

async function filterTradableTickers(candidates: string[], fallbackTicker: string): Promise<string[]> {
  const unique = Array.from(new Set(candidates.map(normalizeSymbol).filter(Boolean))).slice(0, 12);
  if (unique.length === 0) return [fallbackTicker];

  const checks = await Promise.all(
    unique.map(async (symbol) => {
      try {
        const quote = await fetchRealtimeQuote(symbol);
        return Number.isFinite(quote.price) && quote.price > 0 ? symbol : null;
      } catch {
        return null;
      }
    })
  );

  const valid = checks.filter((value): value is string => Boolean(value));
  if (valid.length > 0) return valid;
  return [fallbackTicker];
}

export default function SimulationPanel({
  activeTicker,
  onTickerChange,
  watchlist,
  connected,
  simulationEvent,
  simulationLifecycleEvent
}: Props) {
  const [customName, setCustomName] = useState(DEFAULT_CUSTOM.name);
  const [customPrompt, setCustomPrompt] = useState(DEFAULT_CUSTOM.prompt);
  const [customRisk, setCustomRisk] = useState(DEFAULT_CUSTOM.risk);
  const [customTempo, setCustomTempo] = useState(DEFAULT_CUSTOM.tempo);
  const [customStyle, setCustomStyle] = useState(DEFAULT_CUSTOM.style);
  const [customNews, setCustomNews] = useState(DEFAULT_CUSTOM.news);
  const [customEmoji, setCustomEmoji] = useState("");
  const [addedCustomAgents, setAddedCustomAgents] = useState<UserAgentEntry[]>([]);
  const [customAgentSequence, setCustomAgentSequence] = useState(2);
  const [agentDialogOpen, setAgentDialogOpen] = useState(false);
  const [agentDialogMode, setAgentDialogMode] = useState<"create" | "edit">("create");
  const [editingAgentName, setEditingAgentName] = useState<string | null>(null);

  const [session, setSession] = useState<SimulationState | null>(null);
  const [sessionAgents, setSessionAgents] = useState<AgentConfig[]>([]);
  const [priceSeries, setPriceSeries] = useState<Array<{ tick: number; price: number }>>([]);
  const [loading, setLoading] = useState(false);
  const [autoCommentary, setAutoCommentary] = useState<string>("");
  const [autoCommentaryModel, setAutoCommentaryModel] = useState<string>("");
  const [autoCommentaryLoading, setAutoCommentaryLoading] = useState(false);
  const [startingCapital, setStartingCapital] = useState(DEFAULT_SETTINGS.startingCash);
  const [liveQuote, setLiveQuote] = useState<MarketMetric | null>(null);
  const [quoteLoading, setQuoteLoading] = useState(false);
  const [quoteError, setQuoteError] = useState("");
  const [modalHealth, setModalHealth] = useState<ModalCronHealthResponse | null>(null);
  const [modalHealthError, setModalHealthError] = useState("");
  const [modalSandboxResult, setModalSandboxResult] = useState<ModalSandboxResponse | null>(null);
  const [modalSandboxLoading, setModalSandboxLoading] = useState(false);
  const [modalSandboxError, setModalSandboxError] = useState("");
  const [, setPromptInferredTickers] = useState<string[]>([]);
  const [sessionStartError, setSessionStartError] = useState("");
  const marketBoardWrapRef = useRef<HTMLDivElement | null>(null);
  const [syncTelemetryHeight, setSyncTelemetryHeight] = useState(false);
  const [telemetryCardHeight, setTelemetryCardHeight] = useState<number | null>(null);
  const activeSessionId = session?.session_id ?? null;
  const telemetryHeightSynced = syncTelemetryHeight && Boolean(telemetryCardHeight);

  useEffect(() => {
    const media = window.matchMedia("(min-width: 981px)");
    const onChange = () => setSyncTelemetryHeight(media.matches);
    onChange();
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", onChange);
      return () => media.removeEventListener("change", onChange);
    }
    media.addListener(onChange);
    return () => media.removeListener(onChange);
  }, []);

  useEffect(() => {
    if (!syncTelemetryHeight) {
      setTelemetryCardHeight(null);
      return;
    }
    const target = marketBoardWrapRef.current;
    if (!target) return;

    const updateHeight = () => {
      const next = Math.round(target.getBoundingClientRect().height);
      setTelemetryCardHeight((prev) => (prev === next ? prev : next > 0 ? next : null));
    };

    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(target);
    return () => observer.disconnect();
  }, [syncTelemetryHeight]);

  useEffect(() => {
    if (!agentDialogOpen) return;
    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setAgentDialogOpen(false);
        setAgentDialogMode("create");
        setEditingAgentName(null);
      }
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeydown);

    return () => {
      window.removeEventListener("keydown", handleKeydown);
      document.body.style.overflow = previousOverflow;
    };
  }, [agentDialogOpen]);

  useEffect(() => {
    const symbol = activeTicker.trim().toUpperCase();
    if (!symbol) return;

    let active = true;
    setQuoteLoading(true);
    setQuoteError("");
    void fetchRealtimeQuote(symbol)
      .then((quote) => {
        if (!active) return;
        setLiveQuote(quote);
      })
      .catch(() => {
        if (!active) return;
        setLiveQuote(null);
        setQuoteError("Live quote unavailable");
      })
      .finally(() => {
        if (!active) return;
        setQuoteLoading(false);
      });

    return () => {
      active = false;
    };
  }, [activeTicker]);

  useEffect(() => {
    let active = true;
    setModalHealthError("");
    void getModalCronHealth()
      .then((health) => {
        if (!active) return;
        setModalHealth(health);
      })
      .catch(() => {
        if (!active) return;
        setModalHealth(null);
        setModalHealthError("Modal runtime health unavailable.");
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!simulationEvent || simulationEvent.type !== "tick") return;
    if (!activeSessionId || simulationEvent.session_id !== activeSessionId) return;

    const nextTick = Number(simulationEvent.tick ?? 0);
    const nextPrice = Number(simulationEvent.price ?? 0);

    setPriceSeries((prev) => [...prev, { tick: nextTick, price: nextPrice }].slice(-200));

    setSession((prev) => {
      if (!prev || simulationEvent.session_id !== prev.session_id) return prev;
      return {
        ...prev,
        tick: Number.isFinite(nextTick) && nextTick > 0 ? nextTick : prev.tick,
        current_price: Number.isFinite(nextPrice) && nextPrice > 0 ? nextPrice : prev.current_price,
        tickers: Array.isArray(simulationEvent.tickers) ? (simulationEvent.tickers as string[]) : prev.tickers,
        market_prices:
          (simulationEvent.market_prices as SimulationState["market_prices"]) ?? prev.market_prices,
        paused: prev.paused,
        crash_mode: Boolean(simulationEvent.crash_mode),
        recent_news: Array.isArray(simulationEvent.news) ? (simulationEvent.news as string[]) : prev.recent_news,
        order_book:
          (simulationEvent.order_book as SimulationState["order_book"]) ?? prev.order_book,
        portfolios:
          (simulationEvent.portfolio_snapshot as SimulationState["portfolios"]) ?? prev.portfolios,
        trades: [
          ...((Array.isArray(simulationEvent.trades) ? (simulationEvent.trades as SimulationState["trades"]) : []) ?? []),
          ...prev.trades
        ].slice(0, 120)
      };
    });
  }, [simulationEvent, activeSessionId]);

  async function generateAutoCommentary(reason: "completed" | "stopped", state: SimulationState) {
    setAutoCommentaryLoading(true);
    try {
      const winnerEntry = Object.entries(state.portfolios).sort((a, b) => (b[1].equity ?? 0) - (a[1].equity ?? 0))[0];
      const winner = winnerEntry?.[0] ?? "No clear winner";
      const winnerEquity = winnerEntry?.[1]?.equity ?? 0;

      const prompt =
        `Simulation ${reason}. Summarize what happened, who performed best, why, and one educational lesson for retail traders.` +
        ` Keep it concise and high-signal.`;

      const out = await requestCommentary(prompt, {
        ticker: state.ticker,
        reason,
        tick: state.tick,
        final_price: state.current_price,
        total_trades: state.trades.length,
        crash_mode: state.crash_mode,
        winner,
        winner_equity: winnerEquity
      });
      setAutoCommentary(out.response);
      setAutoCommentaryModel(out.model);
    } catch {
      setAutoCommentary(
        `Simulation ${reason}. ${state.ticker} closed near ${formatCurrency(
          state.current_price
        )}. Review slippage, trade timing, and risk controls before next run.`
      );
      setAutoCommentaryModel("fallback-template");
    } finally {
      setAutoCommentaryLoading(false);
    }
  }

  useEffect(() => {
    if (!simulationLifecycleEvent || !session) return;
    if (!session.running) return;
    if (simulationLifecycleEvent.session_id !== session.session_id) return;
    if (simulationLifecycleEvent.type !== "simulation_completed" && simulationLifecycleEvent.type !== "simulation_stopped") return;

    const reason = simulationLifecycleEvent.type === "simulation_completed" ? "completed" : "stopped";
    const finalState = { ...session, running: false, paused: false };
    setSession(finalState);
    void generateAutoCommentary(reason, finalState);
  }, [simulationLifecycleEvent, session]);

  const customAgentConfig = useMemo(
    () =>
      buildCustomAgent({
        name: customName,
        prompt: customPrompt,
        risk: customRisk,
        tempo: customTempo,
        style: customStyle,
        news: customNews
      }),
    [customName, customPrompt, customRisk, customTempo, customStyle, customNews]
  );

  const previewAgents = useMemo(
    () => [...INSTITUTIONAL_AGENTS, ...addedCustomAgents.map((entry) => entry.config)],
    [addedCustomAgents]
  );

  const displayedAgents = sessionAgents.length > 0 ? sessionAgents : previewAgents;

  const customAgentEmojis = useMemo(() => {
    const mapped: Record<string, string> = {};
    for (const entry of addedCustomAgents) {
      if (entry.iconEmoji) {
        mapped[entry.config.name] = entry.iconEmoji;
      }
    }
    return mapped;
  }, [addedCustomAgents]);

  const userAgentNames = useMemo(
    () => new Set([SELF_AGENT_NAME, ...addedCustomAgents.map((entry) => entry.config.name)]),
    [addedCustomAgents]
  );
  const editableAgentNames = useMemo(
    () => new Set(addedCustomAgents.map((entry) => entry.config.name)),
    [addedCustomAgents]
  );
  const institutionalAgentNames = useMemo(
    () => new Set(INSTITUTIONAL_AGENTS.map((agent) => agent.name)),
    []
  );

  function openCreateAgentDialog() {
    if (customEditorDisabled) return;
    setAgentDialogMode("create");
    setEditingAgentName(null);
    setCustomName("");
    setCustomPrompt("");
    setCustomRisk(DEFAULT_CUSTOM.risk);
    setCustomTempo(DEFAULT_CUSTOM.tempo);
    setCustomStyle(DEFAULT_CUSTOM.style);
    setCustomNews(DEFAULT_CUSTOM.news);
    setCustomEmoji("");
    setAgentDialogOpen(true);
  }

  function openEditAgentDialog(agent: AgentConfig) {
    if (customEditorDisabled) return;
    const entry = addedCustomAgents.find((candidate) => candidate.config.name === agent.name);
    const editor = entry?.editor ?? inferEditorStateFromAgent(agent);

    setCustomName(agent.name);
    setCustomPrompt(agent.strategy_prompt ?? "");
    setCustomRisk(editor.risk);
    setCustomTempo(editor.tempo);
    setCustomStyle(editor.style);
    setCustomNews(editor.news);
    setCustomEmoji(entry?.iconEmoji ?? "");
    setAgentDialogMode("edit");
    setEditingAgentName(agent.name);
    setAgentDialogOpen(true);
  }

  function handleRemoveAddedAgent(name: string) {
    if (customEditorDisabled) return;
    setAddedCustomAgents((previous) => previous.filter((entry) => entry.config.name !== name));
    if (editingAgentName === name) {
      setEditingAgentName(null);
      setAgentDialogMode("create");
    }
  }

  function handleCloseAgentDialog() {
    setAgentDialogOpen(false);
    setAgentDialogMode("create");
    setEditingAgentName(null);
  }

  function handleSaveAgentDialog() {
    if (customEditorDisabled) return;
    const prompt = customPrompt.trim();
    if (!prompt) return;

    if (agentDialogMode === "edit" && editingAgentName) {
      setAddedCustomAgents((previous) => {
        const takenNames = new Set([
          ...INSTITUTIONAL_AGENTS.map((agent) => agent.name),
          ...previous.map((entry) => entry.config.name).filter((name) => name !== editingAgentName)
        ]);
        const uniqueName = makeUniqueName(customName, takenNames);
        const nextConfig = buildCustomAgent({
          name: uniqueName,
          prompt: customPrompt,
          risk: customRisk,
          tempo: customTempo,
          style: customStyle,
          news: customNews
        });

        return previous.map((entry) =>
          entry.config.name === editingAgentName
            ? {
                ...entry,
                config: nextConfig,
                iconEmoji: normalizeEmoji(customEmoji),
                editor: {
                  risk: customRisk,
                  tempo: customTempo,
                  style: customStyle,
                  news: customNews
                }
              }
            : entry
        );
      });
      handleCloseAgentDialog();
      return;
    }

    const takenNames = new Set([
      ...INSTITUTIONAL_AGENTS.map((agent) => agent.name),
      ...addedCustomAgents.map((entry) => entry.config.name)
    ]);
    const uniqueName = makeUniqueName(customName, takenNames);
    const nextConfig = buildCustomAgent({
      name: uniqueName,
      prompt: customPrompt,
      risk: customRisk,
      tempo: customTempo,
      style: customStyle,
      news: customNews
    });

    setAddedCustomAgents((previous) => [
      ...previous,
      {
        config: nextConfig,
        iconEmoji: normalizeEmoji(customEmoji),
        editor: {
          risk: customRisk,
          tempo: customTempo,
          style: customStyle,
          news: customNews
        }
      }
    ]);
    setCustomName(`${SELF_AGENT_NAME} ${customAgentSequence}`);
    setCustomEmoji("");
    setCustomAgentSequence((value) => value + 1);
    handleCloseAgentDialog();
  }

  function buildSandboxPrompt(ticker: string, configuredAgents: AgentConfig[], targetTickers: string[]): string {
    const customStrategies = configuredAgents
      .filter((agent) => userAgentNames.has(agent.name))
      .map((agent) => `${agent.name}: ${agent.strategy_prompt || "No custom prompt provided."}`)
      .slice(0, 3);

    const agentList = configuredAgents.map((agent) => agent.name).join(", ");
    const strategyBlock =
      customStrategies.length > 0 ? customStrategies.join(" | ") : `Use preset arena agents for ${ticker}.`;
    const universe = targetTickers.join(", ");
    return `Run a ${ticker} multi-agent simulation with realistic slippage and delayed news. Preferred universe: ${universe}. Agents: ${agentList}. Custom strategies: ${strategyBlock}`;
  }

  async function handlePlay() {
    setLoading(true);
    setModalSandboxError("");
    setSessionStartError("");
    try {
      if (session?.running && session.paused) {
        const resumed = await resumeSimulation(session.session_id);
        setSession(resumed);
        return;
      }

      if (session?.running && !session.paused) {
        return;
      }

      const strategyContext = [
        customPrompt,
        ...addedCustomAgents.map((entry) => entry.config.strategy_prompt || "")
      ]
        .join(" ")
        .trim();
      const rawResolvedTickers = resolveTickersFromPrompt(strategyContext, activeTicker, watchlist);
      const fallbackTicker = normalizeSymbol(activeTicker) || "AAPL";
      const resolvedTickers = await filterTradableTickers(rawResolvedTickers, fallbackTicker);
      const cleanTicker = resolvedTickers[0] ?? (normalizeSymbol(activeTicker) || "AAPL");
      setPromptInferredTickers(resolvedTickers);
      onTickerChange(cleanTicker);

      let quote = liveQuote;
      if (!quote || quote.ticker !== cleanTicker) {
        quote = await fetchRealtimeQuote(cleanTicker).catch(() => null);
        if (quote) {
          setLiveQuote(quote);
          setQuoteError("");
        }
      }

      const candidateInitialPrice = Number(quote?.price);
      const initialPrice =
        Number.isFinite(candidateInitialPrice) && candidateInitialPrice > 0
          ? candidateInitialPrice
          : DEFAULT_SETTINGS.fallbackInitialPrice;
      const volatility = deriveVolatilityFromQuote(quote?.change_percent);

      const configuredAgents = previewAgents;
      const next = await startSimulation({
        ticker: cleanTicker,
        target_tickers: resolvedTickers,
        duration_seconds: DEFAULT_SETTINGS.duration,
        initial_price: initialPrice,
        starting_cash: Math.max(1000, Number(startingCapital) || DEFAULT_SETTINGS.startingCash),
        volatility,
        inference_runtime: "modal",
        agents: configuredAgents
      });
      setSession(next);
      setSessionAgents(configuredAgents);
      setPriceSeries([{ tick: 0, price: next.current_price }]);
      setAutoCommentary("");
      setAutoCommentaryModel("");
      setModalSandboxResult(null);

      setModalSandboxLoading(true);
      const sandboxPrompt = buildSandboxPrompt(cleanTicker, configuredAgents, resolvedTickers);
      try {
        const sandbox = await spinModalSandbox(sandboxPrompt, next.session_id);
        setModalSandboxResult(sandbox);
        if (sandbox.status === "failed") {
          setModalSandboxError(sandbox.error || sandbox.hint || "Modal sandbox launch failed.");
        }
      } catch {
        setModalSandboxResult(null);
        setModalSandboxError("Modal sandbox launch failed. Simulation continues on local engine.");
      } finally {
        setModalSandboxLoading(false);
      }
    } catch {
      setSessionStartError(
        "Unable to start simulation. Try a known stock ticker or reset to your watchlist symbols."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handlePause() {
    if (!session?.running || session.paused) return;
    setLoading(true);
    try {
      const paused = await pauseSimulation(session.session_id);
      setSession(paused);
    } finally {
      setLoading(false);
    }
  }

  async function handleStop() {
    if (!session?.running) return;
    setLoading(true);
    try {
      await stopSimulation(session.session_id);
      const finalState = { ...session, running: false, paused: false };
      setSession(finalState);
      void generateAutoCommentary("stopped", finalState);
    } finally {
      setLoading(false);
    }
  }

  async function handleReset() {
    setLoading(true);
    try {
      if (session?.running) {
        await stopSimulation(session.session_id);
      }
      setSession(null);
      setSessionAgents([]);
      setPriceSeries([]);
      setAutoCommentary("");
      setAutoCommentaryModel("");
      setAutoCommentaryLoading(false);
      setModalSandboxResult(null);
      setModalSandboxLoading(false);
      setModalSandboxError("");
      setSessionStartError("");
      setPromptInferredTickers([]);
      setAddedCustomAgents([]);
      setCustomEmoji("");
      setCustomAgentSequence(2);
      setAgentDialogOpen(false);
      setAgentDialogMode("create");
      setEditingAgentName(null);
      setStartingCapital(DEFAULT_SETTINGS.startingCash);
      setCustomName(DEFAULT_CUSTOM.name);
      setCustomPrompt(DEFAULT_CUSTOM.prompt);
      setCustomRisk(DEFAULT_CUSTOM.risk);
      setCustomTempo(DEFAULT_CUSTOM.tempo);
      setCustomStyle(DEFAULT_CUSTOM.style);
      setCustomNews(DEFAULT_CUSTOM.news);
    } finally {
      setLoading(false);
    }
  }

  function formatDateTime(value: string) {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString();
  }

  function handleDownloadReport() {
    if (!session || session.running) return;

    const reportTime = new Date();
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 40;
    const contentWidth = pageWidth - margin * 2;
    let y = margin;

    const ensureSpace = (height: number) => {
      if (y + height > pageHeight - margin) {
        doc.addPage();
        y = margin;
      }
    };

    const addLine = (text: string, size = 11, bold = false, gapBefore = 0, gapAfter = 6) => {
      y += gapBefore;
      doc.setFont("helvetica", bold ? "bold" : "normal");
      doc.setFontSize(size);
      const wrapped = doc.splitTextToSize(text, contentWidth);
      const lineHeight = Math.max(14, size + 2);
      ensureSpace(wrapped.length * lineHeight + gapAfter);
      doc.text(wrapped, margin, y);
      y += wrapped.length * lineHeight + gapAfter;
    };

    const rankedAgents = Object.entries(session.portfolios)
      .map(([agent, portfolio]) => {
        const totalPnl =
          portfolio.total_pnl ??
          ((portfolio.realized_pnl ?? 0) + (portfolio.unrealized_pnl ?? 0));
        return {
          agent,
          equity: portfolio.equity ?? 0,
          totalPnl
        };
      })
      .sort((a, b) => b.equity - a.equity);

    addLine("TickerMaster Simulation Report", 20, true, 0, 12);
    addLine(`Generated: ${reportTime.toLocaleString()}`, 10, false, 0, 10);

    addLine("Session Summary", 13, true, 6, 6);
    addLine(`Primary ticker: ${session.ticker}`);
    addLine(`Universe: ${session.tickers.join(", ") || session.ticker}`);
    addLine(`Session ID: ${session.session_id}`, 10);
    addLine(`Started: ${formatDateTime(session.started_at)}`, 10);
    addLine(`Ended: ${formatDateTime(session.ends_at)}`, 10);
    addLine(`Final tick: ${session.tick}`);
    addLine(`Final price: ${formatCurrency(session.current_price)}`);
    addLine(`Combined equity: ${formatCurrency(totalEquity)}`);
    addLine(`Total trades: ${session.trades.length}`);

    addLine("Top Agents", 13, true, 8, 6);
    if (rankedAgents.length === 0) {
      addLine("No agent portfolio data available.", 10);
    } else {
      rankedAgents.slice(0, 8).forEach((row, index) => {
        const pnlLabel = `${row.totalPnl >= 0 ? "+" : ""}${formatCurrency(row.totalPnl)}`;
        addLine(
          `${index + 1}. ${row.agent} | Equity ${formatCurrency(row.equity)} | Net PnL ${pnlLabel}`,
          10
        );
      });
    }

    addLine("Latest News", 13, true, 8, 6);
    if (session.recent_news.length === 0) {
      addLine("No headlines captured for this run.", 10);
    } else {
      session.recent_news.slice(0, 10).forEach((news, index) => addLine(`${index + 1}. ${news}`, 10));
    }

    addLine("Recent Trades", 13, true, 8, 6);
    if (session.trades.length === 0) {
      addLine("No executed trades.", 10);
    } else {
      session.trades.slice(0, 25).forEach((trade, index) => {
        addLine(
          `${index + 1}. ${trade.agent} | ${trade.side.toUpperCase()} ${trade.quantity} ${trade.ticker} @ ${trade.price.toFixed(2)}`,
          10
        );
      });
    }

    addLine("Post-Run Commentary", 13, true, 8, 6);
    addLine(
      autoCommentaryLoading
        ? "Commentary is still generating."
        : autoCommentary || "No automatic commentary was generated.",
      10
    );

    const stamp = reportTime.toISOString().replace(/[:.]/g, "-");
    doc.save(`tickermaster-report-${session.ticker}-${stamp}.pdf`);
  }

  const totalEquity = useMemo(() => {
    if (!session) return 0;
    return Object.values(session.portfolios).reduce((sum, portfolio) => sum + (portfolio.equity ?? 0), 0);
  }, [session]);

  const latestTradeByAgent = useMemo(() => {
    const map = new Map<string, TradeRecord>();
    for (const trade of session?.trades ?? []) {
      if (!map.has(trade.agent)) {
        map.set(trade.agent, trade);
      }
    }
    return map;
  }, [session?.trades]);

  const portfolioLeaderboard = useMemo(() => {
    if (!session) return [];
    return Object.entries(session.portfolios)
      .map(([agent, portfolio]) => {
        const totalPnl =
          portfolio.total_pnl ??
          ((portfolio.realized_pnl ?? 0) + (portfolio.unrealized_pnl ?? 0));
        return {
          agent,
          equity: portfolio.equity ?? 0,
          totalPnl
        };
      })
      .sort((a, b) => b.equity - a.equity);
  }, [session]);

  const topPerformer = portfolioLeaderboard[0] ?? null;
  const sessionFinished = Boolean(session && !session.running);

  const customEditorDisabled = Boolean(session?.running);
  const customEmojiPreview = normalizeEmoji(customEmoji);
  const sandboxRunning = Boolean(session?.running) && modalSandboxResult?.status === "started";
  const normalizedStartingCapital = Math.max(1000, Number(startingCapital) || DEFAULT_SETTINGS.startingCash);
  const myAgents = displayedAgents.filter((agent) => userAgentNames.has(agent.name));
  const arenaAgents = displayedAgents.filter((agent) => !userAgentNames.has(agent.name));

  useEffect(() => {
    if (!customEditorDisabled || !agentDialogOpen) return;
    setAgentDialogOpen(false);
    setAgentDialogMode("create");
    setEditingAgentName(null);
  }, [customEditorDisabled, agentDialogOpen]);

  const renderRoundtableCard = (agent: AgentConfig) => {
    const latestTrade = latestTradeByAgent.get(agent.name);
    const portfolio = session?.portfolios[agent.name];
    const positionRows = positionRowsForCard(portfolio);
    const totalPnl =
      portfolio?.total_pnl ??
      ((portfolio?.realized_pnl ?? 0) + (portfolio?.unrealized_pnl ?? 0));
    const isPreset = institutionalAgentNames.has(agent.name);
    const status = buildAgentStatus(session, latestTrade, isPreset);
    const tone = statusTone(status);
    const isSelf = userAgentNames.has(agent.name);
    const isEditableSelf = editableAgentNames.has(agent.name);
    const logo = AGENT_LOGOS[agent.name];
    const emoji = customAgentEmojis[agent.name];

    return (
      <article
        key={agent.name}
        className={`roundtable-card${isPreset ? " preset" : ""}${isSelf ? " self" : ""}`}
      >
        <div className="roundtable-top">
          <span className="character-avatar" aria-hidden="true">
            {logo ? <img className="character-avatar-logo" src={logo} alt={`${agent.name} logo`} /> : emoji ?? (isSelf ? "🧠" : "🤖")}
          </span>
          {isEditableSelf ? (
            <div className="character-card-actions">
              <button
                type="button"
                className="card-icon-button"
                onClick={() => openEditAgentDialog(agent)}
                disabled={customEditorDisabled}
                aria-label={`Edit ${agent.name}`}
                title={`Edit ${agent.name}`}
              >
                ✎
              </button>
              <button
                type="button"
                className="card-icon-button danger"
                onClick={() => handleRemoveAddedAgent(agent.name)}
                disabled={customEditorDisabled}
                aria-label={`Delete ${agent.name}`}
                title={`Delete ${agent.name}`}
              >
                🗑
              </button>
            </div>
          ) : (
            <span className={`character-status-chip ${tone}`}>{status}</span>
          )}
        </div>
        <h4 className="character-name">{agent.name}</h4>
        <p className="character-role">{personalityLabel(agent.personality)}</p>
        <p className="character-equity">{portfolio ? formatCurrency(portfolio.equity ?? 0) : "-"}</p>
        <p className={`character-total-pnl ${totalPnl >= 0 ? "text-green" : "text-red"}`}>
          {totalPnl >= 0 ? "+" : ""}
          {formatCurrency(totalPnl)}
        </p>
        {positionRows.length > 0 ? (
          <ul className="character-positions">
            {positionRows.map((row) => (
              <li key={`${agent.name}-${row.ticker}`}>
                <span>{row.ticker} · {Math.round(row.holdings)} sh</span>
                <span className={row.netGain >= 0 ? "text-green" : "text-red"}>
                  {row.netGain >= 0 ? "+" : ""}
                  {formatCurrency(row.netGain)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted character-positions-empty">No active positions</p>
        )}
      </article>
    );
  };

  return (
    <section className="panel stack stagger">
      {agentDialogOpen ? (
        <div className="agent-dialog-backdrop" role="presentation" onClick={handleCloseAgentDialog}>
          <div
            className="glass-card stack sim-card sim-card-agent agent-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="agent-dialog-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="panel-header sim-panel-header agent-dialog-header">
              <div className="sim-header-copy">
                <h3 id="agent-dialog-title">{agentDialogMode === "edit" ? "Edit Agent" : "Build Your Agent"}</h3>
              </div>
              <button
                type="button"
                className="secondary agent-dialog-close"
                onClick={handleCloseAgentDialog}
                aria-label="Close agent builder"
              >
                ✕
              </button>
            </div>

            <div className="custom-agent-grid">
              <label>
                Agent Name
                <input
                  value={customName}
                  onChange={(event) => setCustomName(event.target.value)}
                  maxLength={40}
                  placeholder=""
                  disabled={customEditorDisabled}
                />
              </label>
              <label>
                Agent Emoji
                <div className="custom-agent-icon-row">
                  <input
                    className="custom-agent-emoji-input"
                    type="text"
                    maxLength={4}
                    value={customEmoji}
                    onChange={(event) => setCustomEmoji(event.target.value)}
                    placeholder=""
                    disabled={customEditorDisabled}
                  />
                  <span className="custom-agent-icon-preview" aria-hidden="true">
                    {customEmojiPreview}
                  </span>
                  <span className="muted">Default with 🧠</span>
                </div>
              </label>
            </div>

            <label className="custom-agent-prompt-row">
              Strategy Prompt
              <textarea
                value={customPrompt}
                onChange={(event) => setCustomPrompt(event.target.value)}
                rows={6}
                placeholder=""
                disabled={customEditorDisabled}
              />
            </label>

            <div className="strategy-template-row">
              <span className="muted">Quick Templates</span>
              <div className="strategy-template-chips">
                {STRATEGY_TEMPLATES.map((template) => (
                  <button
                    key={template.label}
                    type="button"
                    className="secondary strategy-template-chip"
                    onClick={() => setCustomPrompt(template.prompt)}
                    disabled={customEditorDisabled}
                  >
                    {template.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="custom-slider-grid">
              <label>
                Risk Level ({customRisk})
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={customRisk}
                  style={{ "--range-progress": `${customRisk}%` } as CSSProperties}
                  onChange={(event) => setCustomRisk(Number(event.target.value))}
                  disabled={customEditorDisabled}
                />
              </label>
              <label>
                Trade Frequency ({customTempo})
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={customTempo}
                  style={{ "--range-progress": `${customTempo}%` } as CSSProperties}
                  onChange={(event) => setCustomTempo(Number(event.target.value))}
                  disabled={customEditorDisabled}
                />
              </label>
              <label>
                Style ({styleLabel(customStyle)})
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={customStyle}
                  style={{ "--range-progress": `${customStyle}%` } as CSSProperties}
                  onChange={(event) => setCustomStyle(Number(event.target.value))}
                  disabled={customEditorDisabled}
                />
              </label>
              <label>
                News Reactivity ({customNews})
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={customNews}
                  style={{ "--range-progress": `${customNews}%` } as CSSProperties}
                  onChange={(event) => setCustomNews(Number(event.target.value))}
                  disabled={customEditorDisabled}
                />
              </label>
            </div>

            <div className="custom-agent-meta">
              <span className="pill neutral">{personalityLabel(customAgentConfig.personality)}</span>
              <span className="muted">
                Trade Size {customAgentConfig.trade_size} · Risk Limit {customAgentConfig.risk_limit.toFixed(2)} · Aggression {customAgentConfig.aggressiveness.toFixed(2)}
              </span>
            </div>

            <button
              type="button"
              className="add-agent-button"
              onClick={handleSaveAgentDialog}
              disabled={customEditorDisabled || customPrompt.trim().length === 0}
            >
              {agentDialogMode === "edit" ? "Save Agent" : "+ Add New Agent To Roundtable"}
            </button>
          </div>
        </div>
      ) : null}

      <div id="sim-roundtable" className="glass-card stack sim-card sim-card-roundtable">
        <div className="panel-header sim-panel-header">
          <div className="sim-header-copy">
            <h3>AI Roundtable</h3>
          </div>
          <div className="roundtable-header-actions">
            <button
              type="button"
              className="add-agent-button roundtable-add-button"
              onClick={openCreateAgentDialog}
              disabled={customEditorDisabled}
              aria-label="Add agent"
            >
              + Add Agent
            </button>
          </div>
        </div>
        <div className="roundtable-section roundtable-section-self">
          <div className="roundtable-section-header">
            <h4>My Agents</h4>
            <span className="pill amber">{myAgents.length}</span>
          </div>
          {myAgents.length > 0 ? (
            <div className="roundtable-grid">
              {myAgents.map((agent) => renderRoundtableCard(agent))}
            </div>
          ) : (
            <p className="muted roundtable-empty">No custom agents yet. Click Add Agent to build one.</p>
          )}
        </div>

        <div className="roundtable-section roundtable-section-arena">
          <div className="roundtable-section-header">
            <h4>Arena Agents</h4>
            <span className="pill arena">{arenaAgents.length}</span>
          </div>
          <div className="roundtable-grid">
            {arenaAgents.map((agent) => renderRoundtableCard(agent))}
          </div>
        </div>
      </div>

      <div id="sim-session" className="glass-card stack session-control-card sim-card sim-card-session">
        <div className="panel-header sim-panel-header">
          <div className="sim-header-copy">
            <h3>Session Play</h3>
          </div>
          <span className={sandboxRunning ? "pill bullish" : "pill bearish"}>
            Sandbox {sandboxRunning ? "Running" : "Not Running"}
          </span>
        </div>
        <div className="session-capital-spotlight">
          <label className="session-capital-field">
            <span>Starting Capital</span>
            <input
              className="session-capital-input"
              type="number"
              min={1000}
              step={1000}
              value={startingCapital}
              onChange={(event) => setStartingCapital(Number(event.target.value) || DEFAULT_SETTINGS.startingCash)}
              disabled={loading || Boolean(session?.running)}
            />
          </label>
          <p className="session-capital-preview">{formatCurrency(normalizedStartingCapital)}</p>
        </div>
        <p className="sim-tip">Tip: keep starting capital near 100,000 for easier strategy comparisons.</p>
        {modalSandboxLoading ? <p className="muted">Launching Modal sandbox…</p> : null}
        {modalHealth?.status === "missing_dependency" && modalHealth.install_hint ? (
          <p className="error">{modalHealth.install_hint}</p>
        ) : null}
        {modalHealthError ? <p className="error">{modalHealthError}</p> : null}
        {modalSandboxError ? <p className="error">{modalSandboxError}</p> : null}
        {sessionStartError ? <p className="error">{sessionStartError}</p> : null}

        {!session ? (
          <button className="start-trading-button" onClick={handlePlay} disabled={loading}>
            ▶ Start Trading
          </button>
        ) : session.running ? (
          <div className="simulation-control-row">
            <button
              className={session.paused ? "" : "secondary"}
              onClick={session.paused ? handlePlay : handlePause}
              disabled={loading || !session.running}
            >
              {session.paused ? "▶ Continue" : "⏸ Pause"}
            </button>
            <button className="secondary" onClick={handleStop} disabled={loading || !session.running}>
              ⏹ Stop / Cancel
            </button>
            <button className="secondary" onClick={handleReset} disabled={loading}>
              ↺ Reset (Clean Slate)
            </button>

            <span className={session.paused ? "pill neutral" : "pill bullish"}>
              {session.paused ? "Paused" : "Live"}
            </span>
          </div>
        ) : (
          <div className="post-session-actions">
            <button className="secondary" onClick={handleDownloadReport} disabled={loading}>
              ⬇ Save Report to PDF
            </button>
            <button className="start-trading-button" onClick={handlePlay} disabled={loading}>
              ▶ Start New Trade
            </button>
          </div>
        )}
      </div>

      <div id="sim-market" className="telemetry-board-row sim-market-row">
        <div
          className={`glass-card sim-card sim-card-telemetry${telemetryHeightSynced ? " sim-card-telemetry-synced" : ""}`}
          style={telemetryHeightSynced && telemetryCardHeight ? { height: `${telemetryCardHeight}px` } : undefined}
        >
          <div className="panel-header sim-panel-header">
            <div className="sim-header-copy">
              <h3>Session Telemetry</h3>
            </div>
            <span className={session?.crash_mode ? "pill bearish" : "pill neutral"}>
              {session?.crash_mode ? "Crash Regime" : "Normal Regime"}
            </span>
          </div>
          <div className="kpi-grid">
            <div>
              <p className="muted">Price</p>
              <h3>{session ? formatCurrency(session.current_price) : "-"}</h3>
            </div>
            <div>
              <p className="muted">Tick</p>
              <h3>{session?.tick ?? 0}</h3>
            </div>
            <div>
              <p className="muted">Combined Equity</p>
              <h3>{formatCurrency(totalEquity)}</h3>
            </div>
          </div>
          <div className="chart-box telemetry-chart-box">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={priceSeries}>
                <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
                <XAxis dataKey="tick" stroke="var(--muted)" />
                <YAxis stroke="var(--muted)" domain={["auto", "auto"]} />
                <Tooltip contentStyle={{ background: "var(--surface-2)", border: "1px solid var(--line)", borderRadius: 12 }} />
                <Line dataKey="price" type="monotone" stroke="var(--accent-2)" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div ref={marketBoardWrapRef} className="sim-market-board-wrap">
          <EventRail
            connected={connected}
            simulationEvent={simulationEvent}
            activeSessionId={session?.session_id ?? null}
            simulationActive={Boolean(session?.running)}
            className="event-rail-inline"
          />
        </div>
      </div>

      <div className="card-row-split sim-review-grid">
        <div className="glass-card sim-card sim-card-news">
          <div className="panel-header sim-panel-header">
            <div className="sim-header-copy">
              <h3>Latest News</h3>
            </div>
            <span className="muted">Catalysts that move behavior</span>
          </div>
          <ul className="news-feed">
            {(session?.recent_news ?? []).slice(0, 8).map((news, idx) => {
              const parsed = parseSimulationNewsItem(news);
              const tone = inferNewsTone(parsed.headline);
              return (
                <li key={`news-${idx}`} className={`news-feed-item news-${tone}`}>
                  {parsed.ticker || parsed.source ? (
                    <div className="news-feed-meta">
                      {parsed.ticker ? <span className="news-feed-ticker">{parsed.ticker}</span> : null}
                      {parsed.source ? <span className="news-feed-source">{parsed.source}</span> : null}
                    </div>
                  ) : null}
                  <p className="news-feed-headline">{parsed.headline}</p>
                </li>
              );
            })}
            {(session?.recent_news ?? []).length === 0 ? <li className="muted">No headlines yet.</li> : null}
          </ul>
        </div>

        <div className="glass-card sim-card sim-card-trades">
          <div className="panel-header sim-panel-header">
            <div className="sim-header-copy">
              <h3>Trades</h3>
            </div>
            <span className="muted">Execution log by ticker and agent</span>
          </div>
          <div className="table-wrap trades-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Agent</th>
                  <th>Side</th>
                  <th>Qty</th>
                  <th>Price</th>
                </tr>
              </thead>
              <tbody>
                {(session?.trades ?? []).map((trade, idx) => (
                  <tr key={`trade-${idx}`}>
                    <td>{trade.ticker}</td>
                    <td>{trade.agent}</td>
                    <td className={trade.side === "buy" ? "text-green" : "text-red"}>{trade.side}</td>
                    <td>{trade.quantity}</td>
                    <td>{trade.price.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div id="sim-review" className="glass-card stack sim-card sim-card-commentary">
        <div className="panel-header sim-panel-header">
          <div className="sim-header-copy">
            <h3>Post-Run Commentary</h3>
          </div>
          <span className="muted">Auto-generated when simulation ends</span>
        </div>
        {sessionFinished ? (
          <div className="postrun-structured">
            <section className="postrun-row">
              <h4>Session Outcome:</h4>
              <p>
                {session?.ticker} closed at {formatCurrency(session?.current_price ?? 0)} after {session?.tick ?? 0} ticks with{" "}
                {session?.trades.length ?? 0} trades executed.
              </p>
            </section>

            <section className="postrun-row">
              <h4>🏆 Top Performer:</h4>
              <p>
                {topPerformer
                  ? `${topPerformer.agent} at ${formatCurrency(topPerformer.equity)} (${topPerformer.totalPnl >= 0 ? "+" : ""}${formatCurrency(topPerformer.totalPnl)} net PnL).`
                  : "No portfolio winner available for this run."}
              </p>
            </section>

            <section className="postrun-row">
              <h4>📘 Educational Takeaway:</h4>
              <p>
                {autoCommentaryLoading
                  ? "Generating commentary…"
                  : autoCommentary || "No automatic commentary was generated."}
              </p>
              {autoCommentaryModel ? <span className="postrun-meta">Model: {autoCommentaryModel}</span> : null}
            </section>
          </div>
        ) : (
          <p className="muted">
            Finish or stop a simulation to view a structured post-run breakdown.
          </p>
        )}
      </div>
    </section>
  );
}
