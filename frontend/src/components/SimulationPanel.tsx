import { useEffect, useMemo, useState, type CSSProperties } from "react";
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
import { formatCurrency, formatPercent } from "../lib/format";
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

type RuntimeMode = "modal" | "local";

const ALL_MARKET_PROMPT_PATTERN =
  /\b(all stocks|all tickers|entire market|whole market|market-wide|across the market)\b/i;

const TICKER_PROMPT_STOPWORDS = new Set([
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
  "WHEN",
  "WITH"
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

function extractTickerCandidatesFromPrompt(prompt: string): string[] {
  const matches = prompt.match(/\b[A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?\b/g) ?? [];
  const seen = new Set<string>();
  const out: string[] = [];

  for (const token of matches) {
    const symbol = normalizeSymbol(token);
    if (!symbol) continue;
    if (TICKER_PROMPT_STOPWORDS.has(symbol)) continue;
    if (!/^[A-Z][A-Z0-9-]{0,9}$/.test(symbol)) continue;
    if (!seen.has(symbol)) {
      seen.add(symbol);
      out.push(symbol);
    }
  }

  return out;
}

function resolveTickersFromPrompt(prompt: string, activeTicker: string, watchlist: string[]): string[] {
  const inferred = extractTickerCandidatesFromPrompt(prompt);
  if (inferred.length > 0) return inferred;

  if (ALL_MARKET_PROMPT_PATTERN.test(prompt)) {
    const normalizedWatchlist = watchlist.map(normalizeSymbol).filter(Boolean);
    if (normalizedWatchlist.length > 0) {
      const deduped = Array.from(new Set(normalizedWatchlist));
      if (deduped.includes("SPY")) {
        return ["SPY", ...deduped.filter((symbol) => symbol !== "SPY")];
      }
      return deduped;
    }
  }

  const fallback = normalizeSymbol(activeTicker) || "AAPL";
  return [fallback];
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
  const [editingAgentName, setEditingAgentName] = useState<string | null>(null);
  const [editingAgentPrompt, setEditingAgentPrompt] = useState("");

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
  const [runtimeMode, setRuntimeMode] = useState<RuntimeMode>("modal");
  const [modalHealth, setModalHealth] = useState<ModalCronHealthResponse | null>(null);
  const [modalHealthError, setModalHealthError] = useState("");
  const [modalSandboxResult, setModalSandboxResult] = useState<ModalSandboxResponse | null>(null);
  const [modalSandboxLoading, setModalSandboxLoading] = useState(false);
  const [modalSandboxError, setModalSandboxError] = useState("");
  const [promptInferredTickers, setPromptInferredTickers] = useState<string[]>([]);

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
    if (!session || simulationEvent.session_id !== session.session_id) return;

    const nextTick = Number(simulationEvent.tick ?? session.tick);
    const nextPrice = Number(simulationEvent.price ?? session.current_price);

    setPriceSeries((prev) => [...prev, { tick: nextTick, price: nextPrice }].slice(-200));

    setSession((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        tick: nextTick,
        current_price: nextPrice,
        paused: false,
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
  }, [simulationEvent, session]);

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
  const institutionalAgentNames = useMemo(
    () => new Set(INSTITUTIONAL_AGENTS.map((agent) => agent.name)),
    []
  );

  function handleAddNewAgent() {
    if (customEditorDisabled) return;
    const takenNames = new Set([
      ...INSTITUTIONAL_AGENTS.map((agent) => agent.name),
      ...addedCustomAgents.map((entry) => entry.config.name)
    ]);
    const uniqueName = makeUniqueName(customAgentConfig.name, takenNames);

    setAddedCustomAgents((previous) => [
      ...previous,
      {
        config: { ...customAgentConfig, name: uniqueName },
        iconEmoji: normalizeEmoji(customEmoji)
      }
    ]);
    setCustomName(`${SELF_AGENT_NAME} ${customAgentSequence}`);
    setCustomEmoji("");
    setCustomAgentSequence((value) => value + 1);
  }

  function handleRemoveAddedAgent(name: string) {
    if (customEditorDisabled) return;
    setAddedCustomAgents((previous) => previous.filter((entry) => entry.config.name !== name));
    if (editingAgentName === name) {
      setEditingAgentName(null);
      setEditingAgentPrompt("");
    }
  }

  function handleEditAddedAgent(entry: UserAgentEntry) {
    if (customEditorDisabled) return;
    setEditingAgentName(entry.config.name);
    setEditingAgentPrompt(entry.config.strategy_prompt ?? "");
  }

  function handleCancelEditAgent() {
    setEditingAgentName(null);
    setEditingAgentPrompt("");
  }

  function handleSaveEditedAgent(name: string) {
    if (customEditorDisabled) return;
    const prompt = editingAgentPrompt.trim();
    if (!prompt) return;

    setAddedCustomAgents((previous) =>
      previous.map((entry) =>
        entry.config.name === name
          ? { ...entry, config: { ...entry.config, strategy_prompt: prompt } }
          : entry
      )
    );
    setEditingAgentName(null);
    setEditingAgentPrompt("");
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
      const resolvedTickers = resolveTickersFromPrompt(strategyContext, activeTicker, watchlist);
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

      const initialPrice = quote?.price ?? DEFAULT_SETTINGS.fallbackInitialPrice;
      const volatility = deriveVolatilityFromQuote(quote?.change_percent);

      const configuredAgents = previewAgents;
      const next = await startSimulation({
        ticker: cleanTicker,
        duration_seconds: DEFAULT_SETTINGS.duration,
        initial_price: initialPrice,
        starting_cash: Math.max(1000, Number(startingCapital) || DEFAULT_SETTINGS.startingCash),
        volatility,
        inference_runtime: runtimeMode === "modal" ? "modal" : "direct",
        agents: configuredAgents
      });
      setSession(next);
      setSessionAgents(configuredAgents);
      setPriceSeries([{ tick: 0, price: next.current_price }]);
      setAutoCommentary("");
      setAutoCommentaryModel("");
      setModalSandboxResult(null);

      if (runtimeMode === "modal") {
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
      }
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
      setPromptInferredTickers([]);
      setAddedCustomAgents([]);
      setCustomEmoji("");
      setCustomAgentSequence(2);
      setEditingAgentName(null);
      setEditingAgentPrompt("");
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

  const customEditorDisabled = Boolean(session?.running);
  const customEmojiPreview = normalizeEmoji(customEmoji);
  const displayTicker = activeTicker.trim().toUpperCase() || "AAPL";
  const quoteLine = quoteLoading
    ? "Loading live quote…"
    : liveQuote
      ? `${formatCurrency(liveQuote.price)} · ${formatPercent(liveQuote.change_percent)}`
      : quoteError || `Fallback quote: ${formatCurrency(DEFAULT_SETTINGS.fallbackInitialPrice)}`;
  const modalHealthLine = modalHealth
    ? `${modalHealth.status.toUpperCase()} · ${modalHealth.message}`
    : modalHealthError || "Modal runtime status unavailable.";
  const modalRunLine = modalSandboxResult
    ? `${modalSandboxResult.status.toUpperCase()}${modalSandboxResult.sandbox_id ? ` · ${modalSandboxResult.sandbox_id}` : ""}`
    : "No sandbox launched yet.";
  const inferenceRuntimeLine =
    runtimeMode === "modal"
      ? "Modal function first, then OpenRouter fallback."
      : "Direct OpenRouter from backend.";
  const inferenceFunctionLine =
    modalHealth?.inference_function_name
      ? `${modalHealth.inference_function_name} (${modalHealth.inference_timeout_seconds ?? 15}s timeout)`
      : "Not configured";
  const promptUniverseLine =
    promptInferredTickers.length > 0 ? promptInferredTickers.join(", ") : "No ticker inferred from prompt yet.";

  return (
    <section className="panel stack stagger">
      <header className="panel-header">
        <h2>Simulation Arena</h2>
        <p>Live roundtable sandbox with realistic market impact, slippage, and asynchronous info flow.</p>
      </header>

      <div className="glass-card stack">
        <div className="panel-header">
          <h3>Create Custom Agent</h3>
          <span className="muted">Test your trading strategy prompt</span>
        </div>

        <div className="custom-agent-grid">
          <label>
            Agent Name
            <input
              value={customName}
              onChange={(event) => setCustomName(event.target.value)}
              maxLength={40}
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
                placeholder="Input emoji"
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
            Risk Appetite ({customRisk})
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
            Trading Tempo ({customTempo})
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
          onClick={handleAddNewAgent}
          disabled={customEditorDisabled}
        >
          + Add New Agent To Roundtable
        </button>

        {addedCustomAgents.length > 0 ? (
          <div className="custom-agent-roster">
            {addedCustomAgents.map((entry) => (
              <div key={entry.config.name} className="custom-agent-roster-item">
                <div className="custom-agent-roster-head">
                  <span className="custom-agent-roster-name">
                    {entry.iconEmoji ?? DEFAULT_CUSTOM.emoji} {entry.config.name}
                  </span>
                  <span className="pill neutral">{personalityLabel(entry.config.personality)}</span>
                </div>

                {editingAgentName === entry.config.name ? (
                  <textarea
                    className="custom-agent-roster-edit"
                    value={editingAgentPrompt}
                    onChange={(event) => setEditingAgentPrompt(event.target.value)}
                    rows={4}
                    disabled={customEditorDisabled}
                  />
                ) : (
                  <p className="custom-agent-roster-prompt">{entry.config.strategy_prompt || "No prompt set yet."}</p>
                )}

                <div className="custom-agent-roster-actions">
                  {editingAgentName === entry.config.name ? (
                    <>
                      <button
                        type="button"
                        className="secondary"
                        onClick={handleCancelEditAgent}
                        disabled={customEditorDisabled}
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onClick={() => handleSaveEditedAgent(entry.config.name)}
                        disabled={customEditorDisabled || editingAgentPrompt.trim().length === 0}
                      >
                        Save Prompt
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      className="secondary"
                      onClick={() => handleEditAddedAgent(entry)}
                      disabled={customEditorDisabled}
                    >
                      Edit Prompt
                    </button>
                  )}
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => handleRemoveAddedAgent(entry.config.name)}
                    disabled={customEditorDisabled}
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </div>

      <div className="glass-card stack">
        <div className="panel-header">
          <h3>AI Roundtable</h3>
          <span className="muted">Character-card view of all participants</span>
        </div>
        <div className="roundtable-grid">
          {displayedAgents.map((agent) => {
            const latestTrade = latestTradeByAgent.get(agent.name);
            const portfolio = session?.portfolios[agent.name];
            const isPreset = institutionalAgentNames.has(agent.name);
            const status = buildAgentStatus(session, latestTrade, isPreset);
            const tone = statusTone(status);
            const isSelf = userAgentNames.has(agent.name);
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
                  <span className={`character-status-chip ${tone}`}>{status}</span>
                </div>
                <h4 className="character-name">{agent.name}</h4>
                <p className="character-role">{personalityLabel(agent.personality)}</p>
                <p className="character-equity">{portfolio ? formatCurrency(portfolio.equity ?? 0) : "-"}</p>
              </article>
            );
          })}
        </div>
      </div>

      <div className="glass-card stack session-control-card">
        <div className="panel-header">
          <h3>Session Play</h3>
          <span className="muted">Live input source: {displayTicker}</span>
        </div>
        <p className="muted">
          {quoteLine} · Uses live quote + dynamic volatility + default arena bankroll.
        </p>

        <label className="session-runtime-field">
          Sandbox Runtime
          <select
            value={runtimeMode}
            onChange={(event) => setRuntimeMode(event.target.value as RuntimeMode)}
            disabled={loading || Boolean(session?.running)}
          >
            <option value="modal">Modal Sandbox</option>
            <option value="local">Local Engine</option>
          </select>
        </label>

        <div className="runtime-status-card">
          <p className="muted">Modal Health: {modalHealthLine}</p>
          <p className="muted">Inference Runtime: {inferenceRuntimeLine}</p>
          <p className="muted">Inference Function: {inferenceFunctionLine}</p>
          <p className="muted">Prompt Universe: {promptUniverseLine}</p>
          {runtimeMode === "modal" ? <p className="muted">Current Sandbox: {modalRunLine}</p> : null}
          {modalSandboxLoading ? <p className="muted">Launching Modal sandbox…</p> : null}
          {modalHealth?.status === "missing_dependency" && modalHealth.install_hint ? (
            <p className="error">{modalHealth.install_hint}</p>
          ) : null}
          {modalSandboxError ? <p className="error">{modalSandboxError}</p> : null}
          {modalSandboxResult?.dashboard_url ? (
            <a href={modalSandboxResult.dashboard_url} target="_blank" rel="noreferrer">
              Open Modal App Dashboard
            </a>
          ) : null}
        </div>

        <label className="session-capital-field">
          Starting Capital
          <input
            type="number"
            min={1000}
            step={1000}
            value={startingCapital}
            onChange={(event) => setStartingCapital(Number(event.target.value) || DEFAULT_SETTINGS.startingCash)}
            disabled={loading || Boolean(session?.running)}
          />
        </label>

        <button
          className="start-trading-button"
          onClick={handlePlay}
          disabled={loading || Boolean(session?.running && !session?.paused)}
        >
          ▶ {session?.running && session.paused ? "Resume Trading" : "Start Trading"}
        </button>

        <div className="simulation-control-row">
          <button className="secondary" onClick={handlePause} disabled={loading || !session?.running || Boolean(session?.paused)}>
            ⏸ Pause
          </button>
          <button className="secondary" onClick={handleStop} disabled={loading || !session?.running}>
            ⏹ Stop / Cancel
          </button>
          <button className="secondary" onClick={handleReset} disabled={loading}>
            ↺ Reset (Clean Slate)
          </button>

          <span className={session?.running ? (session.paused ? "pill neutral" : "pill bullish") : "pill bearish"}>
            {session?.running ? (session.paused ? "Paused" : "Live") : "Idle"}
          </span>
        </div>
      </div>

      <div className="telemetry-board-row">
        <div className="glass-card">
          <div className="panel-header">
            <h3>Session Telemetry</h3>
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
          <div className="chart-box">
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

        <EventRail connected={connected} simulationEvent={simulationEvent} className="event-rail-inline" />
      </div>

      <div className="card-row card-row-split">
        <div className="glass-card">
          <h3>Order Book</h3>
          <div className="orderbook-grid">
            <div>
              <h4>Bids</h4>
              <ul>
                {(session?.order_book.bids ?? []).map((bid, idx) => (
                  <li key={`bid-${idx}`}>
                    <span>{bid.price.toFixed(2)}</span>
                    <span>{bid.size}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4>Asks</h4>
              <ul>
                {(session?.order_book.asks ?? []).map((ask, idx) => (
                  <li key={`ask-${idx}`}>
                    <span>{ask.price.toFixed(2)}</span>
                    <span>{ask.size}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <h4>News Latency Feed</h4>
          <ul className="news-feed">
            {(session?.recent_news ?? []).slice(0, 5).map((news, idx) => (
              <li key={`news-${idx}`}>{news}</li>
            ))}
          </ul>
        </div>

        <div className="glass-card">
          <h3>Trades</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Agent</th>
                  <th>Side</th>
                  <th>Qty</th>
                  <th>Price</th>
                </tr>
              </thead>
              <tbody>
                {(session?.trades ?? []).slice(0, 14).map((trade, idx) => (
                  <tr key={`trade-${idx}`}>
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

      <div className="glass-card stack">
        <div className="panel-header">
          <h3>Post-Run Commentary</h3>
          <span className="muted">Auto-generated when simulation ends</span>
        </div>
        {autoCommentaryLoading ? <p className="muted">Generating commentary…</p> : null}
        {autoCommentary ? (
          <article className="chat-output">
            <p>{autoCommentary}</p>
            <span className="muted">model: {autoCommentaryModel}</span>
          </article>
        ) : null}
        {!autoCommentary && !autoCommentaryLoading ? (
          <p className="muted">Run a simulation to receive automatic commentary.</p>
        ) : null}
      </div>
    </section>
  );
}
