import { useEffect, useMemo, useState } from "react";
import {
  createTrackerAgent,
  deleteTrackerAgent,
  getTrackerAgentHistory,
  getTrackerSnapshot,
  interactWithTrackerAgent,
  listTrackerAgents,
  queryTrackerAgentContext,
  searchTickerDirectory,
  triggerTrackerPoll
} from "../lib/api";
import { formatCompactNumber, formatCurrency, formatPercent } from "../lib/format";
import { resolveTickerCandidate } from "../lib/tickerInput";
import type { TickerLookup, TrackerAgent, TrackerSnapshot, TrackerSymbolResolution, WSMessage } from "../lib/types";
import WatchlistBar from "./WatchlistBar";

interface Props {
  activeTicker: string;
  onTickerChange: (ticker: string) => void;
  trackerEvent?: WSMessage;
  watchlist: string[];
  onWatchlistChange: (tickers: string[]) => Promise<string[]>;
}

type ScheduleMode = "realtime" | "custom" | "hourly" | "daily";
type ReportMode = "triggers_only" | "periodic" | "hybrid";
type BaselineMode = "prev_close" | "session_open" | "last_check" | "last_alert";

type TrackerCreateForm = {
  symbol: string;
  name: string;
  managerPrompt: string;
  scheduleMode: ScheduleMode;
  startAtLocal: string;
  reportMode: ReportMode;
  runTimeLocal: string;
  baselineMode: BaselineMode;
  autoSimulate: boolean;
};

function normalizeSymbol(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function buildDefaultCreateForm(activeTicker: string): TrackerCreateForm {
  const now = new Date();
  now.setSeconds(0, 0);
  const nowLocal = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}T${String(
    now.getHours(),
  ).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
  return {
    symbol: activeTicker || "",
    name: activeTicker ? `${activeTicker} Tracker Agent` : "",
    managerPrompt: "",
    scheduleMode: "realtime",
    startAtLocal: nowLocal,
    reportMode: "hybrid",
    runTimeLocal: "09:30",
    baselineMode: "prev_close",
    autoSimulate: false
  };
}

export default function TrackerPanel({
  activeTicker,
  onTickerChange,
  trackerEvent,
  watchlist,
  onWatchlistChange
}: Props) {
  const [watchlistInput, setWatchlistInput] = useState("");
  const [watchlistInputFocused, setWatchlistInputFocused] = useState(false);
  const [watchlistSuggestions, setWatchlistSuggestions] = useState<TickerLookup[]>([]);
  const [watchlistSearchLoading, setWatchlistSearchLoading] = useState(false);
  const [snapshot, setSnapshot] = useState<TrackerSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [agents, setAgents] = useState<TrackerAgent[]>([]);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [createError, setCreateError] = useState("");
  const [createNotice, setCreateNotice] = useState("");
  const [createForm, setCreateForm] = useState<TrackerCreateForm>(() => buildDefaultCreateForm(activeTicker));
  const [createSymbolFocused, setCreateSymbolFocused] = useState(false);
  const [createSearchLoading, setCreateSearchLoading] = useState(false);
  const [createSearchError, setCreateSearchError] = useState("");
  const [createSearchResults, setCreateSearchResults] = useState<TickerLookup[]>([]);
  const [instructionModalAgentId, setInstructionModalAgentId] = useState<string | null>(null);
  const [instructionDraft, setInstructionDraft] = useState("");
  const [instructionError, setInstructionError] = useState("");
  const [instructionLoadingAgentId, setInstructionLoadingAgentId] = useState<string | null>(null);
  const [lastSavedInstructionByAgent, setLastSavedInstructionByAgent] = useState<Record<string, string>>({});
  const [managerResponses, setManagerResponses] = useState<Record<string, string>>({});
  const [managerLoadingAgentId, setManagerLoadingAgentId] = useState<string | null>(null);
  const [contextQuestionByAgent, setContextQuestionByAgent] = useState<Record<string, string>>({});
  const [contextAnswerByAgent, setContextAnswerByAgent] = useState<Record<string, string>>({});
  const [contextLoadingAgentId, setContextLoadingAgentId] = useState<string | null>(null);

  useEffect(() => {
    getTrackerSnapshot().then(setSnapshot).catch(() => null);
    void listTrackerAgents()
      .then((items) => setAgents(items))
      .catch(() => setAgents([]));
  }, []);

  useEffect(() => {
    if (!trackerEvent || trackerEvent.type !== "tracker_snapshot") return;
    setSnapshot({
      generated_at: String(trackerEvent.generated_at ?? new Date().toISOString()),
      tickers: Array.isArray(trackerEvent.tickers) ? (trackerEvent.tickers as TrackerSnapshot["tickers"]) : [],
      alerts_triggered: Array.isArray(trackerEvent.alerts) ? (trackerEvent.alerts as TrackerSnapshot["alerts_triggered"]) : []
    });
  }, [trackerEvent]);

  const sorted = useMemo(
    () => [...(snapshot?.tickers ?? [])].sort((a, b) => Math.abs(b.change_percent) - Math.abs(a.change_percent)),
    [snapshot]
  );

  useEffect(() => {
    const query = watchlistInput.trim();
    if (!query) {
      setWatchlistSuggestions([]);
      setWatchlistSearchLoading(false);
      return;
    }

    let active = true;
    const timer = window.setTimeout(() => {
      setWatchlistSearchLoading(true);
      void searchTickerDirectory(query, 8)
        .then((results) => {
          if (!active) return;
          setWatchlistSuggestions(results);
        })
        .catch(() => {
          if (!active) return;
          setWatchlistSuggestions([]);
        })
        .finally(() => {
          if (!active) return;
          setWatchlistSearchLoading(false);
        });
    }, 180);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [watchlistInput]);

  useEffect(() => {
    if (!createModalOpen) return;
    const query = createForm.symbol.trim();
    if (!query) {
      setCreateSearchResults([]);
      setCreateSearchLoading(false);
      return;
    }

    let active = true;
    const timer = window.setTimeout(() => {
      setCreateSearchLoading(true);
      void searchTickerDirectory(query, 8)
        .then((results) => {
          if (!active) return;
          setCreateSearchResults(results);
        })
        .catch(() => {
          if (!active) return;
          setCreateSearchResults([]);
        })
        .finally(() => {
          if (!active) return;
          setCreateSearchLoading(false);
        });
    }, 180);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [createForm.symbol, createModalOpen]);

  const availableWatchlistSuggestions = useMemo(
    () => watchlistSuggestions.filter((entry) => !watchlist.includes(entry.ticker)),
    [watchlistSuggestions, watchlist]
  );
  const selectedInstructionAgent = useMemo(
    () => agents.find((agent) => agent.id === instructionModalAgentId) ?? null,
    [agents, instructionModalAgentId]
  );

  async function handleAddToWatchlist(rawInput?: string) {
    const candidate = rawInput ?? watchlistInput;
    let resolvedTicker = resolveTickerCandidate(candidate, watchlistSuggestions);
    if (!resolvedTicker && candidate.trim()) {
      try {
        const freshSuggestions = await searchTickerDirectory(candidate, 8);
        if (freshSuggestions.length > 0) {
          setWatchlistSuggestions(freshSuggestions);
          resolvedTicker = resolveTickerCandidate(candidate, freshSuggestions);
        }
      } catch {
        // no-op
      }
    }
    if (!resolvedTicker) return;
    if (watchlist.includes(resolvedTicker)) {
      setWatchlistInput("");
      onTickerChange(resolvedTicker);
      return;
    }

    setLoading(true);
    try {
      await onWatchlistChange([...watchlist, resolvedTicker]);
      const refreshed = await triggerTrackerPoll();
      setSnapshot(refreshed);
      onTickerChange(resolvedTicker);
      setWatchlistInput("");
      setWatchlistInputFocused(false);
    } finally {
      setLoading(false);
    }
  }

  async function handlePoll() {
    setLoading(true);
    try {
      const refreshed = await triggerTrackerPoll();
      setSnapshot(refreshed);
    } finally {
      setLoading(false);
    }
  }

  function openCreateModal() {
    setCreateForm(buildDefaultCreateForm(activeTicker));
    setCreateError("");
    setCreateNotice("");
    setCreateSearchError("");
    setCreateSearchResults([]);
    setCreateModalOpen(true);
  }

  function updateCreateForm<K extends keyof TrackerCreateForm>(key: K, value: TrackerCreateForm[K]) {
    setCreateForm((prev) => ({ ...prev, [key]: value }));
  }

  function applyCreateTickerResult(result: TickerLookup) {
    const resolvedTicker = String(result.ticker || "").trim().toUpperCase();
    if (!resolvedTicker) return;
    const currentName = createForm.name.trim();
    updateCreateForm("symbol", resolvedTicker);
    if (!currentName || currentName === `${createForm.symbol} Tracker Agent`) {
      updateCreateForm("name", `${resolvedTicker} Tracker Agent`);
    }
    setCreateSearchError("");
  }

  async function handleCreateAgentWithSettings() {
    const symbol = normalizeSymbol(createForm.symbol);
    const prompt = createForm.managerPrompt.trim();
    if (!symbol && !prompt) {
      setCreateError("Provide a ticker/company or include it in the initial manager instruction.");
      return;
    }
    if (!createForm.startAtLocal.trim()) {
      setCreateError("Start timer is required.");
      return;
    }
    const startAtDate = new Date(createForm.startAtLocal);
    if (Number.isNaN(startAtDate.getTime())) {
      setCreateError("Start timer is invalid.");
      return;
    }
    const startAtIso = startAtDate.toISOString();

    const name = createForm.name.trim() || (symbol ? `${symbol} Tracker Agent` : "Tracker Agent");
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || "America/New_York";

    let pollSeconds = 120;
    let reportSeconds = createForm.reportMode === "triggers_only" ? 120 : 120;
    let customTimeEnabled = false;
    const runTime = createForm.runTimeLocal.trim() || "09:30";

    if (createForm.scheduleMode === "hourly") {
      pollSeconds = 3600;
      reportSeconds = createForm.reportMode === "triggers_only" ? 3600 : 3600;
    } else if (createForm.scheduleMode === "daily") {
      pollSeconds = 86400;
      reportSeconds = createForm.reportMode === "triggers_only" ? 86400 : 86400;
    } else if (createForm.scheduleMode === "custom") {
      customTimeEnabled = true;
      pollSeconds = 86400;
      reportSeconds = createForm.reportMode === "triggers_only" ? 86400 : 86400;
    }

    const runTimeMatch = /^(\d{2}):(\d{2})$/.exec(runTime);
    if ((createForm.scheduleMode === "daily" || createForm.scheduleMode === "custom") && !runTimeMatch) {
      setCreateError("Run time must use HH:MM format.");
      return;
    }

    const triggers: Record<string, unknown> = {
      schedule_mode: createForm.scheduleMode,
      poll_interval_seconds: pollSeconds,
      report_interval_seconds: reportSeconds,
      report_mode: createForm.reportMode,
      baseline_mode: createForm.baselineMode,
      simulate_on_alert: createForm.autoSimulate,
      tool_mode: "auto",
      notification_style: "auto",
      timezone,
      custom_time_enabled: customTimeEnabled
    };
    triggers.start_at = startAtIso;
    if (createForm.scheduleMode === "daily" || createForm.scheduleMode === "custom") {
      triggers.daily_run_time = runTime;
    }

    setCreateLoading(true);
    setCreateError("");
    setCreateNotice("");
    try {
      const created = await createTrackerAgent({
        symbol: symbol || undefined,
        name: name || undefined,
        triggers,
        auto_simulate: createForm.autoSimulate,
        create_prompt: prompt || undefined
      });
      const creationNotification = created._creation_notification;
      const symbolResolution = created.symbol_resolution ?? undefined;
      const twilioInfo =
        creationNotification && typeof creationNotification === "object"
          ? ((creationNotification.twilio as Record<string, unknown> | undefined) ?? undefined)
          : undefined;
      const delivered = Boolean(twilioInfo?.delivered);
      const twilioError = typeof twilioInfo?.error === "string" ? twilioInfo.error : "";
      const [freshAgents, refreshed] = await Promise.all([listTrackerAgents(), triggerTrackerPoll().catch(() => null)]);
      setAgents(freshAgents);
      if (refreshed) setSnapshot(refreshed);
      setCreateModalOpen(false);
      if (symbolResolution?.auto_corrected && symbolResolution.resolved_symbol) {
        const input = String(symbolResolution.input_symbol || symbol || "").trim();
        setCreateNotice(
          `Agent created for ${String(symbolResolution.resolved_symbol).toUpperCase()}${input ? ` (auto-corrected from ${input}).` : "."}`,
        );
      }
      if (creationNotification && !delivered) {
        setCreateError(
          `Agent created, but creation SMS failed${twilioError ? `: ${twilioError}` : "."}`,
        );
      }
    } catch (error) {
      const typedError = error as Error & {
        detail?: string;
        symbol_resolution?: TrackerSymbolResolution;
      };
      const symbolResolution = typedError.symbol_resolution;
      const suggestedMatches = Array.isArray(symbolResolution?.suggestions)
        ? symbolResolution.suggestions
            .map((item) => ({
              ticker: String(item.ticker || "").trim().toUpperCase(),
              name: String(item.name || item.ticker || "").trim(),
              exchange: typeof item.exchange === "string" ? item.exchange : undefined,
            }))
            .filter((item) => item.ticker.length > 0)
        : [];
      if (typedError.detail === "symbol_ambiguous" && suggestedMatches.length > 0) {
        setCreateSearchResults(suggestedMatches);
        setCreateSearchError("Multiple ticker matches found. Select one suggestion, then create again.");
        setCreateError("Ticker is ambiguous.");
      } else {
        setCreateError(error instanceof Error ? error.message : "Could not create tracker agent.");
      }
    } finally {
      setCreateLoading(false);
    }
  }

  async function handleDeleteAgent(agentId: string) {
    try {
      await deleteTrackerAgent(agentId);
      const freshAgents = await listTrackerAgents();
      setAgents(freshAgents);
    } catch {
      // no-op
    }
  }

  async function handleManagerInstruction(agentId: string, messageRaw?: string) {
    const message = (messageRaw ?? instructionDraft).trim();
    if (!message) return;
    setInstructionError("");
    setManagerLoadingAgentId(agentId);
    try {
      const result = await interactWithTrackerAgent(agentId, message);
      setManagerResponses((prev) => ({
        ...prev,
        [agentId]: result.reply?.response || "No response received."
      }));
      setLastSavedInstructionByAgent((prev) => ({
        ...prev,
        [agentId]: message
      }));
      setInstructionDraft("");
      setInstructionModalAgentId(null);
      const refreshed = await triggerTrackerPoll().catch(() => null);
      if (refreshed) setSnapshot(refreshed);
    } catch (error) {
      setInstructionError(error instanceof Error ? error.message : "Manager interaction failed.");
      setManagerResponses((prev) => ({
        ...prev,
        [agentId]: error instanceof Error ? error.message : "Manager interaction failed."
      }));
    } finally {
      setManagerLoadingAgentId((prev) => (prev === agentId ? null : prev));
    }
  }

  async function openInstructionModal(agentId: string) {
    setInstructionModalAgentId(agentId);
    setInstructionDraft(lastSavedInstructionByAgent[agentId] ?? "");
    setInstructionError("");
    setInstructionLoadingAgentId(agentId);
    try {
      const history = await getTrackerAgentHistory(agentId, 40);
      const lastInstruction =
        history.find((item) => item.event_type === "manager_instruction" && String(item.raw_prompt || "").trim()) ||
        history.find((item) => item.event_type === "create_prompt" && String(item.raw_prompt || "").trim());
      if (lastInstruction?.raw_prompt) {
        setInstructionDraft(String(lastInstruction.raw_prompt));
      }
    } catch (error) {
      setInstructionError(error instanceof Error ? error.message : "Could not load previous instruction.");
    } finally {
      setInstructionLoadingAgentId((prev) => (prev === agentId ? null : prev));
    }
  }

  function commonAsks(symbol: string): Array<{ label: string; prompt: string }> {
    const ticker = symbol.toUpperCase();
    return [
      { label: "2% Drop Alert", prompt: `Notify me when ${ticker} drops 2% vs previous close.` },
      { label: "Hourly Sentiment", prompt: `Send hourly sentiment summaries for ${ticker} from Perplexity, X, and Reddit.` },
      { label: "Reddit Only", prompt: `Track ${ticker} Reddit sentiment only and alert on bearish reversal.` },
      { label: "Sim On Alert", prompt: `Run simulation on every alert for ${ticker} and include result in the notification.` },
      { label: "Daily Thesis", prompt: `Give me a daily thesis update for ${ticker} at 09:30 America/New_York.` }
    ];
  }

  function formatAgentSettings(agent: TrackerAgent): string {
    const triggers = (agent.triggers ?? {}) as Record<string, unknown>;
    const scheduleMode = String(triggers.schedule_mode ?? "realtime");
    const reportMode = String(triggers.report_mode ?? "hybrid");
    const baselineMode = String(triggers.baseline_mode ?? "prev_close");
    const startAt = String(triggers.start_at ?? "");
    const customTimeEnabled = Boolean(triggers.custom_time_enabled);
    const runTime = String(triggers.daily_run_time ?? "").trim();
    const timezone = String(triggers.timezone ?? "").trim();
    const sourceList = Array.isArray(triggers.research_sources)
      ? (triggers.research_sources as unknown[]).map((item) => String(item)).join(", ")
      : "auto";
    let scheduleLine = "realtime (every 2m)";
    if (scheduleMode === "hourly") scheduleLine = "hourly";
    if (scheduleMode === "daily") scheduleLine = `daily @ ${runTime || "09:30"}${timezone ? ` ${timezone}` : ""}`;
    if (scheduleMode === "custom") {
      scheduleLine = customTimeEnabled
        ? `custom time @ ${runTime || "09:30"}${timezone ? ` ${timezone}` : ""}`
        : "custom interval";
    }
    const startLine = startAt ? ` | Start: ${formatDate(startAt)}` : "";
    return `Schedule: ${scheduleLine}${startLine} | Mode: ${reportMode} | Baseline: ${baselineMode} | Sources: ${sourceList}`;
  }

  async function handleContextQuery(agentId: string) {
    const question = (contextQuestionByAgent[agentId] ?? "").trim();
    if (!question) return;
    setContextLoadingAgentId(agentId);
    try {
      const result = await queryTrackerAgentContext(agentId, question, {
        run_limit: 60,
        history_limit: 60,
        csv_limit: 180
      });
      setContextAnswerByAgent((prev) => ({
        ...prev,
        [agentId]: result.answer?.response || "No context answer returned."
      }));
    } catch (error) {
      setContextAnswerByAgent((prev) => ({
        ...prev,
        [agentId]: error instanceof Error ? error.message : "Context query failed."
      }));
    } finally {
      setContextLoadingAgentId((prev) => (prev === agentId ? null : prev));
    }
  }

  function handleRemoveWatchlistTicker(symbol: string) {
    void onWatchlistChange(watchlist.filter((tickerSymbol) => tickerSymbol !== symbol));
  }

  return (
    <section className="panel stack stagger">
      <WatchlistBar
        watchlist={watchlist}
        activeTicker={activeTicker}
        onSelectTicker={onTickerChange}
        onRemoveTicker={handleRemoveWatchlistTicker}
      />

      <div className="glass-card card-row tracker-actions tracker-watchlist-card">
        <label className="watchlist-add-field">
          Add Company / Ticker
          <div className="ticker-autocomplete">
            <input
              value={watchlistInput}
              placeholder="Type Tesla, Apple, Nvidia, SPY…"
              onFocus={() => setWatchlistInputFocused(true)}
              onBlur={() => setTimeout(() => setWatchlistInputFocused(false), 120)}
              onChange={(event) => setWatchlistInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key !== "Enter") return;
                event.preventDefault();
                const topSuggestion = availableWatchlistSuggestions[0]?.ticker;
                void handleAddToWatchlist(topSuggestion ?? watchlistInput);
              }}
            />
            {watchlistInputFocused && watchlistInput.trim() ? (
              <div className="ticker-suggestions" role="listbox" aria-label="Ticker suggestions">
                {availableWatchlistSuggestions.length > 0 ? (
                  availableWatchlistSuggestions.map((entry) => (
                    <button
                      key={`${entry.ticker}-${entry.name}`}
                      type="button"
                      className="ticker-suggestion"
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => void handleAddToWatchlist(entry.ticker)}
                    >
                      <span className="ticker-suggestion-symbol">{entry.ticker}</span>
                      <span className="ticker-suggestion-name">{entry.name}</span>
                    </button>
                  ))
                ) : watchlistSearchLoading ? (
                  <p className="ticker-suggestion-empty">Searching…</p>
                ) : (
                  <p className="ticker-suggestion-empty">No matches found</p>
                )}
              </div>
            ) : null}
          </div>
        </label>
        <button onClick={() => void handleAddToWatchlist()} disabled={loading || watchlistInput.trim().length === 0}>
          Add To Watchlist
        </button>
        <button className="secondary" onClick={handlePoll} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh Market Data"}
        </button>
      </div>

      <div className="glass-card stack">
        <div className="panel-header">
          <h3>Tracker Agents</h3>
          <span className="muted">Simple cadence setup + natural-language behavior</span>
        </div>
        <button type="button" className="tracker-create-open" onClick={openCreateModal}>
          Add Agent
        </button>
        <p className="muted">Keep setup simple: pick cadence + baseline, then describe behavior in natural language.</p>
        {createNotice ? <p className="muted">{createNotice}</p> : null}
        {createError ? <p className="error">{createError}</p> : null}
      </div>

      <div className="glass-card stack">
        <div className="panel-header">
          <h3>Active Tracker Agents</h3>
          <span className="muted">{agents.length} deployed</span>
        </div>
        {agents.length === 0 ? <p className="muted">No tracker agents deployed yet.</p> : null}
        {agents.map((agent) => (
          <article key={agent.id} className="event-item tracker-agent-card">
            <div className="event-meta tracker-agent-meta">
              <strong>
                {agent.name} | {agent.symbol}
              </strong>
              <span>{agent.status}</span>
            </div>
            <p className="muted">
              Alerts: {agent.total_alerts ?? 0}
              {agent.last_alert_at ? ` | Last alert ${new Date(agent.last_alert_at).toLocaleString()}` : ""}
            </p>
            <p className="muted tracker-agent-settings">{formatAgentSettings(agent)}</p>
            <div className="tracker-agent-actions">
              <button type="button" className="secondary" onClick={() => handleDeleteAgent(agent.id)}>
                Delete Agent
              </button>
              <button type="button" onClick={() => void openInstructionModal(agent.id)}>
                Edit Instruction
              </button>
            </div>
            <div className="source-citations">
              <span className="muted">Common asks:</span>
              {commonAsks(agent.symbol).map((suggestion, idx) => (
                <button
                  key={`${agent.id}-suggestion-${idx}`}
                  type="button"
                  className="secondary"
                  onClick={() => void handleManagerInstruction(agent.id, suggestion.prompt)}
                  disabled={managerLoadingAgentId === agent.id}
                  title={suggestion.prompt}
                >
                  {suggestion.label}
                </button>
              ))}
            </div>
            <div className="tracker-context-block">
              <textarea
                className="tracker-context-input"
                rows={5}
                value={contextQuestionByAgent[agent.id] ?? ""}
                onChange={(event) =>
                  setContextQuestionByAgent((prev) => ({
                    ...prev,
                    [agent.id]: event.target.value
                  }))
                }
                placeholder={`Ask ${agent.name} from saved history/CSV context (e.g., "What changed in sentiment over last 10 runs?").`}
              />
              <button
                type="button"
                onClick={() => void handleContextQuery(agent.id)}
                disabled={contextLoadingAgentId === agent.id || !(contextQuestionByAgent[agent.id] ?? "").trim()}
              >
                {contextLoadingAgentId === agent.id ? "Querying..." : "Ask Saved Context"}
              </button>
              {managerResponses[agent.id] ? <article className="tracker-context-answer">{managerResponses[agent.id]}</article> : null}
              {contextAnswerByAgent[agent.id] ? <article className="tracker-context-answer">{contextAnswerByAgent[agent.id]}</article> : null}
            </div>
          </article>
        ))}
      </div>

      <div className="glass-card">
        <div className="panel-header">
          <h3>Market Grid</h3>
          <span className="muted">{snapshot?.generated_at ? new Date(snapshot.generated_at).toLocaleTimeString() : "No data"}</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Price</th>
                <th>Change</th>
                <th>P/E</th>
                <th>Beta</th>
                <th>Volume</th>
                <th>Mkt Cap</th>
                <th>Signal</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((item) => {
                const signal = item.change_percent > 1.5 ? "Momentum" : item.change_percent < -1.5 ? "Risk" : "Neutral";
                return (
                  <tr
                    key={item.ticker}
                    className={item.ticker === activeTicker ? "selected-row" : ""}
                    onClick={() => onTickerChange(item.ticker)}
                    role="button"
                  >
                    <td>{item.ticker}</td>
                    <td>{formatCurrency(item.price)}</td>
                    <td className={item.change_percent >= 0 ? "text-green" : "text-red"}>{formatPercent(item.change_percent)}</td>
                    <td>{item.pe_ratio?.toFixed(2) ?? "-"}</td>
                    <td>{item.beta?.toFixed(2) ?? "-"}</td>
                    <td>{formatCompactNumber(item.volume)}</td>
                    <td>{formatCompactNumber(item.market_cap)}</td>
                    <td>{signal}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="glass-card">
        <h3>Triggered Alerts</h3>
        <div className="stack small-gap">
          {(snapshot?.alerts_triggered ?? []).slice(0, 8).map((alert, idx) => (
            <article key={`alert-${idx}`} className="alert-item">
              <strong>{String(alert.ticker ?? "Market Event")}</strong>
              <p>{String(alert.analysis ?? alert.reason ?? "No details")}</p>
            </article>
          ))}
          {(snapshot?.alerts_triggered ?? []).length === 0 ? <p className="muted">No triggers in the latest poll.</p> : null}
        </div>
      </div>

      {createModalOpen ? (
        <div className="auth-modal-backdrop" onClick={() => (!createLoading ? setCreateModalOpen(false) : null)}>
          <div className="auth-modal tracker-create-modal" onClick={(event) => event.stopPropagation()}>
            <h3>Create Tracker Agent</h3>
            <p className="muted">
              Keep this simple: choose cadence and baseline, then describe behavior in plain English.
            </p>

            <div className="tracker-create-grid">
              <label>
                Ticker or Company
                <div className="ticker-autocomplete">
                  <input
                    value={createForm.symbol}
                    onFocus={() => setCreateSymbolFocused(true)}
                    onBlur={() => setTimeout(() => setCreateSymbolFocused(false), 120)}
                    onChange={(event) => {
                      updateCreateForm("symbol", normalizeSymbol(event.target.value));
                      setCreateSearchError("");
                    }}
                    onKeyDown={(event) => {
                      if (event.key !== "Enter") return;
                      event.preventDefault();
                      const top = createSearchResults[0];
                      if (top) applyCreateTickerResult(top);
                    }}
                    placeholder="NVDA or Nvidia"
                  />
                  {createSymbolFocused && createForm.symbol.trim() ? (
                    <div className="ticker-suggestions" role="listbox" aria-label="Ticker suggestions">
                      {createSearchResults.length > 0 ? (
                        createSearchResults.slice(0, 6).map((result) => (
                          <button
                            key={`${result.ticker}-${result.name}`}
                            type="button"
                            className="ticker-suggestion"
                            onMouseDown={(event) => event.preventDefault()}
                            onClick={() => applyCreateTickerResult(result)}
                          >
                            <span className="ticker-suggestion-symbol">{result.ticker}</span>
                            <span className="ticker-suggestion-name">{result.name}</span>
                          </button>
                        ))
                      ) : createSearchLoading ? (
                        <p className="ticker-suggestion-empty">Searching…</p>
                      ) : (
                        <p className="ticker-suggestion-empty">No matches found</p>
                      )}
                    </div>
                  ) : null}
                </div>
                {createSearchError ? <span className="muted">{createSearchError}</span> : null}
              </label>

              <label>
                Agent Name
                <input
                  value={createForm.name}
                  onChange={(event) => updateCreateForm("name", event.target.value)}
                  placeholder="Nvidia Tracker Agent"
                />
              </label>

              <label>
                Schedule
                <select
                  value={createForm.scheduleMode}
                  onChange={(event) => updateCreateForm("scheduleMode", event.target.value as ScheduleMode)}
                >
                  <option value="realtime">Livestream (every 2 min)</option>
                  <option value="hourly">Hourly</option>
                  <option value="daily">Daily</option>
                  <option value="custom">Custom Time</option>
                </select>
              </label>

              <label>
                Start Timer (Required)
                <input
                  type="datetime-local"
                  value={createForm.startAtLocal}
                  onChange={(event) => updateCreateForm("startAtLocal", event.target.value)}
                />
              </label>

              <label>
                Report Mode
                <select value={createForm.reportMode} onChange={(event) => updateCreateForm("reportMode", event.target.value as ReportMode)}>
                  <option value="triggers_only">Trigger-Only Alerts</option>
                  <option value="periodic">Periodic Reports</option>
                  <option value="hybrid">Both Alerts + Reports</option>
                </select>
              </label>

              {createForm.scheduleMode === "daily" || createForm.scheduleMode === "custom" ? (
                <label>
                  Run Time
                  <input
                    type="time"
                    value={createForm.runTimeLocal}
                    onChange={(event) => updateCreateForm("runTimeLocal", event.target.value)}
                  />
                </label>
              ) : null}

              <label>
                Baseline
                <select
                  value={createForm.baselineMode}
                  onChange={(event) => updateCreateForm("baselineMode", event.target.value as BaselineMode)}
                >
                  <option value="prev_close">Previous Close</option>
                  <option value="session_open">Session Open</option>
                  <option value="last_check">Last Check</option>
                  <option value="last_alert">Last Alert</option>
                </select>
              </label>

              <label className="tracker-check tracker-check-inline">
                Auto-Simulate On Alert
                <input
                  type="checkbox"
                  checked={createForm.autoSimulate}
                  onChange={(event) => updateCreateForm("autoSimulate", event.target.checked)}
                />
              </label>

              <label className="full-span">
                Initial Manager Instruction (Optional)
                <textarea
                  rows={6}
                  value={createForm.managerPrompt}
                  onChange={(event) => updateCreateForm("managerPrompt", event.target.value)}
                  placeholder='Example: "Track sentiment divergence and send periodic reports even without price triggers."'
                />
              </label>
            </div>

            <div className="tracker-modal-actions">
              <button onClick={handleCreateAgentWithSettings} disabled={createLoading}>
                {createLoading ? "Creating..." : "Create Agent"}
              </button>
              <button className="secondary" onClick={() => setCreateModalOpen(false)} disabled={createLoading}>
                Cancel
              </button>
            </div>
            {createError ? <p className="auth-error">{createError}</p> : null}
          </div>
        </div>
      ) : null}

      {instructionModalAgentId && selectedInstructionAgent ? (
        <div className="auth-modal-backdrop" onClick={() => setInstructionModalAgentId(null)}>
          <div className="auth-modal tracker-create-modal" onClick={(event) => event.stopPropagation()}>
            <h3>Edit Instruction</h3>
            <p className="muted">
              Agent: {selectedInstructionAgent.name} ({selectedInstructionAgent.symbol})
            </p>
            <textarea
              rows={7}
              value={instructionDraft}
              onChange={(event) => setInstructionDraft(event.target.value)}
              placeholder={`Message ${selectedInstructionAgent.name} (e.g., "switch to periodic reports every 2 minutes and include sentiment summary").`}
            />
            {instructionLoadingAgentId === selectedInstructionAgent.id ? <p className="muted">Loading previous instruction...</p> : null}
            <button
              type="button"
              onClick={() => void handleManagerInstruction(selectedInstructionAgent.id, instructionDraft)}
              disabled={
                managerLoadingAgentId === selectedInstructionAgent.id ||
                instructionLoadingAgentId === selectedInstructionAgent.id ||
                !instructionDraft.trim()
              }
            >
              {managerLoadingAgentId === selectedInstructionAgent.id ? "Sending..." : "Send Instruction"}
            </button>
            <button
              className="secondary"
              type="button"
              onClick={() => setInstructionModalAgentId(null)}
              disabled={managerLoadingAgentId === selectedInstructionAgent.id}
            >
              Cancel
            </button>
            {instructionError ? <p className="auth-error">{instructionError}</p> : null}
          </div>
        </div>
      ) : null}
    </section>
  );
}
