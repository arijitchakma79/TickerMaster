import { useEffect, useMemo, useState } from "react";
import {
  fetchAdvancedStockData,
  fetchCandles,
  getAgentActivity,
  runDeepResearch,
  runResearch,
  searchTickerDirectory
} from "../lib/api";
import { formatCompactNumber, formatPercent } from "../lib/format";
import { resolveTickerCandidate } from "../lib/tickerInput";
import type { AgentActivity, AdvancedStockData, CandlePoint, DeepResearchResponse, ResearchResponse, TickerLookup, WSMessage } from "../lib/types";
import StockChart from "./StockChart";

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

export default function ResearchPanel({ activeTicker, onTickerChange, connected, events }: Props) {
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
  const [chartMode, setChartMode] = useState<"candles" | "line">("candles");
  const [chartPeriod, setChartPeriod] = useState("6mo");
  const [chartInterval, setChartInterval] = useState("1d");
  const [showSma, setShowSma] = useState(true);
  const [showEma, setShowEma] = useState(true);
  const [agentHistory, setAgentHistory] = useState<AgentActivity[]>([]);
  const [error, setError] = useState("");
  const normalizedTicker = useMemo(() => normalizeSymbol(activeTicker), [activeTicker]);

  const sentimentLabel = useMemo(() => {
    if (!research) return "-";
    return `${formatPercent(research.aggregate_sentiment * 100)} / ${research.recommendation.replace("_", " ")}`;
  }, [research]);

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
    try {
      const [analysis, chart, advancedData] = await Promise.all([
        runResearch(ticker, timeframe),
        fetchCandles(ticker, chartPeriod, chartInterval),
        fetchAdvancedStockData(ticker),
      ]);
      setResearch(analysis);
      setCandles(chart);
      setAdvanced(advancedData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Research request failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleDeepResearch() {
    setDeepLoading(true);
    setError("");
    const ticker = tickerInput.trim().toUpperCase();
    try {
      const result = await runDeepResearch(ticker);
      setDeepResearch(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Deep research request failed");
    } finally {
      setDeepLoading(false);
    }
  }

  return (
    <section className="panel stack stagger">
      <header className="panel-header">
        <h2>Research</h2>
        <p>Perplexity Sonar + X + Reddit + prediction markets, summarized into high-signal narratives.</p>
      </header>

      <div className="card-row research-actions research-actions-card">
        <label className="research-ticker-field">
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
        <label>
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
        <label>
          Chart Type
          <select value={chartMode} onChange={(event) => setChartMode(event.target.value as "candles" | "line")}>
            <option value="candles">Candlesticks</option>
            <option value="line">Line</option>
          </select>
        </label>
        <label>
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
        <label>
          Interval
          <select value={chartInterval} onChange={(event) => setChartInterval(event.target.value)}>
            <option value="1d">1d</option>
            <option value="1wk">1wk</option>
            <option value="1mo">1mo</option>
          </select>
        </label>
        <label>
          <span>Indicators</span>
          <div className="inline-links">
            <button type="button" className={showSma ? "" : "secondary"} onClick={() => setShowSma((v) => !v)}>SMA20</button>
            <button type="button" className={showEma ? "" : "secondary"} onClick={() => setShowEma((v) => !v)}>EMA21</button>
          </div>
        </label>
        <button onClick={() => void handleAnalyze()} disabled={loading}>
          {loading ? "Analyzing…" : "Run Research"}
        </button>
        <button className="secondary" onClick={handleDeepResearch} disabled={deepLoading}>
          {deepLoading ? "Deep researching…" : "Deep Research"}
        </button>
      </div>

      {error ? <p className="error">{error}</p> : null}

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
        <StockChart points={candles} mode={chartMode} showSma={showSma} showEma={showEma} />
      </div>

      {advanced ? (
        <div className="glass-card">
          <div className="panel-header">
            <h3>{advanced.company_name ?? activeTicker} Snapshot</h3>
            <span className="muted">{advanced.exchange ?? "-"}</span>
          </div>
          <div className="kpi-grid">
            <div><p className="muted">Market Cap</p><h3>{formatCompactNumber(advanced.market_cap)}</h3></div>
            <div><p className="muted">Trailing P/E</p><h3>{advanced.trailing_pe ?? "-"}</h3></div>
            <div><p className="muted">Forward P/E</p><h3>{advanced.forward_pe ?? "-"}</h3></div>
            <div><p className="muted">EPS (TTM)</p><h3>{advanced.eps_trailing ?? "-"}</h3></div>
            <div><p className="muted">Target Price</p><h3>{advanced.target_mean_price ?? "-"}</h3></div>
            <div><p className="muted">Recommendation</p><h3>{advanced.recommendation ?? "-"}</h3></div>
          </div>
          <p className="muted" style={{ marginTop: "0.75rem" }}>
            {advanced.sector ?? "-"} / {advanced.industry ?? "-"}
          </p>
          {advanced.description ? <p>{advanced.description}</p> : null}
        </div>
      ) : null}

      <div className="card-row card-row-split">
        <div className="glass-card">
          <h3>Source Sentiment</h3>
          <div className="stack small-gap">
            {research?.source_breakdown.map((entry) => (
              <article key={entry.source} className="source-item">
                <div className="source-title">
                  <strong>{entry.source}</strong>
                  <span className={`pill ${entry.sentiment}`}>{entry.sentiment}</span>
                </div>
                <p>{entry.summary}</p>
                <div className="inline-links">
                  {entry.links.map((link) => (
                    <a key={`${entry.source}-${link.url}`} href={link.url} target="_blank" rel="noreferrer">
                      {link.title}
                    </a>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </div>

        <div className="glass-card">
          <h3>Prediction Market Read</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Market</th>
                  <th>Signal</th>
                </tr>
              </thead>
              <tbody>
                {(research?.prediction_markets ?? []).slice(0, 8).map((market, idx) => (
                  <tr key={`market-${idx}`}>
                    <td>{String(market.source ?? "-")}</td>
                    <td>{String(market.market ?? "-")}</td>
                    <td>{String(market.probability ?? market.yes_price ?? "-")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
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
          <h3>Deep Research Results</h3>
          <p><strong>Analyst Ratings:</strong> {deepResearch.analyst_ratings ?? "-"}</p>
          <p><strong>Insider Trading:</strong> {deepResearch.insider_trading ?? "-"}</p>
          <p><strong>Reddit DD:</strong> {deepResearch.reddit_dd_summary ?? "-"}</p>
          {deepResearch.notes ? <p className="muted">{deepResearch.notes}</p> : null}
        </div>
      ) : null}

      <div className="glass-card">
        <div className="panel-header">
          <h3>Insider Trading</h3>
          <span className="muted">{advanced?.insider_transactions?.length ?? 0} recent filings</span>
        </div>
        <div className="table-wrap">
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
              {(advanced?.insider_transactions ?? []).slice(0, 20).map((row, idx) => (
                <tr key={`insider-${idx}`}>
                  <td>{row.start_date ?? "-"}</td>
                  <td>{row.filer_name ?? "-"}</td>
                  <td>{row.filer_relation ?? "-"}</td>
                  <td>{row.money_text ?? "-"}</td>
                  <td>{formatCompactNumber(row.shares)}</td>
                  <td>{formatCompactNumber(row.value)}</td>
                </tr>
              ))}
              {(advanced?.insider_transactions ?? []).length === 0 ? (
                <tr>
                  <td colSpan={6} className="muted">No insider filings returned for this symbol.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
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
