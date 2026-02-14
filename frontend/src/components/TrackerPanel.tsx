import { useEffect, useMemo, useState } from "react";
import {
  addAlert,
  createTrackerAgentByPrompt,
  createTrackerAgent,
  deleteTrackerAgent,
  fetchCandles,
  getTrackerAgentDetail,
  interactWithTrackerAgent,
  getTrackerSnapshot,
  listTrackerAgents,
  setWatchlist,
  triggerTrackerPoll
} from "../lib/api";
import { formatCompactNumber, formatCurrency, formatPercent } from "../lib/format";
import type { CandlePoint, TrackerAgent, TrackerAgentDetail, TrackerSnapshot, WSMessage } from "../lib/types";
import StockChart from "./StockChart";

interface Props {
  activeTicker: string;
  onTickerChange: (ticker: string) => void;
  trackerEvent?: WSMessage;
  focusAgent?: { agentId: string; requestedAt: number } | null;
}

export default function TrackerPanel({ activeTicker, onTickerChange, trackerEvent, focusAgent }: Props) {
  const [watchlistInput, setWatchlistInput] = useState("AAPL,MSFT,NVDA,TSLA,SPY");
  const [newSymbol, setNewSymbol] = useState("");
  const [snapshot, setSnapshot] = useState<TrackerSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [nlPrompt, setNlPrompt] = useState("");
  const [alertTicker, setAlertTicker] = useState(activeTicker);
  const [threshold, setThreshold] = useState(2);
  const [direction, setDirection] = useState<"up" | "down" | "either">("either");
  const [agentName, setAgentName] = useState(`${activeTicker} Watch`);
  const [agents, setAgents] = useState<TrackerAgent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<TrackerAgentDetail | null>(null);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentMessage, setAgentMessage] = useState("");
  const [agentChat, setAgentChat] = useState<Array<{ role: "manager" | "agent"; text: string; at: string }>>([]);
  const [agentChart, setAgentChart] = useState<{ period: string; interval: string; points: CandlePoint[] } | null>(null);
  const [agentResearch, setAgentResearch] = useState<Record<string, unknown> | null>(null);
  const [agentSimulation, setAgentSimulation] = useState<{ session_id: string; ticker: string } | null>(null);

  useEffect(() => {
    getTrackerSnapshot().then(setSnapshot).catch(() => null);
    listTrackerAgents().then(setAgents).catch(() => setAgents([]));
  }, []);

  useEffect(() => {
    if (!trackerEvent || trackerEvent.type !== "tracker_snapshot") return;
    setSnapshot({
      generated_at: String(trackerEvent.generated_at ?? new Date().toISOString()),
      tickers: Array.isArray(trackerEvent.tickers) ? (trackerEvent.tickers as TrackerSnapshot["tickers"]) : [],
      alerts_triggered: Array.isArray(trackerEvent.alerts) ? (trackerEvent.alerts as TrackerSnapshot["alerts_triggered"]) : []
    });
  }, [trackerEvent]);

  useEffect(() => {
    if (!focusAgent?.agentId) return;
    setAgentLoading(true);
    getTrackerAgentDetail(focusAgent.agentId)
      .then((detail) => {
        setSelectedAgent(detail);
        onTickerChange(detail.agent.symbol);
      })
      .catch(() => null)
      .finally(() => setAgentLoading(false));
  }, [focusAgent?.agentId, focusAgent?.requestedAt, onTickerChange]);

  useEffect(() => {
    if (!selectedAgent) return;
    const timer = window.setInterval(() => {
      getTrackerAgentDetail(selectedAgent.agent.id)
        .then((detail) => setSelectedAgent(detail))
        .catch(() => null);
    }, 7000);
    return () => window.clearInterval(timer);
  }, [selectedAgent?.agent.id]);

  const sorted = useMemo(
    () => [...(snapshot?.tickers ?? [])].sort((a, b) => Math.abs(b.change_percent) - Math.abs(a.change_percent)),
    [snapshot]
  );

  async function handleUpdateWatchlist() {
    setLoading(true);
    try {
      const values = watchlistInput
        .split(",")
        .map((token) => token.trim().toUpperCase())
        .filter(Boolean);
      const updated = await setWatchlist(values);
      setWatchlistInput(updated.join(","));
      const refreshed = await triggerTrackerPoll();
      setSnapshot(refreshed);
    } finally {
      setLoading(false);
    }
  }

  async function handleAddSymbol() {
    const symbol = newSymbol.trim().toUpperCase();
    if (!symbol) return;
    const existing = (snapshot?.tickers ?? []).map((item) => item.ticker.toUpperCase());
    if (existing.includes(symbol)) {
      setNewSymbol("");
      return;
    }
    setLoading(true);
    try {
      const next = [...existing, symbol];
      const updated = await setWatchlist(next);
      setWatchlistInput(updated.join(","));
      const refreshed = await triggerTrackerPoll();
      setSnapshot(refreshed);
      setNewSymbol("");
    } finally {
      setLoading(false);
    }
  }

  async function handleRemoveSymbol(symbol: string) {
    const existing = (snapshot?.tickers ?? []).map((item) => item.ticker.toUpperCase());
    const next = existing.filter((ticker) => ticker !== symbol.toUpperCase());
    setLoading(true);
    try {
      const updated = await setWatchlist(next);
      setWatchlistInput(updated.join(","));
      const refreshed = await triggerTrackerPoll();
      setSnapshot(refreshed);
      if (activeTicker.toUpperCase() === symbol.toUpperCase() && refreshed.tickers[0]?.ticker) {
        onTickerChange(refreshed.tickers[0].ticker);
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleAddAlert() {
    await addAlert({
      ticker: alertTicker.trim().toUpperCase(),
      threshold_percent: threshold,
      direction
    });
  }

  async function handleDeployAgent() {
    const symbol = alertTicker.trim().toUpperCase();
    if (!symbol) return;
    setLoading(true);
    try {
      await createTrackerAgent({
        symbol,
        name: agentName.trim() || `${symbol} Watch`,
        triggers: {
          price_change_pct: threshold,
          direction
        },
        auto_simulate: true
      });
      const refreshed = await listTrackerAgents();
      setAgents(refreshed);
    } finally {
      setLoading(false);
    }
  }

  async function handleOpenAgent(agent: TrackerAgent) {
    setAgentLoading(true);
    try {
      const detail = await getTrackerAgentDetail(agent.id);
      setSelectedAgent(detail);
      setAgentChat([]);
      setAgentMessage("");
      setAgentResearch(null);
      setAgentSimulation(null);
      const points = await fetchCandles(agent.symbol, "6mo", "1d");
      setAgentChart({ period: "6mo", interval: "1d", points });
      onTickerChange(agent.symbol);
    } finally {
      setAgentLoading(false);
    }
  }

  async function handleDeleteAgent(agent: TrackerAgent) {
    setLoading(true);
    try {
      await deleteTrackerAgent(agent.id);
      const refreshed = await listTrackerAgents();
      setAgents(refreshed);
      if (selectedAgent?.agent.id === agent.id) setSelectedAgent(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleDeployByPrompt() {
    if (!nlPrompt.trim()) return;
    setLoading(true);
    try {
      await createTrackerAgentByPrompt(nlPrompt.trim());
      const refreshed = await listTrackerAgents();
      setAgents(refreshed);
      setNlPrompt("");
    } finally {
      setLoading(false);
    }
  }

  async function handleAskAgent() {
    if (!selectedAgent || !agentMessage.trim()) return;
    const managerMessage = agentMessage.trim();
    setAgentLoading(true);
    try {
      setAgentChat((prev) => [...prev, { role: "manager", text: managerMessage, at: new Date().toISOString() }]);
      const response = await interactWithTrackerAgent(selectedAgent.agent.id, managerMessage);
      setAgentChat((prev) => [...prev, { role: "agent", text: response.reply?.response ?? "", at: new Date().toISOString() }]);
      if (response.tool_outputs?.chart) setAgentChart(response.tool_outputs.chart);
      if (response.tool_outputs?.research) setAgentResearch(response.tool_outputs.research);
      if (response.tool_outputs?.simulation) setAgentSimulation(response.tool_outputs.simulation);
      setAgentMessage("");
      const detail = await getTrackerAgentDetail(selectedAgent.agent.id);
      setSelectedAgent(detail);
    } finally {
      setAgentLoading(false);
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

  return (
    <section className="panel stack stagger">
      <header className="panel-header">
        <h2>Tracker</h2>
        <p>Yahoo-style ticker monitor with valuation metrics, alerting, and live anomaly detection pipeline.</p>
      </header>

      <div className="glass-card card-row tracker-actions">
        <label>
          Watchlist
          <input value={watchlistInput} onChange={(event) => setWatchlistInput(event.target.value.toUpperCase())} />
        </label>
        <label>
          Add Symbol
          <input value={newSymbol} onChange={(event) => setNewSymbol(event.target.value.toUpperCase())} maxLength={12} />
        </label>
        <button className="secondary" onClick={handleAddSymbol} disabled={loading || !newSymbol.trim()}>
          Add Stock
        </button>
        <button onClick={handleUpdateWatchlist} disabled={loading}>
          Update List
        </button>
        <button className="secondary" onClick={handlePoll} disabled={loading}>
          {loading ? "Polling…" : "Poll Now"}
        </button>
      </div>

      <div className="glass-card card-row tracker-actions">
        <label style={{ minWidth: 340, flex: 1 }}>
          Manager Instruction (Natural Language)
          <input
            value={nlPrompt}
            onChange={(event) => setNlPrompt(event.target.value)}
            placeholder="e.g. Track NVDA, alert if social sentiment turns very bearish or price drops 3%"
          />
        </label>
        <button className="secondary" onClick={handleDeployByPrompt} disabled={loading || !nlPrompt.trim()}>
          {loading ? "Parsing…" : "Deploy from Prompt"}
        </button>
      </div>

      <div className="glass-card card-row tracker-actions">
        <label>
          Alert Ticker
          <input value={alertTicker} onChange={(event) => setAlertTicker(event.target.value.toUpperCase())} />
        </label>
        <label>
          Agent Name
          <input value={agentName} onChange={(event) => setAgentName(event.target.value)} />
        </label>
        <label>
          Threshold %
          <input
            type="number"
            min={0.1}
            max={25}
            step={0.1}
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value) || 2)}
          />
        </label>
        <label>
          Direction
          <select value={direction} onChange={(event) => setDirection(event.target.value as "up" | "down" | "either")}>
            <option value="either">Either</option>
            <option value="up">Up</option>
            <option value="down">Down</option>
          </select>
        </label>
        <button onClick={handleAddAlert}>Add Alert</button>
        <button className="secondary" onClick={handleDeployAgent} disabled={loading}>
          {loading ? "Deploying…" : "Deploy Agent"}
        </button>
      </div>

      <div className="glass-card">
        <div className="panel-header">
          <h3>Tracker Agents</h3>
          <span className="muted">{agents.length} active</span>
        </div>
        <div className="card-row wrap">
          {agents.map((agent) => (
            <article key={agent.id} className="glass-card" style={{ minWidth: 240, padding: "0.75rem" }}>
              <div className="source-title">
                <strong>{agent.name}</strong>
                <span className={`pill ${agent.status === "active" ? "bullish" : "neutral"}`}>{agent.status}</span>
              </div>
              <p className="muted">{agent.symbol}</p>
              <p className="muted">Triggers: {JSON.stringify(agent.triggers)}</p>
              <div className="inline-links">
                <button onClick={() => void handleOpenAgent(agent)}>{agentLoading ? "Opening…" : "View Agent"}</button>
                <button className="secondary" onClick={() => void handleDeleteAgent(agent)}>Delete</button>
              </div>
            </article>
          ))}
          {agents.length === 0 ? <p className="muted">No deployed agents yet.</p> : null}
        </div>
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
                <th>Watchlist</th>
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
                    <td>
                      <button
                        type="button"
                        className="secondary"
                        onClick={(event) => {
                          event.stopPropagation();
                          void handleRemoveSymbol(item.ticker);
                        }}
                      >
                        Remove
                      </button>
                    </td>
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

      {selectedAgent ? (
        <div
          role="dialog"
          aria-modal="true"
          className="modal-overlay"
          onClick={() => setSelectedAgent(null)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(2, 6, 23, 0.72)",
            display: "grid",
            placeItems: "center",
            zIndex: 50
          }}
        >
          <div
            className="glass-card"
            onClick={(e) => e.stopPropagation()}
            style={{ width: "min(920px, 94vw)", maxHeight: "84vh", overflow: "auto", padding: "1rem" }}
          >
            <div className="panel-header">
              <h3>{selectedAgent.agent.name} ({selectedAgent.agent.symbol})</h3>
              <button className="secondary" onClick={() => setSelectedAgent(null)}>Close</button>
            </div>
            <div className="kpi-grid">
              <div>
                <p className="muted">Price</p>
                <h3>{selectedAgent.market ? formatCurrency(selectedAgent.market.price) : "-"}</h3>
              </div>
              <div>
                <p className="muted">Change</p>
                <h3 className={selectedAgent.market && selectedAgent.market.change_percent >= 0 ? "text-green" : "text-red"}>
                  {selectedAgent.market ? formatPercent(selectedAgent.market.change_percent) : "-"}
                </h3>
              </div>
              <div>
                <p className="muted">Volume</p>
                <h3>{selectedAgent.market ? formatCompactNumber(selectedAgent.market.volume) : "-"}</h3>
              </div>
            </div>

            <div className="card-row card-row-split" style={{ marginTop: "0.75rem" }}>
              <div className="glass-card">
                <h4>Recent Agent Actions</h4>
                {(selectedAgent.recent_actions ?? []).slice(0, 20).map((action, idx) => (
                  <article key={`action-${idx}`} className="alert-item">
                    <strong>{String(action.agent_name ?? "Tracker")}</strong>
                    <p>{String(action.action ?? "-")}</p>
                    <p className="muted">{String(action.created_at ?? "")}</p>
                  </article>
                ))}
                {(selectedAgent.recent_actions ?? []).length === 0 ? <p className="muted">No actions yet.</p> : null}
              </div>

              <div className="glass-card">
                <h4>Recent Alerts</h4>
                {(selectedAgent.recent_alerts ?? []).slice(0, 20).map((alert, idx) => (
                  <article key={`agent-alert-${idx}`} className="alert-item">
                    <strong>{String(alert.symbol ?? selectedAgent.agent.symbol)}</strong>
                    <p>{String(alert.trigger_reason ?? alert.reason ?? "-")}</p>
                    <p className="muted">{String(alert.created_at ?? "")}</p>
                  </article>
                ))}
                {(selectedAgent.recent_alerts ?? []).length === 0 ? <p className="muted">No alerts yet.</p> : null}
              </div>
            </div>
            <div className="glass-card" style={{ marginTop: "0.75rem" }}>
              <div className="panel-header">
                <h4>Agent Market Workspace</h4>
                <span className="muted">
                  {agentChart ? `${agentChart.period} / ${agentChart.interval}` : "Loading chart"}
                </span>
              </div>
              {agentChart ? <StockChart points={agentChart.points} mode="candles" showSma showEma /> : <p className="muted">No chart yet.</p>}
            </div>
            <div className="glass-card" style={{ marginTop: "0.75rem" }}>
              <h4>Ask This Agent</h4>
              <div className="card-row">
                <label style={{ minWidth: 320, flex: 1 }}>
                  Message
                  <textarea
                    value={agentMessage}
                    onChange={(event) => setAgentMessage(event.target.value)}
                    rows={3}
                    placeholder="What are you seeing in NVDA sentiment right now, and what changed since last poll?"
                  />
                </label>
                <button onClick={() => void handleAskAgent()} disabled={agentLoading || !agentMessage.trim()}>
                  {agentLoading ? "Thinking…" : "Ask Agent"}
                </button>
              </div>
              <div className="stack small-gap" style={{ marginTop: "0.5rem", maxHeight: 240, overflowY: "auto" }}>
                {agentChat.map((entry, idx) => (
                  <article key={`agent-chat-${idx}`} className="alert-item">
                    <strong>{entry.role === "manager" ? "Manager" : "Agent"}</strong>
                    <p>{entry.text}</p>
                    <p className="muted">{new Date(entry.at).toLocaleTimeString()}</p>
                  </article>
                ))}
                {agentChat.length === 0 ? <p className="muted">Start a conversation. Ask for chart, research, or simulation.</p> : null}
              </div>
            </div>
            {agentResearch ? (
              <div className="glass-card" style={{ marginTop: "0.75rem" }}>
                <h4>Tool Output: Research Snapshot</h4>
                <pre>{JSON.stringify(agentResearch, null, 2)}</pre>
              </div>
            ) : null}
            {agentSimulation ? (
              <div className="glass-card" style={{ marginTop: "0.75rem" }}>
                <h4>Tool Output: Simulation</h4>
                <p>Session: <code>{agentSimulation.session_id}</code></p>
                <p>Ticker: {agentSimulation.ticker}</p>
              </div>
            ) : null}
            <div className="glass-card" style={{ marginTop: "0.75rem" }}>
              <h4>Autonomous Activity Log</h4>
              <div className="stack small-gap" style={{ maxHeight: 220, overflowY: "auto" }}>
                {(selectedAgent.recent_actions ?? []).slice(0, 20).map((action, idx) => (
                  <article key={`auto-log-${idx}`} className="alert-item">
                    <strong>{String(action.agent_name ?? "Agent")}</strong>
                    <p>{String(action.action ?? "-")}</p>
                    <p className="muted">{String(action.created_at ?? "")}</p>
                  </article>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
