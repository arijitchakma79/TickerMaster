import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { requestCommentary, spinModalSandbox, startSimulation, stopSimulation } from "../lib/api";
import { formatCurrency } from "../lib/format";
import type { AgentConfig, SimulationState, WSMessage } from "../lib/types";

interface Props {
  activeTicker: string;
  onTickerChange: (ticker: string) => void;
  simulationEvent?: WSMessage;
  simulationLifecycleEvent?: WSMessage;
}

const defaultAgents: AgentConfig[] = [
  {
    name: "Quant Pulse",
    personality: "quant_momentum",
    model: "meta-llama/llama-3.1-8b-instruct",
    aggressiveness: 0.72,
    risk_limit: 0.65,
    trade_size: 30,
    active: true
  },
  {
    name: "Value Anchor",
    personality: "fundamental_value",
    model: "meta-llama/llama-3.1-8b-instruct",
    aggressiveness: 0.42,
    risk_limit: 0.45,
    trade_size: 20,
    active: true
  },
  {
    name: "Retail Wave",
    personality: "retail_reactive",
    model: "meta-llama/llama-3.1-8b-instruct",
    aggressiveness: 0.58,
    risk_limit: 0.55,
    trade_size: 15,
    active: true
  }
];

export default function SimulationPanel({
  activeTicker,
  onTickerChange,
  simulationEvent,
  simulationLifecycleEvent
}: Props) {
  const [ticker, setTicker] = useState(activeTicker);
  const [duration, setDuration] = useState(180);
  const [initialPrice, setInitialPrice] = useState(185);
  const [startingCash, setStartingCash] = useState(100000);
  const [volatility, setVolatility] = useState(0.02);
  const [agents, setAgents] = useState<AgentConfig[]>(defaultAgents);
  const [session, setSession] = useState<SimulationState | null>(null);
  const [priceSeries, setPriceSeries] = useState<Array<{ tick: number; price: number }>>([]);
  const [loading, setLoading] = useState(false);
  const [sandboxPrompt, setSandboxPrompt] = useState("Launch a momentum-vs-value duel with higher volatility and strict risk limits.");
  const [sandboxResult, setSandboxResult] = useState<string>("");
  const [autoCommentary, setAutoCommentary] = useState<string>("");
  const [autoCommentaryModel, setAutoCommentaryModel] = useState<string>("");
  const [autoCommentaryLoading, setAutoCommentaryLoading] = useState(false);

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
    const finalState = { ...session, running: false };
    setSession(finalState);
    void generateAutoCommentary(reason, finalState);
  }, [simulationLifecycleEvent, session]);

  async function handleStart() {
    setLoading(true);
    const cleanTicker = ticker.trim().toUpperCase();
    onTickerChange(cleanTicker);
    try {
      const next = await startSimulation({
        ticker: cleanTicker,
        duration_seconds: duration,
        initial_price: initialPrice,
        starting_cash: startingCash,
        volatility,
        agents
      });
      setSession(next);
      setPriceSeries([{ tick: 0, price: next.current_price }]);
      setSandboxResult("");
      setAutoCommentary("");
      setAutoCommentaryModel("");
    } finally {
      setLoading(false);
    }
  }

  async function handleStop() {
    if (!session) return;
    await stopSimulation(session.session_id);
    const finalState = { ...session, running: false };
    setSession(finalState);
    void generateAutoCommentary("stopped", finalState);
  }

  async function handleSandbox() {
    if (!session) return;
    const result = await spinModalSandbox(sandboxPrompt, session.session_id);
    setSandboxResult(JSON.stringify(result, null, 2));
  }

  const totalEquity = useMemo(() => {
    if (!session) return 0;
    return Object.values(session.portfolios).reduce((sum, portfolio) => sum + (portfolio.equity ?? 0), 0);
  }, [session]);

  return (
    <section className="panel stack stagger">
      <header className="panel-header">
        <h2>Simulation Arena</h2>
        <p>Agent-vs-agent sandbox with order book impact, slippage, delayed news propagation, and crash regimes.</p>
      </header>

      <div className="glass-card stack">
        <div className="card-row simulation-form">
          <label>
            Ticker
            <input value={ticker} onChange={(event) => setTicker(event.target.value.toUpperCase())} />
          </label>
          <label>
            Initial Price
            <input
              type="number"
              min={1}
              value={initialPrice}
              onChange={(event) => setInitialPrice(Number(event.target.value) || 1)}
            />
          </label>
          <label>
            Duration (s)
            <input type="number" min={30} max={3600} value={duration} onChange={(event) => setDuration(Number(event.target.value) || 30)} />
          </label>
          <label>
            Volatility
            <input
              type="number"
              min={0.001}
              max={0.3}
              step={0.001}
              value={volatility}
              onChange={(event) => setVolatility(Number(event.target.value) || 0.02)}
            />
          </label>
          <label>
            Starting Cash
            <input
              type="number"
              min={1000}
              step={1000}
              value={startingCash}
              onChange={(event) => setStartingCash(Number(event.target.value) || 1000)}
            />
          </label>
          <button onClick={handleStart} disabled={loading || Boolean(session?.running)}>
            {loading ? "Starting…" : "Start Session"}
          </button>
          <button className="secondary" onClick={handleStop} disabled={!session?.running}>
            Stop
          </button>
        </div>

        <div className="agents-grid">
          {agents.map((agent, idx) => (
            <article key={agent.name} className="agent-card">
              <h4>{agent.name}</h4>
              <p className="muted">{agent.personality.replace("_", " ")}</p>
              <label>
                Aggressiveness {agent.aggressiveness.toFixed(2)}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={agent.aggressiveness}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    setAgents((prev) => prev.map((row, i) => (i === idx ? { ...row, aggressiveness: value } : row)));
                  }}
                />
              </label>
              <label>
                Risk Limit {agent.risk_limit.toFixed(2)}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={agent.risk_limit}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    setAgents((prev) => prev.map((row, i) => (i === idx ? { ...row, risk_limit: value } : row)));
                  }}
                />
              </label>
              <label>
                Trade Size
                <input
                  type="number"
                  min={1}
                  max={1000}
                  value={agent.trade_size}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    setAgents((prev) => prev.map((row, i) => (i === idx ? { ...row, trade_size: value } : row)));
                  }}
                />
              </label>
            </article>
          ))}
        </div>
      </div>

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
          <h3>Modal Sandbox Trigger</h3>
          <span className="muted">Natural-language spin-up</span>
        </div>
        <textarea value={sandboxPrompt} onChange={(event) => setSandboxPrompt(event.target.value)} rows={4} />
        <button onClick={handleSandbox} disabled={!session}>
          Spin Modal Sandbox
        </button>
        {sandboxResult ? <pre>{sandboxResult}</pre> : null}
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
