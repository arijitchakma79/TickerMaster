import { useEffect, useMemo, useState } from "react";
import { formatCurrency, formatPercent } from "../lib/format";
import type { WSMessage } from "../lib/types";

interface Props {
  connected: boolean;
  simulationEvent?: WSMessage;
  activeSessionId?: string | null;
  simulationActive?: boolean;
  className?: string;
}

interface StandingRow {
  agent: string;
  equity: number;
  pnl: number;
  pnlPercent: number;
}

function toPortfolioSnapshot(raw: unknown): Record<string, { equity?: number }> {
  if (!raw || typeof raw !== "object") return {};
  const out: Record<string, { equity?: number }> = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (value && typeof value === "object") {
      const equity = Number((value as Record<string, unknown>).equity);
      out[key] = { equity: Number.isFinite(equity) ? equity : undefined };
    }
  }
  return out;
}

export default function EventRail({
  connected,
  simulationEvent,
  activeSessionId,
  simulationActive = false,
  className = ""
}: Props) {
  const [ticker, setTicker] = useState<string>("-");
  const [tick, setTick] = useState<number>(0);
  const [baselineEquity, setBaselineEquity] = useState<Record<string, number>>({});
  const [currentEquity, setCurrentEquity] = useState<Record<string, number>>({});
  const hasSession = Boolean(activeSessionId);

  useEffect(() => {
    if (!activeSessionId) {
      setTicker("-");
      setTick(0);
      setBaselineEquity({});
      setCurrentEquity({});
      return;
    }
  }, [activeSessionId]);

  useEffect(() => {
    if (!activeSessionId) return;
    if (!simulationEvent || simulationEvent.type !== "tick") return;

    const sessionId = typeof simulationEvent.session_id === "string" ? simulationEvent.session_id : "";
    if (!sessionId || sessionId !== activeSessionId) return;
    const eventTicker = typeof simulationEvent.ticker === "string" ? simulationEvent.ticker : "-";
    const eventTick = Number(simulationEvent.tick ?? 0);
    const portfolio = toPortfolioSnapshot(simulationEvent.portfolio_snapshot);
    const current: Record<string, number> = {};
    for (const [agent, snapshot] of Object.entries(portfolio)) {
      const equity = Number(snapshot.equity);
      if (Number.isFinite(equity)) current[agent] = equity;
    }

    setTicker(eventTicker);
    setTick(eventTick);
    setCurrentEquity(current);

    setBaselineEquity((prev) => {
      if (Object.keys(prev).length === 0) return current;
      const next = { ...prev };
      for (const [agent, equity] of Object.entries(current)) {
        if (!(agent in next)) next[agent] = equity;
      }
      return next;
    });
  }, [simulationEvent, activeSessionId]);

  const standings = useMemo<StandingRow[]>(() => {
    return Object.entries(currentEquity).map(([agent, equity]) => {
      const base = baselineEquity[agent] ?? equity;
      const pnl = equity - base;
      const pnlPercent = base ? (pnl / base) * 100 : 0;
      return { agent, equity, pnl, pnlPercent };
    });
  }, [currentEquity, baselineEquity]);

  const winners = useMemo(
    () => [...standings].sort((a, b) => b.pnl - a.pnl).slice(0, 4),
    [standings]
  );
  const losers = useMemo(
    () => [...standings].sort((a, b) => a.pnl - b.pnl).slice(0, 4),
    [standings]
  );

  return (
    <aside className={`event-rail glass-card ${className}`.trim()}>
      <div className="panel-header">
        <h3>Market Board</h3>
        <span className={connected ? "dot dot-live" : "dot dot-offline"}>
          {connected ? "Socket Live" : "Socket Offline"}
        </span>
      </div>

      <div className="board-meta">
        <span>{hasSession ? (simulationActive ? `${ticker} roundtable` : `${ticker} final`) : "No active session"}</span>
        <span>{hasSession ? `Tick ${tick}` : "Tick -"}</span>
      </div>

      {standings.length === 0 ? <p className="muted">Run a simulation to view top winners and losers.</p> : null}

      <section className="market-board-section">
        <h4>Top Winners</h4>
        <div className="board-list">
          {winners.map((row) => (
            <article key={`winner-${row.agent}`} className="board-item">
              <div>
                <p className="board-agent">{row.agent}</p>
                <p className="board-equity">{formatCurrency(row.equity)}</p>
              </div>
              <div className="board-pnl positive">
                <strong>{formatCurrency(row.pnl)}</strong>
                <span>{formatPercent(row.pnlPercent)}</span>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="market-board-section">
        <h4>Top Losers</h4>
        <div className="board-list">
          {losers.map((row) => (
            <article key={`loser-${row.agent}`} className="board-item">
              <div>
                <p className="board-agent">{row.agent}</p>
                <p className="board-equity">{formatCurrency(row.equity)}</p>
              </div>
              <div className={`board-pnl ${row.pnl < 0 ? "negative" : "positive"}`}>
                <strong>{formatCurrency(row.pnl)}</strong>
                <span>{formatPercent(row.pnlPercent)}</span>
              </div>
            </article>
          ))}
        </div>
      </section>
    </aside>
  );
}
