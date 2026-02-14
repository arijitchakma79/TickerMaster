import { useEffect, useMemo, useState } from "react";
import { addAlert, getTrackerSnapshot, setWatchlist, triggerTrackerPoll } from "../lib/api";
import { formatCompactNumber, formatCurrency, formatPercent } from "../lib/format";
import type { TrackerSnapshot, WSMessage } from "../lib/types";

interface Props {
  activeTicker: string;
  onTickerChange: (ticker: string) => void;
  trackerEvent?: WSMessage;
}

export default function TrackerPanel({ activeTicker, onTickerChange, trackerEvent }: Props) {
  const [watchlistInput, setWatchlistInput] = useState("AAPL,MSFT,NVDA,TSLA,SPY");
  const [snapshot, setSnapshot] = useState<TrackerSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [alertTicker, setAlertTicker] = useState(activeTicker);
  const [threshold, setThreshold] = useState(2);
  const [direction, setDirection] = useState<"up" | "down" | "either">("either");

  useEffect(() => {
    getTrackerSnapshot().then(setSnapshot).catch(() => null);
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

  async function handleAddAlert() {
    await addAlert({
      ticker: alertTicker.trim().toUpperCase(),
      threshold_percent: threshold,
      direction
    });
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
        <button onClick={handleUpdateWatchlist} disabled={loading}>
          Update List
        </button>
        <button className="secondary" onClick={handlePoll} disabled={loading}>
          {loading ? "Polling…" : "Poll Now"}
        </button>
      </div>

      <div className="glass-card card-row tracker-actions">
        <label>
          Alert Ticker
          <input value={alertTicker} onChange={(event) => setAlertTicker(event.target.value.toUpperCase())} />
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
    </section>
  );
}
