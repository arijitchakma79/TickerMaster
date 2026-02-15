import { useEffect, useMemo, useState } from "react";
import { addAlert, getTrackerSnapshot, searchTickerDirectory, triggerTrackerPoll } from "../lib/api";
import { formatCompactNumber, formatCurrency, formatPercent } from "../lib/format";
import { resolveTickerCandidate } from "../lib/tickerInput";
import type { TickerLookup, TrackerSnapshot, WSMessage } from "../lib/types";
import WatchlistBar from "./WatchlistBar";

interface Props {
  activeTicker: string;
  onTickerChange: (ticker: string) => void;
  trackerEvent?: WSMessage;
  watchlist: string[];
  favorites: string[];
  onWatchlistChange: (tickers: string[]) => Promise<string[]>;
  onToggleFavorite: (ticker: string) => Promise<void>;
}

export default function TrackerPanel({
  activeTicker,
  onTickerChange,
  trackerEvent,
  watchlist,
  favorites,
  onWatchlistChange,
  onToggleFavorite
}: Props) {
  const [watchlistInput, setWatchlistInput] = useState("");
  const [watchlistInputFocused, setWatchlistInputFocused] = useState(false);
  const [watchlistSuggestions, setWatchlistSuggestions] = useState<TickerLookup[]>([]);
  const [watchlistSearchLoading, setWatchlistSearchLoading] = useState(false);
  const [snapshot, setSnapshot] = useState<TrackerSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
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

  const availableWatchlistSuggestions = useMemo(
    () => watchlistSuggestions.filter((entry) => !watchlist.includes(entry.ticker)),
    [watchlistSuggestions, watchlist]
  );
  const resolvedInputTicker = useMemo(
    () => resolveTickerCandidate(watchlistInput, watchlistSuggestions),
    [watchlistInput, watchlistSuggestions]
  );
  const alertTargetTicker = resolvedInputTicker || activeTicker;

  async function handleAddToWatchlist(rawInput?: string) {
    const candidate = rawInput ?? watchlistInput;
    const resolvedTicker = resolveTickerCandidate(candidate, watchlistSuggestions);
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

  async function handleAddAlert() {
    if (!alertTargetTicker) return;
    await addAlert({
      ticker: alertTargetTicker,
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

  function handleRemoveWatchlistTicker(symbol: string) {
    if (watchlist.length <= 1) return;
    void onWatchlistChange(watchlist.filter((tickerSymbol) => tickerSymbol !== symbol));
  }

  return (
    <section className="panel stack stagger">
      <WatchlistBar
        watchlist={watchlist}
        activeTicker={activeTicker}
        onSelectTicker={onTickerChange}
        onRemoveTicker={handleRemoveWatchlistTicker}
        favorites={favorites}
        onToggleFavorite={(ticker) => void onToggleFavorite(ticker)}
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

      <div className="glass-card card-row tracker-actions">
        <div className="tracker-alert-target">
          <p className="muted">Alert Target</p>
          <strong>{alertTargetTicker}</strong>
          <p className="muted">
            {resolvedInputTicker
              ? "Using the ticker resolved from Add Company / Ticker."
              : "Using the currently selected watchlist ticker."}
          </p>
        </div>
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
