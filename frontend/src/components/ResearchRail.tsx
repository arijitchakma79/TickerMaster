import { useEffect, useMemo, useState } from "react";
import { formatCurrency, formatPercent } from "../lib/format";
import type { MarketMetric, WSMessage } from "../lib/types";

interface Props {
  connected: boolean;
  activeTicker: string;
  onTickerSelect: (ticker: string) => void;
  trackerEvent?: WSMessage;
}

function parseTickers(raw: unknown): MarketMetric[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null;
      const item = entry as Record<string, unknown>;
      const ticker = typeof item.ticker === "string" ? item.ticker.toUpperCase() : "";
      const price = Number(item.price);
      const change = Number(item.change_percent);
      if (!ticker || !Number.isFinite(price) || !Number.isFinite(change)) return null;
      return { ticker, price, change_percent: change } as MarketMetric;
    })
    .filter((item): item is MarketMetric => item !== null);
}

export default function ResearchRail({ connected, activeTicker, onTickerSelect, trackerEvent }: Props) {
  const [tickers, setTickers] = useState<MarketMetric[]>([]);
  const [generatedAt, setGeneratedAt] = useState<string>("");

  useEffect(() => {
    if (!trackerEvent || trackerEvent.type !== "tracker_snapshot") return;
    setTickers(parseTickers(trackerEvent.tickers));
    setGeneratedAt(String(trackerEvent.generated_at ?? ""));
  }, [trackerEvent]);

  const winners = useMemo(
    () =>
      [...tickers]
        .filter((item) => item.change_percent > 0)
        .sort((a, b) => b.change_percent - a.change_percent)
        .slice(0, 5),
    [tickers]
  );
  const losers = useMemo(
    () =>
      [...tickers]
        .filter((item) => item.change_percent < 0)
        .sort((a, b) => a.change_percent - b.change_percent)
        .slice(0, 5),
    [tickers]
  );

  const queryTicker = activeTicker.trim().toUpperCase() || "SPY";
  const perplexityUrl = `https://www.perplexity.ai/search?q=${encodeURIComponent(`${queryTicker} stock news catalyst today`)}`;

  return (
    <aside className="event-rail glass-card research-rail">
      <div className="panel-header">
        <h3>Market Movers</h3>
        <span className={connected ? "dot dot-live" : "dot dot-offline"}>
          {connected ? "Realtime" : "Offline"}
        </span>
      </div>

      <div className="board-meta">
        <span>US Equities</span>
        <span>{generatedAt ? new Date(generatedAt).toLocaleTimeString() : "No snapshot"}</span>
      </div>

      {tickers.length === 0 ? <p className="muted">Waiting for tracker snapshot to compute current movers.</p> : null}

      <section className="market-board-section">
        <h4>Top Winners</h4>
        <div className="board-list">
          {winners.length > 0 ? (
            winners.map((row) => (
              <button
                key={`research-winner-${row.ticker}`}
                type="button"
                className={`board-item board-item-clickable${row.ticker === queryTicker ? " active" : ""}`}
                onClick={() => onTickerSelect(row.ticker)}
                aria-label={`Select ${row.ticker}`}
              >
                <div>
                  <p className="board-agent">{row.ticker}</p>
                  <p className="board-equity">{formatCurrency(row.price)}</p>
                </div>
                <div className="board-pnl positive">
                  <strong>{formatPercent(row.change_percent)}</strong>
                </div>
              </button>
            ))
          ) : (
            <p className="muted">No gainers in the latest snapshot.</p>
          )}
        </div>
      </section>

      <section className="market-board-section">
        <h4>Top Losers</h4>
        <div className="board-list">
          {losers.length > 0 ? (
            losers.map((row) => (
              <button
                key={`research-loser-${row.ticker}`}
                type="button"
                className={`board-item board-item-clickable${row.ticker === queryTicker ? " active" : ""}`}
                onClick={() => onTickerSelect(row.ticker)}
                aria-label={`Select ${row.ticker}`}
              >
                <div>
                  <p className="board-agent">{row.ticker}</p>
                  <p className="board-equity">{formatCurrency(row.price)}</p>
                </div>
                <div className="board-pnl negative">
                  <strong>{formatPercent(row.change_percent)}</strong>
                </div>
              </button>
            ))
          ) : (
            <p className="muted">No decliners in the latest snapshot.</p>
          )}
        </div>
      </section>

      <div className="glass-card quick-news-card">
        <h4>Perplexity News</h4>
        <p className="muted">Open a live catalyst brief for {queryTicker}.</p>
        <a href={perplexityUrl} target="_blank" rel="noreferrer">
          Open Perplexity Search
        </a>
      </div>
    </aside>
  );
}
