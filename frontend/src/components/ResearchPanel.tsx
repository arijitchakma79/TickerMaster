import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { fetchCandles, runResearch } from "../lib/api";
import { formatPercent } from "../lib/format";
import type { CandlePoint, ResearchResponse } from "../lib/types";

interface Props {
  activeTicker: string;
  onTickerChange: (ticker: string) => void;
}

export default function ResearchPanel({ activeTicker, onTickerChange }: Props) {
  const [tickerInput, setTickerInput] = useState(activeTicker);
  const [timeframe, setTimeframe] = useState("7d");
  const [loading, setLoading] = useState(false);
  const [research, setResearch] = useState<ResearchResponse | null>(null);
  const [candles, setCandles] = useState<CandlePoint[]>([]);
  const [error, setError] = useState("");

  const sentimentLabel = useMemo(() => {
    if (!research) return "-";
    return `${formatPercent(research.aggregate_sentiment * 100)} / ${research.recommendation.replace("_", " ")}`;
  }, [research]);

  async function handleAnalyze() {
    setLoading(true);
    setError("");
    const ticker = tickerInput.trim().toUpperCase();
    onTickerChange(ticker);
    try {
      const [analysis, chart] = await Promise.all([runResearch(ticker, timeframe), fetchCandles(ticker, "3mo", "1d")]);
      setResearch(analysis);
      setCandles(chart);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Research request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel stack stagger">
      <header className="panel-header">
        <h2>Research</h2>
        <p>Perplexity Sonar + X + Reddit + prediction markets, summarized into high-signal narratives.</p>
      </header>

      <div className="card-row research-actions">
        <label>
          Ticker
          <input value={tickerInput} onChange={(event) => setTickerInput(event.target.value.toUpperCase())} maxLength={12} />
        </label>
        <label>
          Timeframe
          <select value={timeframe} onChange={(event) => setTimeframe(event.target.value)}>
            <option value="24h">24h</option>
            <option value="7d">7d</option>
            <option value="30d">30d</option>
          </select>
        </label>
        <button onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing…" : "Run Research"}
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
          <span className="muted">3M Daily</span>
        </div>
        <div className="chart-box">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={candles}>
              <defs>
                <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value: string) => new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                stroke="var(--muted)"
                minTickGap={24}
              />
              <YAxis stroke="var(--muted)" domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{ background: "var(--surface-2)", border: "1px solid var(--line)", borderRadius: 12 }}
                formatter={(value: number) => value.toFixed(2)}
                labelFormatter={(value: string) => new Date(value).toLocaleString()}
              />
              <Area type="monotone" dataKey="close" stroke="var(--accent)" fill="url(#priceFill)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

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
    </section>
  );
}
