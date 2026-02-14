import { useEffect, useMemo, useState } from "react";
import ResearchRail from "./components/ResearchRail";
import ResearchPanel from "./components/ResearchPanel";
import SimulationPanel from "./components/SimulationPanel";
import TrackerPrefsRail from "./components/TrackerPrefsRail";
import TrackerPanel from "./components/TrackerPanel";
import { getApiUrl, getWatchlist, setWatchlist as setTrackerWatchlist } from "./lib/api";
import { useSocket } from "./hooks/useSocket";
import brandLogo from "./images/TickerMaster.png";
import moonIcon from "./images/moon.png";
import sunIcon from "./images/sun.png";

type Tab = "research" | "simulation" | "tracker";
type Theme = "light" | "dark";
const DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"];

function tabFromQuery(): Tab {
  const params = new URLSearchParams(window.location.search);
  const value = params.get("tab");
  if (value === "research" || value === "simulation" || value === "tracker") return value;
  return "simulation";
}

function tickerFromQuery() {
  return (new URLSearchParams(window.location.search).get("ticker") ?? "AAPL").toUpperCase();
}

function normalizeWatchlist(tickers: string[]) {
  const cleaned = tickers
    .map((symbol) => symbol.trim().toUpperCase())
    .filter(Boolean);
  return Array.from(new Set(cleaned));
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromQuery());
  const [ticker, setTicker] = useState(tickerFromQuery());
  const [watchlist, setWatchlist] = useState<string[]>(() =>
    normalizeWatchlist([tickerFromQuery(), ...DEFAULT_WATCHLIST])
  );
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = window.localStorage.getItem("tickermaster-theme");
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  const { connected, events, lastSimulationTick, lastSimulationLifecycle, lastTrackerSnapshot } = useSocket();

  useEffect(() => {
    getWatchlist()
      .then((serverWatchlist) => {
        const synced = normalizeWatchlist(serverWatchlist);
        if (synced.length === 0) return;
        setWatchlist(synced);
        if (!synced.includes(ticker)) {
          setTicker(synced[0]);
        }
      })
      .catch(() => null);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    document.documentElement.style.colorScheme = theme;
    window.localStorage.setItem("tickermaster-theme", theme);
  }, [theme]);

  const title = useMemo(() => {
    if (tab === "research") return "Research Workbench";
    if (tab === "simulation") return "Simulation Arena";
    return "Ticker Tracker";
  }, [tab]);

  async function handleWatchlistChange(nextSymbols: string[]) {
    const normalized = normalizeWatchlist(nextSymbols);
    if (normalized.length === 0) return watchlist;

    try {
      const serverUpdated = normalizeWatchlist(await setTrackerWatchlist(normalized));
      const nextList = serverUpdated.length > 0 ? serverUpdated : normalized;
      setWatchlist(nextList);
      if (!nextList.includes(ticker)) {
        setTicker(nextList[0]);
      }
      return nextList;
    } catch {
      setWatchlist(normalized);
      if (!normalized.includes(ticker)) {
        setTicker(normalized[0]);
      }
      return normalized;
    }
  }

  return (
    <div className="app-shell">
      <div className="brand-corner">
        <div className="brand-lockup" aria-label="TickerMaster">
          <img src={brandLogo} alt="TickerMaster" className="brand-logo-image" />
        </div>
      </div>
      <div className="toggle-corner">
        <div className="theme-switch-wrap">
          <button
            type="button"
            className={`theme-switch ${theme}`}
            onClick={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            aria-pressed={theme === "dark"}
            title={theme === "light" ? "Light mode" : "Dark mode"}
          >
            <span className="theme-switch-track" />
            <span className="theme-switch-thumb">
              <img src={theme === "light" ? sunIcon : moonIcon} alt="" />
            </span>
          </button>
        </div>
      </div>

      <div className="ambient ambient-1" />
      <div className="ambient ambient-2" />

      <header className="hero glass-card">
        <div>
          <h1>{title}</h1>
          <p className="subtitle">
            Financial AI sandbox for learning trade execution, sentiment asymmetry, and real-time catalyst tracking.
          </p>
        </div>
        <div className="hero-meta">
          <span className={connected ? "dot dot-live" : "dot dot-offline"}>{connected ? "Realtime Online" : "Socket Offline"}</span>
          <span className="muted">API: {getApiUrl()}</span>
        </div>
      </header>

      <nav className="tab-row">
        <button
          className={tab === "research" ? "tab active" : "tab"}
          onClick={() => setTab("research")}
          aria-current={tab === "research" ? "page" : undefined}
        >
          <span className="tab-inner">
            <span className="tab-icon" aria-hidden="true">
              🔎
            </span>
            <span>Research</span>
          </span>
        </button>
        <button
          className={tab === "simulation" ? "tab active core" : "tab core"}
          onClick={() => setTab("simulation")}
          aria-current={tab === "simulation" ? "page" : undefined}
        >
          <span className="tab-inner">
            <span className="tab-icon arena" aria-hidden="true">
              🛡️⚔️
            </span>
            <span>Simulation</span>
          </span>
        </button>
        <button
          className={tab === "tracker" ? "tab active" : "tab"}
          onClick={() => setTab("tracker")}
          aria-current={tab === "tracker" ? "page" : undefined}
        >
          <span className="tab-inner">
            <span className="tab-icon" aria-hidden="true">
              📊
            </span>
            <span>Tracker</span>
          </span>
        </button>
      </nav>

      <main className={tab === "simulation" ? "layout-grid layout-grid-single" : "layout-grid"}>
        <div>
          {tab === "research" ? (
            <ResearchPanel activeTicker={ticker} onTickerChange={setTicker} connected={connected} events={events} />
          ) : null}
          {tab === "simulation" ? (
            <SimulationPanel
              activeTicker={ticker}
              onTickerChange={setTicker}
              connected={connected}
              simulationEvent={lastSimulationTick}
              simulationLifecycleEvent={lastSimulationLifecycle}
            />
          ) : null}
          {tab === "tracker" ? (
            <TrackerPanel
              activeTicker={ticker}
              onTickerChange={setTicker}
              trackerEvent={lastTrackerSnapshot}
              watchlist={watchlist}
              onWatchlistChange={handleWatchlistChange}
            />
          ) : null}
        </div>

        {tab === "research" ? (
          <ResearchRail
            connected={connected}
            activeTicker={ticker}
            onTickerSelect={setTicker}
            trackerEvent={lastTrackerSnapshot}
          />
        ) : null}
        {tab === "tracker" ? <TrackerPrefsRail connected={connected} /> : null}
      </main>
    </div>
  );
}
