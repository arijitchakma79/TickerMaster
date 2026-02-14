import { useEffect, useMemo, useState } from "react";
import EventRail from "./components/EventRail";
import IntegrationStatus from "./components/IntegrationStatus";
import ResearchPanel from "./components/ResearchPanel";
import SimulationPanel from "./components/SimulationPanel";
import TrackerPanel from "./components/TrackerPanel";
import { fetchIntegrations, getApiUrl } from "./lib/api";
import { useSocket } from "./hooks/useSocket";
import brandLogo from "./images/TickerMaster.png";
import moonIcon from "./images/moon.png";
import sunIcon from "./images/sun.png";

type Tab = "research" | "simulation" | "tracker";
type Theme = "light" | "dark";

function tabFromQuery(): Tab {
  const params = new URLSearchParams(window.location.search);
  const value = params.get("tab");
  if (value === "simulation" || value === "tracker") return value;
  return "research";
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromQuery());
  const [ticker, setTicker] = useState((new URLSearchParams(window.location.search).get("ticker") ?? "AAPL").toUpperCase());
  const [integrations, setIntegrations] = useState<Record<string, boolean>>({});
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = window.localStorage.getItem("tickermaster-theme");
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  const { connected, events, lastSimulationTick, lastSimulationLifecycle, lastTrackerSnapshot } = useSocket();

  useEffect(() => {
    fetchIntegrations().then(setIntegrations).catch(() => setIntegrations({}));
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

      <IntegrationStatus integrations={integrations} />

      <nav className="tab-row">
        <button
          className={tab === "research" ? "tab active" : "tab"}
          onClick={() => setTab("research")}
          aria-current={tab === "research" ? "page" : undefined}
        >
          Research
        </button>
        <button
          className={tab === "simulation" ? "tab active" : "tab"}
          onClick={() => setTab("simulation")}
          aria-current={tab === "simulation" ? "page" : undefined}
        >
          Simulation
        </button>
        <button
          className={tab === "tracker" ? "tab active" : "tab"}
          onClick={() => setTab("tracker")}
          aria-current={tab === "tracker" ? "page" : undefined}
        >
          Tracker
        </button>
      </nav>

      <main className="layout-grid">
        <div>
          {tab === "research" ? <ResearchPanel activeTicker={ticker} onTickerChange={setTicker} /> : null}
          {tab === "simulation" ? (
            <SimulationPanel
              activeTicker={ticker}
              onTickerChange={setTicker}
              simulationEvent={lastSimulationTick}
              simulationLifecycleEvent={lastSimulationLifecycle}
            />
          ) : null}
          {tab === "tracker" ? (
            <TrackerPanel activeTicker={ticker} onTickerChange={setTicker} trackerEvent={lastTrackerSnapshot} />
          ) : null}
        </div>

        <EventRail events={events} connected={connected} />
      </main>
    </div>
  );
}
