import type { WSMessage } from "../lib/types";

type TabTarget = "research" | "simulation" | "tracker";

interface Props {
  events: WSMessage[];
  connected: boolean;
  onNavigate: (target: { tab: TabTarget; ticker?: string; agentId?: string }) => void;
}

interface WireCard {
  key: string;
  module: string;
  agentName: string;
  action: string;
  description: string;
  status: string;
  ticker?: string;
  agentId?: string;
  timestamp?: string;
}

function extractTicker(event: WSMessage): string | undefined {
  const details = (event.details ?? {}) as Record<string, unknown>;
  const direct = details.symbol ?? details.ticker ?? event.symbol;
  if (typeof direct === "string" && direct.trim()) return direct.trim().toUpperCase();

  const action = typeof event.action === "string" ? event.action : "";
  const match = action.toUpperCase().match(/\b[A-Z]{1,5}\b/);
  return match?.[0];
}

function toCards(events: WSMessage[]): WireCard[] {
  const items: WireCard[] = [];
  const seen = new Set<string>();

  for (const event of events) {
    if (event.type !== "agent_activity") continue;
    const module = String(event.module ?? "global").toLowerCase();
    const agentName = String(event.agent_name ?? "Unnamed Agent");
    const key = `${module}:${agentName}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const details = (event.details ?? {}) as Record<string, unknown>;
    items.push({
      key,
      module,
      agentName,
      action: String(event.action ?? "No action reported"),
      description: String(details.description ?? "Running background workflow."),
      status: String(event.status ?? "success"),
      ticker: extractTicker(event),
      agentId: typeof details.agent_id === "string" ? details.agent_id : undefined,
      timestamp: (typeof event.created_at === "string" ? event.created_at : undefined) ?? (typeof event.timestamp === "string" ? event.timestamp : undefined)
    });
    if (items.length >= 18) break;
  }

  return items;
}

function routeFor(module: string): TabTarget {
  if (module === "simulation") return "simulation";
  if (module === "tracker") return "tracker";
  return "research";
}

export default function EventRail({ events, connected, onNavigate }: Props) {
  const cards = toCards(events);

  return (
    <aside className="event-rail glass-card">
      <div className="panel-header">
        <h3>Live Wire</h3>
        <span className={connected ? "dot dot-live" : "dot dot-offline"}>
          {connected ? "Socket Live" : "Socket Offline"}
        </span>
      </div>
      <div className="event-list">
        {cards.length === 0 ? <p className="muted">Waiting for agent activity…</p> : null}
        {cards.map((card) => (
          <button
            type="button"
            key={card.key}
            className="event-item event-item-button"
            onClick={() => onNavigate({ tab: routeFor(card.module), ticker: card.ticker, agentId: card.agentId })}
            title="Open related module"
          >
            <div className="event-meta">
              <strong>{card.agentName}</strong>
              <span>{card.module}</span>
            </div>
            <p className="event-action">{card.action}</p>
            <p className="event-desc">{card.description}</p>
            <div className="event-meta event-meta-bottom">
              <span className={`pill ${card.status === "success" ? "bullish" : card.status === "running" ? "neutral" : "bearish"}`}>
                {card.status}
              </span>
              {card.ticker ? <span className="muted">{card.ticker}</span> : null}
            </div>
            {card.timestamp ? <time>{new Date(card.timestamp).toLocaleTimeString()}</time> : null}
          </button>
        ))}
      </div>
    </aside>
  );
}
