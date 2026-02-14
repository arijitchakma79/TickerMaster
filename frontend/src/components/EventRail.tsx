import type { WSMessage } from "../lib/types";

interface Props {
  events: WSMessage[];
  connected: boolean;
}

export default function EventRail({ events, connected }: Props) {
  return (
    <aside className="event-rail glass-card">
      <div className="panel-header">
        <h3>Live Wire</h3>
        <span className={connected ? "dot dot-live" : "dot dot-offline"}>
          {connected ? "Socket Live" : "Socket Offline"}
        </span>
      </div>
      <div className="event-list">
        {events.length === 0 ? <p className="muted">Waiting for events…</p> : null}
        {events.slice(0, 24).map((event, idx) => (
          <article key={`${event.type}-${idx}`} className="event-item">
            <div className="event-meta">
              <strong>{event.type}</strong>
              <span>{(event.channel as string) ?? "global"}</span>
            </div>
            {typeof event.timestamp === "string" ? <time>{new Date(event.timestamp).toLocaleTimeString()}</time> : null}
          </article>
        ))}
      </div>
    </aside>
  );
}
