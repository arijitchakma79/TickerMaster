import { useEffect, useMemo, useState } from "react";
import { getWsUrl } from "../lib/api";
import type { WSMessage } from "../lib/types";

export function useSocket() {
  const [events, setEvents] = useState<WSMessage[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const socket = new WebSocket(getWsUrl());

    socket.onopen = () => {
      setConnected(true);
      socket.send("ping");
    };

    socket.onclose = () => {
      setConnected(false);
    };

    socket.onmessage = (ev) => {
      try {
        const message = JSON.parse(ev.data) as WSMessage;
        setEvents((prev) => [message, ...prev].slice(0, 120));
      } catch {
        // Ignore malformed payloads.
      }
    };

    const heartbeat = window.setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send("ping");
      }
    }, 10000);

    return () => {
      window.clearInterval(heartbeat);
      socket.close();
    };
  }, []);

  const lastSimulationTick = useMemo(
    () => events.find((event) => event.channel === "simulation" && event.type === "tick"),
    [events]
  );

  const lastSimulationLifecycle = useMemo(
    () =>
      events.find(
        (event) =>
          event.channel === "simulation" &&
          (event.type === "simulation_completed" || event.type === "simulation_stopped")
      ),
    [events]
  );

  const lastTrackerSnapshot = useMemo(
    () => events.find((event) => event.channel === "tracker" && event.type === "tracker_snapshot"),
    [events]
  );

  return {
    connected,
    events,
    lastSimulationTick,
    lastSimulationLifecycle,
    lastTrackerSnapshot
  };
}
