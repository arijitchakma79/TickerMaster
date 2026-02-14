import { useEffect, useMemo, useState } from "react";
import { getWsUrl } from "../lib/api";
import type { WSMessage } from "../lib/types";

export function useSocket() {
  const [events, setEvents] = useState<WSMessage[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let closed = false;
    let socket: WebSocket | null = null;
    let heartbeat: number | null = null;
    let reconnectTimer: number | null = null;

    const clearHeartbeat = () => {
      if (heartbeat !== null) {
        window.clearInterval(heartbeat);
        heartbeat = null;
      }
    };

    const clearReconnect = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const scheduleReconnect = () => {
      if (closed || reconnectTimer !== null) return;
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        connect();
      }, 2000);
    };

    const connect = () => {
      if (closed) return;
      socket = new WebSocket(getWsUrl());

      socket.onopen = () => {
        setConnected(true);
        clearReconnect();
        socket?.send("ping");
        clearHeartbeat();
        heartbeat = window.setInterval(() => {
          if (socket?.readyState === WebSocket.OPEN) {
            socket.send("ping");
          }
        }, 10000);
      };

      socket.onclose = () => {
        setConnected(false);
        clearHeartbeat();
        scheduleReconnect();
      };

      socket.onerror = () => {
        setConnected(false);
      };

      socket.onmessage = (ev) => {
        try {
          const message = JSON.parse(ev.data) as WSMessage;
          if (message.type === "pong") return;
          setEvents((prev) => [message, ...prev].slice(0, 120));
        } catch {
          // Ignore malformed payloads.
        }
      };
    };

    connect();

    return () => {
      closed = true;
      clearReconnect();
      clearHeartbeat();
      socket?.close();
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
