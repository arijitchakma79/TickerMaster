import { useEffect, useMemo, useState } from "react";

interface Props {
  connected: boolean;
}

interface NotificationPreferences {
  phone: string;
  email: string;
  channel: "sms" | "email" | "push";
  frequency: "realtime" | "hourly" | "daily";
  priceAlerts: boolean;
  volumeAlerts: boolean;
  simulationSummary: boolean;
  quietStart: string;
  quietEnd: string;
}

const STORAGE_KEY = "tickermaster-notification-preferences";

const DEFAULT_PREFS: NotificationPreferences = {
  phone: "",
  email: "",
  channel: "push",
  frequency: "realtime",
  priceAlerts: true,
  volumeAlerts: true,
  simulationSummary: true,
  quietStart: "22:00",
  quietEnd: "07:00"
};

function readStoredPreferences(): NotificationPreferences {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_PREFS;
    const parsed = JSON.parse(raw) as Partial<NotificationPreferences>;
    return {
      ...DEFAULT_PREFS,
      ...parsed
    };
  } catch {
    return DEFAULT_PREFS;
  }
}

export default function TrackerPrefsRail({ connected }: Props) {
  const [prefs, setPrefs] = useState<NotificationPreferences>(DEFAULT_PREFS);
  const [savedAt, setSavedAt] = useState<string>("");

  useEffect(() => {
    setPrefs(readStoredPreferences());
  }, []);

  const statusLabel = useMemo(() => {
    if (!savedAt) return "Not saved yet";
    return `Saved ${new Date(savedAt).toLocaleTimeString()}`;
  }, [savedAt]);

  function update<K extends keyof NotificationPreferences>(key: K, value: NotificationPreferences[K]) {
    setPrefs((prev) => ({ ...prev, [key]: value }));
  }

  function savePreferences() {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
    setSavedAt(new Date().toISOString());
  }

  return (
    <aside className="event-rail glass-card tracker-prefs-rail">
      <div className="panel-header">
        <h3>Notification Preferences</h3>
        <span className={connected ? "dot dot-live" : "dot dot-offline"}>
          {connected ? "Delivery Ready" : "Offline"}
        </span>
      </div>

      <label>
        Phone Number
        <input
          value={prefs.phone}
          onChange={(event) => update("phone", event.target.value)}
          placeholder="+1 555 123 4567"
        />
      </label>

      <label>
        Email
        <input
          type="email"
          value={prefs.email}
          onChange={(event) => update("email", event.target.value)}
          placeholder="you@example.com"
        />
      </label>

      <label>
        Preferred Channel
        <select value={prefs.channel} onChange={(event) => update("channel", event.target.value as NotificationPreferences["channel"])}>
          <option value="push">Push</option>
          <option value="sms">SMS</option>
          <option value="email">Email</option>
        </select>
      </label>

      <label>
        Alert Frequency
        <select value={prefs.frequency} onChange={(event) => update("frequency", event.target.value as NotificationPreferences["frequency"])}>
          <option value="realtime">Realtime</option>
          <option value="hourly">Hourly Digest</option>
          <option value="daily">Daily Digest</option>
        </select>
      </label>

      <label className="tracker-pref-check">
        <input
          type="checkbox"
          checked={prefs.priceAlerts}
          onChange={(event) => update("priceAlerts", event.target.checked)}
        />
        Price Spike Alerts
      </label>

      <label className="tracker-pref-check">
        <input
          type="checkbox"
          checked={prefs.volumeAlerts}
          onChange={(event) => update("volumeAlerts", event.target.checked)}
        />
        Volume Anomaly Alerts
      </label>

      <label className="tracker-pref-check">
        <input
          type="checkbox"
          checked={prefs.simulationSummary}
          onChange={(event) => update("simulationSummary", event.target.checked)}
        />
        Simulation Summaries
      </label>

      <div className="tracker-quiet-hours">
        <label>
          Quiet Start
          <input
            type="time"
            value={prefs.quietStart}
            onChange={(event) => update("quietStart", event.target.value)}
          />
        </label>
        <label>
          Quiet End
          <input
            type="time"
            value={prefs.quietEnd}
            onChange={(event) => update("quietEnd", event.target.value)}
          />
        </label>
      </div>

      <button onClick={savePreferences}>Save Notification Preferences</button>
      <p className="muted">{statusLabel}</p>
    </aside>
  );
}
