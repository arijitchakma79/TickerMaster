import { useEffect, useMemo, useState } from "react";
import {
  getNotificationPreferences,
  updateNotificationPreferences,
  type NotificationPreferencesPayload,
} from "../lib/api";

interface Props {
  connected: boolean;
  userId?: string | null;
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

function toUiPreferences(source: NotificationPreferencesPayload | null | undefined): NotificationPreferences {
  if (!source) return DEFAULT_PREFS;
  return {
    phone: String(source.phone_number ?? "").trim(),
    email: String(source.email ?? "").trim(),
    channel: source.preferred_channel ?? "push",
    frequency: source.alert_frequency ?? "realtime",
    priceAlerts: source.price_alerts ?? true,
    volumeAlerts: source.volume_alerts ?? true,
    simulationSummary: source.simulation_summary ?? true,
    quietStart: String(source.quiet_start ?? "22:00").slice(0, 5),
    quietEnd: String(source.quiet_end ?? "07:00").slice(0, 5)
  };
}

function toPayload(prefs: NotificationPreferences): Partial<NotificationPreferencesPayload> {
  return {
    phone_number: prefs.phone.trim() || null,
    email: prefs.email.trim() || null,
    preferred_channel: prefs.channel,
    alert_frequency: prefs.frequency,
    price_alerts: prefs.priceAlerts,
    volume_alerts: prefs.volumeAlerts,
    simulation_summary: prefs.simulationSummary,
    quiet_start: prefs.quietStart,
    quiet_end: prefs.quietEnd
  };
}

export default function TrackerPrefsRail({ connected, userId }: Props) {
  const [prefs, setPrefs] = useState<NotificationPreferences>(DEFAULT_PREFS);
  const [savedAt, setSavedAt] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const quietHoursDisabled = prefs.quietStart === prefs.quietEnd;

  useEffect(() => {
    let active = true;
    if (!userId) {
      setPrefs(DEFAULT_PREFS);
      setSavedAt("");
      setError("");
      return () => {
        active = false;
      };
    }

    setLoading(true);
    void getNotificationPreferences()
      .then((data) => {
        if (!active) return;
        setPrefs(toUiPreferences(data.preferences));
      })
      .catch((err: unknown) => {
        if (!active) return;
        setPrefs(DEFAULT_PREFS);
        setError(err instanceof Error ? err.message : "Could not load notification preferences.");
      })
      .finally(() => {
        if (!active) return;
        setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [userId]);

  const statusLabel = useMemo(() => {
    if (!userId) return "Sign in to save preferences";
    if (loading) return "Loading...";
    if (saving) return "Saving...";
    if (!savedAt) return "Not saved yet";
    return `Saved ${new Date(savedAt).toLocaleTimeString()}`;
  }, [loading, savedAt, saving, userId]);

  function update<K extends keyof NotificationPreferences>(key: K, value: NotificationPreferences[K]) {
    setPrefs((prev) => ({ ...prev, [key]: value }));
  }

  function savePreferences() {
    if (!userId) return;
    setSaving(true);
    setError("");
    void updateNotificationPreferences(toPayload(prefs))
      .then((data) => {
        if (data.preferences) {
          setPrefs(toUiPreferences(data.preferences));
        }
        setSavedAt(new Date().toISOString());
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Could not save notification preferences.");
      })
      .finally(() => {
        setSaving(false);
      });
  }

  function disableQuietHours() {
    setPrefs((prev) => {
      const marker = prev.quietStart || "00:00";
      return {
        ...prev,
        quietStart: marker,
        quietEnd: marker,
      };
    });
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
      <div className="tracker-quiet-hours-actions">
        <button
          type="button"
          className="secondary"
          onClick={disableQuietHours}
          disabled={quietHoursDisabled}
        >
          {quietHoursDisabled ? "Quiet Hours Disabled" : "Disable Quiet Hours"}
        </button>
        <p className="muted">
          {quietHoursDisabled
            ? "Quiet hours are off. Notifications can send any time."
            : "Set start and end to the same time to disable quiet hours."}
        </p>
      </div>

      <button onClick={savePreferences} disabled={!userId || loading || saving}>
        {saving ? "Saving..." : "Save Notification Preferences"}
      </button>
      <p className="muted">{statusLabel}</p>
      {error ? <p className="error">{error}</p> : null}
    </aside>
  );
}
