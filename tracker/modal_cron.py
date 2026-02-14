from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import modal

app = modal.App("tickermaster-tracker")

tracker_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "yfinance==0.2.54",
    "httpx==0.28.1",
    "supabase==2.15.0",
    "pandas==2.2.3",
)


def _compute_rsi(series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    return float((100 - (100 / (1 + rs))).iloc[-1])


def evaluate_triggers(agent: dict[str, Any], snapshot: dict[str, Any]) -> str | None:
    triggers = agent.get("triggers", {}) or {}
    reasons: list[str] = []

    threshold = triggers.get("price_change_pct")
    if threshold and abs(snapshot["change_5m_pct"]) >= float(threshold):
        direction = "up" if snapshot["change_5m_pct"] > 0 else "down"
        reasons.append(f"Price moved {direction} {abs(snapshot['change_5m_pct']):.2f}% in 5m (threshold {threshold}%)")

    vol = triggers.get("volume_spike_ratio")
    if vol and snapshot["volume_ratio"] >= float(vol):
        reasons.append(f"Volume spike {snapshot['volume_ratio']:.2f}x (threshold {vol}x)")

    upper = triggers.get("rsi_upper")
    if upper and snapshot["rsi"] > float(upper):
        reasons.append(f"RSI overbought at {snapshot['rsi']:.1f} (> {upper})")

    lower = triggers.get("rsi_lower")
    if lower and snapshot["rsi"] < float(lower):
        reasons.append(f"RSI oversold at {snapshot['rsi']:.1f} (< {lower})")

    above = triggers.get("price_above")
    if above and snapshot["price"] >= float(above):
        reasons.append(f"Price crossed above ${above}")

    below = triggers.get("price_below")
    if below and snapshot["price"] <= float(below):
        reasons.append(f"Price crossed below ${below}")

    return " | ".join(reasons) if reasons else None


@app.function(
    schedule=modal.Period(minutes=1),
    image=tracker_image,
    secrets=[modal.Secret.from_name("tickermaster-secrets")],
    timeout=120,
)
def poll_all_agents() -> dict[str, Any]:
    import httpx
    import yfinance as yf
    from supabase import create_client

    supabase_url = os.environ["SUPABASE_URL"]
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")

    supabase = create_client(supabase_url, supabase_key)

    result = supabase.table("tracker_agents").select("*").eq("status", "active").execute()
    agents = result.data or []
    if not agents:
        return {"ok": True, "polled": 0, "alerts": 0}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for agent in agents:
        grouped.setdefault(str(agent["symbol"]).upper(), []).append(agent)

    alerts_count = 0

    with httpx.Client(timeout=15.0) as client:
        for symbol, symbol_agents in grouped.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")
                if hist.empty:
                    continue

                current_price = float(hist["Close"].iloc[-1])
                current_volume = float(hist["Volume"].iloc[-1])
                idx_5 = -5 if len(hist) >= 5 else 0
                idx_15 = -15 if len(hist) >= 15 else 0
                price_5m = float(hist["Close"].iloc[idx_5])
                price_15m = float(hist["Close"].iloc[idx_15])
                avg_volume = float(hist["Volume"].mean() or 1.0)
                rsi = _compute_rsi(hist["Close"])

                snapshot = {
                    "symbol": symbol,
                    "price": round(current_price, 2),
                    "change_5m_pct": round(((current_price - price_5m) / price_5m) * 100 if price_5m else 0.0, 4),
                    "change_15m_pct": round(((current_price - price_15m) / price_15m) * 100 if price_15m else 0.0, 4),
                    "volume": int(current_volume),
                    "volume_ratio": round(current_volume / avg_volume if avg_volume else 1.0, 3),
                    "rsi": round(rsi, 2),
                }

                for agent in symbol_agents:
                    # 15-minute cooldown
                    last_alert_at = agent.get("last_alert_at")
                    if last_alert_at:
                        try:
                            ts = datetime.fromisoformat(str(last_alert_at).replace("Z", "+00:00"))
                            if datetime.now(timezone.utc) - ts < timedelta(minutes=15):
                                continue
                        except Exception:
                            pass

                    reason = evaluate_triggers(agent, snapshot)
                    if not reason:
                        continue

                    payload = {
                        "symbol": symbol,
                        "trigger_reason": reason,
                        "narrative": None,
                        "market_snapshot": snapshot,
                        "investigation_data": {"source": "modal_cron"},
                        "agent_id": agent.get("id"),
                        "user_id": agent.get("user_id"),
                    }
                    response = client.post(f"{backend_url}/api/tracker/emit-alert", json=payload)
                    if response.status_code < 300:
                        alerts_count += 1
                        supabase.table("tracker_agents").update(
                            {
                                "last_alert_at": datetime.now(timezone.utc).isoformat(),
                                "last_checked_at": datetime.now(timezone.utc).isoformat(),
                                "last_price": snapshot["price"],
                                "total_alerts": int(agent.get("total_alerts", 0)) + 1,
                            }
                        ).eq("id", agent["id"]).execute()
            except Exception as exc:
                print(f"[modal_cron] polling error for {symbol}: {exc}")
                continue

    return {"ok": True, "polled": len(agents), "alerts": alerts_count}


@app.local_entrypoint()
def main():
    print(poll_all_agents.local())
