from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx

from app.config import Settings
from app.schemas import AlertConfig, TrackerSnapshot
from app.services.market_data import fetch_watchlist_metrics
from app.services.notifications import prepare_poke_recipe_handoff
from app.ws_manager import WSManager


class TrackerService:
    def __init__(self, settings: Settings, ws_manager: WSManager) -> None:
        self.settings = settings
        self.ws_manager = ws_manager
        self.watchlist = {ticker.upper() for ticker in settings.default_watchlist}
        self.alerts: List[AlertConfig] = []
        self._previous: Dict[str, Dict[str, float]] = {}
        self._latest_snapshot: TrackerSnapshot | None = None
        self._task: asyncio.Task | None = None

    def set_watchlist(self, tickers: List[str]) -> List[str]:
        clean = {t.strip().upper() for t in tickers if t.strip()}
        if clean:
            self.watchlist = clean
        return sorted(self.watchlist)

    def list_watchlist(self) -> List[str]:
        return sorted(self.watchlist)

    def add_alert(self, alert: AlertConfig) -> None:
        self.alerts.append(alert)

    def list_alerts(self) -> List[AlertConfig]:
        return self.alerts

    async def run_forever(self) -> None:
        while True:
            try:
                await self.poll_once()
            except Exception as exc:
                await self.ws_manager.broadcast(
                    {
                        "channel": "tracker",
                        "type": "tracker_error",
                        "error": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    channel="tracker",
                )
            await asyncio.sleep(max(10, self.settings.tracker_poll_interval_seconds))

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self.run_forever(), name="tracker-poller")

    async def snapshot(self) -> TrackerSnapshot:
        if self._latest_snapshot is None:
            await self.poll_once()
        assert self._latest_snapshot is not None
        return self._latest_snapshot

    async def _investigate_with_perplexity(self, ticker: str, context: str) -> str:
        if not self.settings.perplexity_api_key:
            return f"No Perplexity key configured. Baseline inference: {ticker} moved due to momentum + headline flow."

        body = {
            "model": self.settings.perplexity_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Explain probable catalysts for this stock move in one paragraph. "
                        f"Ticker: {ticker}. Trigger context: {context}."
                    ),
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.settings.perplexity_api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post("https://api.perplexity.ai/chat/completions", json=body, headers=headers)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return f"Perplexity investigation failed: {exc}."

    async def _synthesize_narrative(self, ticker: str, raw_context: str) -> str:
        prompt = (
            "Synthesize this into a concise high-signal trading narrative with risk factors. "
            f"Ticker: {ticker}. Context: {raw_context}"
        )

        if self.settings.cerebras_api_key:
            headers = {
                "Authorization": f"Bearer {self.settings.cerebras_api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": "llama-3.3-70b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post("https://api.cerebras.ai/v1/chat/completions", headers=headers, json=body)
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                pass

        if self.settings.nvidia_nim_api_key:
            headers = {
                "Authorization": f"Bearer {self.settings.nvidia_nim_api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": "meta/llama-3.1-70b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=body)
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                pass

        return (
            "Synthesis fallback: flow suggests short-term momentum displacement rather than structural repricing. "
            "Monitor volume follow-through, options skew, and macro event timing before conviction sizing."
        )

    async def poll_once(self) -> TrackerSnapshot:
        metrics = await asyncio.to_thread(fetch_watchlist_metrics, self.list_watchlist())

        alerts_triggered: List[Dict[str, Any]] = []
        for metric in metrics:
            previous = self._previous.get(metric.ticker)
            if previous:
                price_change = ((metric.price - previous["price"]) / previous["price"]) * 100 if previous["price"] else 0
                volume_spike = metric.volume and previous.get("volume") and metric.volume > previous["volume"] * 1.8

                for alert in self.alerts:
                    if alert.ticker != metric.ticker:
                        continue
                    if alert.direction == "up" and price_change < alert.threshold_percent:
                        continue
                    if alert.direction == "down" and price_change > -alert.threshold_percent:
                        continue
                    if alert.direction == "either" and abs(price_change) < alert.threshold_percent:
                        continue
                    alerts_triggered.append(
                        {
                            "ticker": metric.ticker,
                            "reason": f"Custom alert threshold hit ({price_change:.2f}%).",
                            "change_percent": round(price_change, 2),
                        }
                    )

                if abs(price_change) > 2.0 or volume_spike:
                    trigger_context = (
                        f"price_change={price_change:.2f}% volume={metric.volume} prior_volume={previous.get('volume')}"
                    )
                    why = await self._investigate_with_perplexity(metric.ticker, trigger_context)
                    synthesis = await self._synthesize_narrative(metric.ticker, why)
                    event = {
                        "ticker": metric.ticker,
                        "reason": trigger_context,
                        "investigation": why,
                        "analysis": synthesis,
                        "change_percent": round(price_change, 2),
                        "price": metric.price,
                    }
                    poke_handoff = await prepare_poke_recipe_handoff(
                        settings=self.settings,
                        title=f"TickerMaster Alert: {metric.ticker}",
                        body=synthesis[:200],
                        link=f"https://localhost:5173?tab=simulation&ticker={metric.ticker}",
                        metadata={
                            "ticker": metric.ticker,
                            "price": metric.price,
                            "change_percent": round(price_change, 2),
                            "reason": trigger_context,
                        },
                    )
                    event["poke"] = poke_handoff
                    alerts_triggered.append(event)

            self._previous[metric.ticker] = {
                "price": metric.price,
                "volume": float(metric.volume or 0),
            }

        snapshot = TrackerSnapshot(
            generated_at=datetime.now(timezone.utc),
            tickers=metrics,
            alerts_triggered=alerts_triggered,
        )
        self._latest_snapshot = snapshot

        await self.ws_manager.broadcast(
            {
                "channel": "tracker",
                "type": "tracker_snapshot",
                "generated_at": snapshot.generated_at.isoformat(),
                "tickers": [metric.model_dump() for metric in metrics],
                "alerts": alerts_triggered,
            },
            channel="tracker",
        )
        return snapshot
