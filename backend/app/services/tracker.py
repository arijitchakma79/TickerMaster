from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx

from app.config import Settings
from app.schemas import AlertConfig, ResearchRequest, TrackerSnapshot
from app.services.agent_logger import log_agent_activity
from app.services.market_data import fetch_watchlist_metrics
from app.services.notifications import prepare_poke_recipe_handoff
from app.services.sentiment import run_research
from app.services.tracker_repository import tracker_repo
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
        self._research_cache: Dict[str, Dict[str, Any]] = {}

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

    def _agent_in_cooldown(self, agent: dict[str, Any], minutes: int = 15) -> bool:
        raw = agent.get("last_alert_at")
        if not raw:
            return False
        try:
            stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            return False
        now = datetime.now(timezone.utc)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return now - stamp < timedelta(minutes=minutes)

    async def _get_research_snapshot(self, ticker: str, timeframe: str = "7d") -> dict[str, Any]:
        key = f"{ticker.upper()}:{timeframe}"
        cached = self._research_cache.get(key)
        if cached:
            at = cached.get("cached_at")
            if isinstance(at, datetime) and (datetime.now(timezone.utc) - at) < timedelta(minutes=5):
                return dict(cached.get("data") or {})
        try:
            data = await run_research(
                ResearchRequest(ticker=ticker.upper(), timeframe=timeframe, include_prediction_markets=False),
                self.settings,
            )
            payload = {
                "aggregate_sentiment": data.aggregate_sentiment,
                "recommendation": data.recommendation,
                "breakdown": {item.source: item.score for item in data.source_breakdown},
            }
            self._research_cache[key] = {"cached_at": datetime.now(timezone.utc), "data": payload}
            return payload
        except Exception:
            return {"aggregate_sentiment": 0.0, "recommendation": "hold", "breakdown": {}}

    async def _evaluate_agent(
        self,
        agent: dict[str, Any],
        metric: Any,
        price_change: float,
        volume_spike: bool,
    ) -> dict[str, Any] | None:
        triggers = agent.get("triggers") or {}
        def _as_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return default

        timeframe = str(triggers.get("research_timeframe") or "7d")
        research = await self._get_research_snapshot(metric.ticker, timeframe=timeframe)
        aggregate = float(research.get("aggregate_sentiment", 0.0))
        x_score = float((research.get("breakdown") or {}).get("X API", 0.0))

        bearish_threshold = _as_float(triggers.get("sentiment_bearish_threshold"), -0.25)
        bullish_threshold = _as_float(triggers.get("sentiment_bullish_threshold"), 0.25)
        x_bearish = _as_float(triggers.get("x_bearish_threshold"), -0.25)
        price_threshold = _as_float(triggers.get("price_change_pct"), 2.0)
        volume_ratio = _as_float(triggers.get("volume_spike_ratio"), 1.8)

        reasons: list[str] = []
        if abs(price_change) >= price_threshold:
            reasons.append(f"price move {price_change:.2f}% >= {price_threshold:.2f}%")
        if volume_spike:
            reasons.append(f"volume spike exceeded ratio {volume_ratio:.2f}")
        if aggregate <= bearish_threshold:
            reasons.append(f"broad sentiment bearish ({aggregate:.2f})")
        if aggregate >= bullish_threshold:
            reasons.append(f"broad sentiment bullish ({aggregate:.2f})")
        if x_score <= x_bearish:
            reasons.append(f"X sentiment deeply bearish ({x_score:.2f})")

        await log_agent_activity(
            module="tracker",
            agent_name=str(agent.get("name") or f"{metric.ticker} Associate"),
            action=f"Evaluated multi-factor signals for {metric.ticker}",
            status="running",
            user_id=agent.get("user_id"),
            details={
                "agent_id": agent.get("id"),
                "symbol": metric.ticker,
                "description": "Checked price, volume, and social sentiment signals.",
                "price_change_pct": round(price_change, 3),
                "volume_spike": bool(volume_spike),
                "aggregate_sentiment": round(aggregate, 3),
                "x_sentiment": round(x_score, 3),
                "trigger_matches": reasons,
            },
        )
        if not reasons:
            return None

        context = (
            f"Agent={agent.get('name')} symbol={metric.ticker} price={metric.price} "
            f"change={price_change:.2f}% aggregate_sentiment={aggregate:.2f} x_score={x_score:.2f} "
            f"matches={'; '.join(reasons)}"
        )
        narrative = await self._synthesize_narrative(metric.ticker, context)
        alert = tracker_repo.create_alert(
            symbol=metric.ticker,
            trigger_reason="; ".join(reasons),
            narrative=narrative,
            market_snapshot={
                "price": metric.price,
                "change_percent": round(price_change, 2),
                "volume": metric.volume,
                "sentiment": aggregate,
                "x_sentiment": x_score,
            },
            investigation_data={"research": research},
            user_id=agent.get("user_id"),
            agent_id=agent.get("id"),
        )
        update_payload = {
            "last_alert_at": datetime.now(timezone.utc).isoformat(),
            "total_alerts": int(agent.get("total_alerts") or 0) + 1,
        }
        tracker_repo.update_agent(user_id=agent.get("user_id"), agent_id=str(agent.get("id")), updates=update_payload)
        await log_agent_activity(
            module="tracker",
            agent_name=str(agent.get("name") or f"{metric.ticker} Associate"),
            action=f"Published associate alert for {metric.ticker}",
            status="success",
            user_id=agent.get("user_id"),
            details={
                "agent_id": agent.get("id"),
                "symbol": metric.ticker,
                "description": "Generated narrative from price/volume/sentiment context and persisted alert.",
                "trigger_reason": "; ".join(reasons),
                "narrative": narrative[:300],
            },
        )
        return {
            "ticker": metric.ticker,
            "agent_id": agent.get("id"),
            "agent_name": agent.get("name"),
            "reason": "; ".join(reasons),
            "analysis": narrative,
            "change_percent": round(price_change, 2),
            "price": metric.price,
            "alert_id": alert.get("id"),
        }

    async def poll_once(self) -> TrackerSnapshot:
        metrics = await asyncio.to_thread(fetch_watchlist_metrics, self.list_watchlist())
        symbol_agents: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for agent in tracker_repo.list_agents(user_id=None):
            if str(agent.get("status", "")).lower() != "active":
                continue
            symbol = str(agent.get("symbol", "")).upper()
            if symbol:
                symbol_agents[symbol].append(agent)

        alerts_triggered: List[Dict[str, Any]] = []
        for metric in metrics:
            attached_agents = symbol_agents.get(metric.ticker, [])
            if attached_agents:
                for agent in attached_agents:
                    await log_agent_activity(
                        module="tracker",
                        agent_name=str(agent.get("name") or f"{metric.ticker} Tracker"),
                        action=f"Monitoring {metric.ticker}: price {metric.price:.2f}, move {metric.change_percent:.2f}%",
                        status="running",
                        user_id=agent.get("user_id"),
                        details={
                            "agent_id": agent.get("id"),
                            "symbol": metric.ticker,
                            "description": "Scanning live quote and trigger thresholds.",
                            "change_percent": metric.change_percent,
                            "volume": metric.volume,
                        },
                    )
            else:
                await log_agent_activity(
                    module="tracker",
                    agent_name=f"{metric.ticker} Tracker",
                    action=f"Price check {metric.ticker} @ {metric.price}",
                    status="running",
                    details={
                        "symbol": metric.ticker,
                        "description": "Price scan completed with no dedicated deployed agent found.",
                        "change_percent": metric.change_percent,
                        "volume": metric.volume,
                    },
                )
            previous = self._previous.get(metric.ticker)
            if previous:
                price_change = ((metric.price - previous["price"]) / previous["price"]) * 100 if previous["price"] else 0
                volume_spike = metric.volume and previous.get("volume") and metric.volume > previous["volume"] * 1.8
                for agent in attached_agents:
                    if self._agent_in_cooldown(agent):
                        await log_agent_activity(
                            module="tracker",
                            agent_name=str(agent.get("name") or f"{metric.ticker} Associate"),
                            action=f"Agent cooldown active for {metric.ticker}",
                            status="pending",
                            user_id=agent.get("user_id"),
                            details={
                                "agent_id": agent.get("id"),
                                "symbol": metric.ticker,
                                "description": "Skipped trigger actions due to 15-minute cooldown window.",
                            },
                        )
                        continue
                    agent_alert = await self._evaluate_agent(agent=agent, metric=metric, price_change=price_change, volume_spike=bool(volume_spike))
                    if agent_alert:
                        alerts_triggered.append(agent_alert)

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
                    tracker_repo.create_alert(
                        symbol=metric.ticker,
                        trigger_reason=f"Custom alert threshold hit ({price_change:.2f}%).",
                        narrative=None,
                        market_snapshot={"price": metric.price, "change_percent": round(price_change, 2)},
                        user_id=None,
                    )
                    await log_agent_activity(
                        module="tracker",
                        agent_name=f"{metric.ticker} Tracker",
                        action=f"Custom trigger fired for {metric.ticker}",
                        status="success",
                        details={
                            "symbol": metric.ticker,
                            "description": "Alert condition matched custom price-change threshold.",
                            "price_change": round(price_change, 2),
                            "threshold": alert.threshold_percent,
                        },
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
                    event["poke_sent"] = bool(poke_handoff.get("delivered"))
                    alerts_triggered.append(event)
                    tracker_repo.create_alert(
                        symbol=metric.ticker,
                        trigger_reason=trigger_context,
                        narrative=synthesis,
                        market_snapshot={
                            "price": metric.price,
                            "change_percent": round(price_change, 2),
                            "volume": metric.volume,
                        },
                        investigation_data={"investigation": why, "poke": poke_handoff},
                        user_id=None,
                        poke_sent=bool(poke_handoff.get("delivered")),
                    )
                    await log_agent_activity(
                        module="tracker",
                        agent_name=f"{metric.ticker} Tracker",
                        action=f"Pipeline alert fired for {metric.ticker}",
                        status="success",
                        details={
                            "symbol": metric.ticker,
                            "description": "Full investigation pipeline ran and narrative alert was generated.",
                            "reason": trigger_context,
                        },
                    )

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
        await log_agent_activity(
            module="tracker",
            agent_name="Tracker Poller",
            action=f"Completed watchlist poll for {len(metrics)} symbols",
            status="success",
            details={
                "description": "Snapshot broadcasted to frontend monitoring feed.",
                "symbols": [metric.ticker for metric in metrics],
                "alerts_triggered": len(alerts_triggered),
            },
        )
        return snapshot
