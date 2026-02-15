from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List
from zoneinfo import ZoneInfo

import httpx
import numpy as np

from app.config import Settings
from app.schemas import AlertConfig, ResearchRequest, TrackerSnapshot
from app.services.agent_logger import log_agent_activity
from app.services.browserbase_scraper import run_deep_research
from app.services.llm import decide_tracker_runtime_plan
from app.services.market_data import (
    fetch_metric,
    fetch_watchlist_metrics,
    is_metric_quality_valid,
    resolve_symbol_input,
    search_tickers,
)
from app.services.mcp_tool_router import collect_tracker_research_via_mcp
from app.services.notifications import dispatch_alert_notification
from app.services.sentiment import run_research_with_source_selection
from app.services.tracker_csv import append_agent_memory_documents, append_alert_context_csv
from app.services.tracker_repository import tracker_repo
from app.services.database import get_supabase
from app.ws_manager import WSManager

if TYPE_CHECKING:
    from app.services.simulation import SimulationOrchestrator


class TrackerService:
    def __init__(
        self,
        settings: Settings,
        ws_manager: WSManager,
        orchestrator: "SimulationOrchestrator | None" = None,
    ) -> None:
        self.settings = settings
        self.ws_manager = ws_manager
        self.orchestrator = orchestrator
        self.watchlist = {ticker.upper() for ticker in settings.default_watchlist}
        self.alerts: List[AlertConfig] = []
        self._previous: Dict[str, Dict[str, float]] = {}
        self._latest_snapshot: TrackerSnapshot | None = None
        self._task: asyncio.Task | None = None
        self._research_cache: Dict[str, Dict[str, Any]] = {}
        self._notification_pref_cache: Dict[str, Dict[str, Any]] = {}
        self._instruction_cache: Dict[str, Dict[str, Any]] = {}
        self._tool_plan_cache: Dict[str, Dict[str, Any]] = {}
        self._rng = np.random.default_rng()
        self._startup_migrations_done = False

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
        if not self._startup_migrations_done:
            try:
                await self._run_startup_migrations()
            except Exception as exc:
                await log_agent_activity(
                    module="tracker",
                    agent_name="Tracker Poller",
                    action="Startup migration failed",
                    status="error",
                    details={
                        "description": "Tracker startup migration encountered an exception and was skipped.",
                        "error": str(exc),
                    },
                )
            self._startup_migrations_done = True
        self._task = asyncio.create_task(self.run_forever(), name="tracker-poller")

    async def stop(self) -> None:
        if self._task is None:
            return
        if self._task.done():
            self._task = None
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _symbol_quality_ok(self, symbol: str) -> bool:
        token = str(symbol or "").strip().upper().replace(".", "-")
        if not token:
            return False
        try:
            metric = await asyncio.to_thread(fetch_metric, token)
        except Exception:
            return False
        return is_metric_quality_valid(metric)

    def _is_confident_symbol_match(self, input_symbol: str, candidate_symbol: str) -> bool:
        source = re.sub(r"\s+", "", str(input_symbol or "").strip().upper().replace(".", "-"))
        target = re.sub(r"\s+", "", str(candidate_symbol or "").strip().upper().replace(".", "-"))
        if not source or not target:
            return False
        if source == target:
            return True
        if len(target) < len(source):
            return False
        if len(source) >= 2 and not target.startswith(source[:2]):
            return False
        return target.startswith(source) and (len(target) - len(source) <= 2)

    async def _symbol_repair_suggestions(self, token: str, limit: int = 5) -> list[dict[str, Any]]:
        results = await search_tickers(token, limit=max(3, limit))
        if not results and re.fullmatch(r"[A-Z]{2,5}", token):
            results = await search_tickers(f"{token} stock", limit=max(3, limit))

        suggestions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in results:
            ticker = str(getattr(item, "ticker", "") or "").upper().strip()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            suggestions.append(
                {
                    "ticker": ticker,
                    "name": str(getattr(item, "name", "") or "").strip(),
                    "exchange": str(getattr(item, "exchange", "") or "").strip() or None,
                }
            )
            if len(suggestions) >= limit:
                break
        return suggestions

    async def _resolve_symbol_repair(self, symbol: str) -> tuple[str | None, list[dict[str, Any]], str]:
        token = str(symbol or "").strip().upper().replace(".", "-")
        if not token:
            return None, [], "missing_symbol"
        if await self._symbol_quality_ok(token):
            return token, [], "valid_symbol"

        if re.fullmatch(r"[A-Z]{2,5}", token):
            try:
                provider_guess = str(resolve_symbol_input(f"{token} stock") or "").strip().upper().replace(".", "-")
            except Exception:
                provider_guess = ""
            if (
                provider_guess
                and provider_guess != token
                and self._is_confident_symbol_match(token, provider_guess)
                and await self._symbol_quality_ok(provider_guess)
            ):
                return provider_guess, [{"ticker": provider_guess, "name": "", "exchange": None}], "provider_guess"

        suggestions = await self._symbol_repair_suggestions(token, limit=5)
        if len(suggestions) == 1:
            candidate = str(suggestions[0].get("ticker") or "").strip().upper()
            if candidate and self._is_confident_symbol_match(token, candidate) and await self._symbol_quality_ok(candidate):
                return candidate, suggestions, "single_suggestion"

        return None, suggestions, "ambiguous_or_invalid"

    async def _run_startup_migrations(self) -> None:
        agents = tracker_repo.list_agents(user_id=None)
        for agent in agents:
            status = str(agent.get("status") or "").lower()
            if status == "deleted":
                continue

            agent_id = str(agent.get("id") or "")
            user_id = agent.get("user_id")
            symbol = str(agent.get("symbol") or "").strip().upper()
            if not agent_id:
                continue

            existing_triggers = agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {}
            next_triggers = dict(existing_triggers or {})
            updates: dict[str, Any] = {}
            notes: list[str] = []
            migration_resolution: dict[str, Any] | None = None

            repaired_symbol, suggestions, repair_reason = await self._resolve_symbol_repair(symbol)
            if repaired_symbol and repaired_symbol != symbol:
                updates["symbol"] = repaired_symbol
                migration_resolution = {
                    "input_symbol": symbol,
                    "resolved_symbol": repaired_symbol,
                    "auto_corrected": True,
                    "confidence": "high",
                    "suggestions": suggestions,
                    "reason": repair_reason,
                }
                notes.append(f"Auto-repaired symbol from {symbol} to {repaired_symbol}.")
                next_triggers.pop("symbol_review_needed", None)
                next_triggers.pop("symbol_review_input", None)
                next_triggers.pop("symbol_review_candidates", None)
                next_triggers.pop("symbol_review_checked_at", None)
            elif not repaired_symbol:
                if not bool(next_triggers.get("symbol_review_needed")):
                    next_triggers["symbol_review_needed"] = True
                    next_triggers["symbol_review_input"] = symbol
                    next_triggers["symbol_review_candidates"] = suggestions
                    next_triggers["symbol_review_checked_at"] = datetime.now(timezone.utc).isoformat()
                    notes.append("Symbol requires manual confirmation due to ambiguous/invalid mapping.")
            else:
                # Clean stale review flags when symbol validates.
                had_review_flags = any(
                    key in next_triggers
                    for key in ("symbol_review_needed", "symbol_review_input", "symbol_review_candidates", "symbol_review_checked_at")
                )
                if had_review_flags:
                    next_triggers.pop("symbol_review_needed", None)
                    next_triggers.pop("symbol_review_input", None)
                    next_triggers.pop("symbol_review_candidates", None)
                    next_triggers.pop("symbol_review_checked_at", None)
                    notes.append("Cleared stale symbol-review flags after validation.")

            schedule_mode = str(next_triggers.get("schedule_mode") or "realtime").strip().lower()
            report_mode = str(next_triggers.get("report_mode") or "triggers_only").strip().lower()
            if schedule_mode == "realtime" and report_mode == "triggers_only":
                try:
                    poll_seconds = int(float(next_triggers.get("poll_interval_seconds") or self.settings.tracker_poll_interval_seconds))
                except Exception:
                    poll_seconds = max(30, int(self.settings.tracker_poll_interval_seconds))
                poll_seconds = max(30, min(3600, poll_seconds))
                try:
                    report_seconds = int(float(next_triggers.get("report_interval_seconds") or poll_seconds))
                except Exception:
                    report_seconds = poll_seconds
                report_seconds = max(30, min(86400, report_seconds))
                next_triggers["report_mode"] = "hybrid"
                next_triggers["poll_interval_seconds"] = poll_seconds
                next_triggers["report_interval_seconds"] = report_seconds
                notes.append("Migrated realtime agent from trigger-only to hybrid notification mode.")

            if next_triggers != existing_triggers:
                updates["triggers"] = next_triggers

            if not updates:
                continue

            updated = tracker_repo.update_agent(user_id=user_id, agent_id=agent_id, updates=updates)
            if updated is None:
                continue

            tracker_repo.create_history(
                user_id=user_id,
                agent_id=agent_id,
                event_type="system_update",
                parsed_intent={
                    "intent": "startup_migration",
                    "symbol_resolution": migration_resolution,
                    "changes": list(updates.keys()),
                },
                trigger_snapshot=updated.get("triggers") if isinstance(updated.get("triggers"), dict) else {},
                note=" ".join(notes)[:500],
            )
            await log_agent_activity(
                module="tracker",
                agent_name=str(updated.get("name") or "Tracker Agent"),
                action=f"Startup migration applied for {str(updated.get('symbol') or symbol).upper()}",
                status="success",
                user_id=user_id,
                details={
                    "agent_id": agent_id,
                    "symbol_before": symbol,
                    "symbol_after": str(updated.get("symbol") or symbol).upper(),
                    "updates": updates,
                    "notes": notes,
                    "symbol_resolution": migration_resolution,
                },
            )

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

    def _normalize_notification_style(self, style: Any, event_type: str = "cycle") -> str:
        token = str(style or "").strip().lower()
        if token in {"short", "long"}:
            return token
        if event_type == "report":
            return "long"
        if event_type == "alert":
            return "short"
        return "auto"

    async def _latest_manager_instruction(self, agent: dict[str, Any]) -> str:
        agent_id = str(agent.get("id") or "")
        if not agent_id:
            return ""

        cached = self._instruction_cache.get(agent_id)
        if cached:
            cached_at = cached.get("cached_at")
            if isinstance(cached_at, datetime) and (datetime.now(timezone.utc) - cached_at) < timedelta(minutes=2):
                return str(cached.get("prompt") or "")

        history = tracker_repo.list_history(
            user_id=agent.get("user_id"),
            agent_id=agent_id,
            limit=20,
        )
        latest_prompt = next(
            (
                str(row.get("raw_prompt") or "").strip()
                for row in history
                if str(row.get("event_type") or "") in {"manager_instruction", "create_prompt"} and str(row.get("raw_prompt") or "").strip()
            ),
            "",
        )
        self._instruction_cache[agent_id] = {"cached_at": datetime.now(timezone.utc), "prompt": latest_prompt}
        return latest_prompt

    def _extract_prompt_source_intent(self, manager_prompt: str) -> dict[str, Any]:
        lower = f" {str(manager_prompt or '').lower()} "
        aliases: dict[str, tuple[str, ...]] = {
            "reddit": (" reddit ", " r/", " at reddit"),
            "x": (" twitter ", " x ", " from x", " on x", " at x", " at twitter"),
            "perplexity": (" perplexity ",),
            "prediction_markets": (" prediction market", " prediction markets", " polymarket", " kalshi", " trading market"),
            "deep": (" deep research", " deep-research", " browserbase", " at deep research"),
        }

        mentioned: list[str] = []
        for source, needles in aliases.items():
            if any(needle in lower for needle in needles):
                mentioned.append(source)
        mentioned = list(dict.fromkeys(mentioned))

        exclusive_markers = (" only ", " just ", " strictly ", " exclusively ")
        exclusive = any(marker in lower for marker in exclusive_markers)
        locked_sources: list[str] = []

        if exclusive and mentioned:
            locked_sources = list(mentioned)
        else:
            directional_patterns: dict[str, tuple[str, ...]] = {
                "reddit": (" from reddit", " on reddit", " at reddit", " via reddit", " reddit sentiment"),
                "x": (" from x", " on x", " at x", " via x", " from twitter", " on twitter", " at twitter", " twitter sentiment"),
                "perplexity": (" from perplexity", " at perplexity", " via perplexity", " perplexity search"),
                "prediction_markets": (" from prediction market", " at prediction market", " from prediction markets", " from polymarket", " from kalshi", " from trading market"),
                "deep": (" from deep research", " at deep research", " via deep research", " from browserbase"),
            }
            for source, patterns in directional_patterns.items():
                if any(pattern in lower for pattern in patterns):
                    locked_sources.append(source)
            locked_sources = list(dict.fromkeys([token for token in locked_sources if token in mentioned]))

        return {
            "mentioned_sources": mentioned,
            "locked_sources": locked_sources,
            "lock": bool(locked_sources),
        }

    async def _resolve_runtime_tooling(
        self,
        *,
        agent: dict[str, Any],
        tooling: dict[str, Any],
        metric: Any,
        price_change: float,
        volume_ratio: float,
    ) -> dict[str, Any]:
        merged = dict(tooling)
        tool_mode = str(tooling.get("tool_mode") or "auto").strip().lower()
        if tool_mode != "auto":
            merged["notification_style"] = self._normalize_notification_style(
                tooling.get("notification_style"),
                "cycle",
            )
            return merged

        manager_prompt = await self._latest_manager_instruction(agent)
        cache_key = f"{str(agent.get('id') or '')}:{manager_prompt.strip().lower()}"
        cached = self._tool_plan_cache.get(cache_key)
        plan: dict[str, Any] | None = None
        if cached:
            cached_at = cached.get("cached_at")
            if isinstance(cached_at, datetime) and (datetime.now(timezone.utc) - cached_at) < timedelta(minutes=5):
                maybe_plan = cached.get("plan")
                if isinstance(maybe_plan, dict):
                    plan = dict(maybe_plan)

        if plan is None:
            plan = await decide_tracker_runtime_plan(
                self.settings,
                manager_prompt=manager_prompt,
                available_tools=["price", "volume", "sentiment", "news", "prediction_markets", "deep_research", "simulation"],
                available_sources=list(tooling.get("research_sources") or ["perplexity", "x", "reddit", "prediction_markets", "deep"]),
                market_state={
                    "ticker": str(metric.ticker),
                    "price": float(metric.price or 0.0),
                    "change_percent": float(price_change),
                    "volume_ratio": float(volume_ratio),
                },
                event_hint="cycle",
            )
            self._tool_plan_cache[cache_key] = {"cached_at": datetime.now(timezone.utc), "plan": plan}

        chosen_tools = plan.get("tools") if isinstance(plan.get("tools"), list) else []
        chosen_tools = [str(item).strip().lower() for item in chosen_tools if str(item).strip()]

        chosen_sources = plan.get("research_sources") if isinstance(plan.get("research_sources"), list) else []
        chosen_sources = [str(item).strip().lower() for item in chosen_sources if str(item).strip()]
        lower_prompt = f" {manager_prompt.lower()} "
        source_intent = self._extract_prompt_source_intent(manager_prompt)
        mentioned_sources = [str(item).strip().lower() for item in (source_intent.get("mentioned_sources") or []) if str(item).strip()]
        locked_sources = [str(item).strip().lower() for item in (source_intent.get("locked_sources") or []) if str(item).strip()]
        broad_source_markers = {
            " all sources",
            " cross-source",
            " cross source",
            " combine sources",
            " blend sources",
            " multi-source",
        }
        source_specific_prompt = bool(mentioned_sources) and not any(token in lower_prompt for token in broad_source_markers)

        trigger_sources = [str(item).strip().lower() for item in (tooling.get("research_sources") or []) if str(item).strip()]
        source_lock_from_trigger = bool(tooling.get("research_source_lock"))

        effective_sources: list[str] = []
        if source_lock_from_trigger and trigger_sources:
            effective_sources = list(dict.fromkeys(trigger_sources))
        elif locked_sources:
            effective_sources = list(dict.fromkeys(locked_sources))
        elif source_specific_prompt:
            chosen_specific = [token for token in chosen_sources if token in set(mentioned_sources)]
            effective_sources = list(dict.fromkeys(chosen_specific or mentioned_sources))
        else:
            effective_sources = list(dict.fromkeys(chosen_sources + mentioned_sources))
            if not effective_sources and trigger_sources:
                effective_sources = list(dict.fromkeys(trigger_sources))

        if effective_sources:
            merged["research_sources"] = effective_sources
        merged["research_source_lock"] = bool(source_lock_from_trigger or bool(locked_sources) or source_specific_prompt)

        # Ensure tools match the selected source(s) even if the LLM omits them.
        effective_source_set = set(effective_sources)
        if any(source in effective_source_set for source in {"reddit", "x"}):
            if "sentiment" not in chosen_tools:
                chosen_tools.append("sentiment")
        if "perplexity" in effective_source_set:
            if "sentiment" not in chosen_tools:
                chosen_tools.append("sentiment")
            if "news" not in chosen_tools:
                chosen_tools.append("news")
        if "prediction_markets" in effective_source_set and "prediction_markets" not in chosen_tools:
            chosen_tools.append("prediction_markets")
        if "deep" in effective_source_set and "deep_research" not in chosen_tools:
            chosen_tools.append("deep_research")

        if " sentiment " in lower_prompt and "sentiment" not in chosen_tools:
            chosen_tools.append("sentiment")
        news_requested = any(
            token in lower_prompt
            for token in {" news ", " catalyst ", " catalysts ", " investigate ", " search ", " what happened ", " why moved "}
        )
        if news_requested and "news" not in chosen_tools:
            chosen_tools.append("news")
        if (
            bool(merged.get("research_source_lock"))
            and "news" in chosen_tools
            and "perplexity" not in effective_source_set
            and not news_requested
        ):
            chosen_tools = [token for token in chosen_tools if token != "news"]

        if not chosen_tools:
            chosen_tools = list(tooling.get("tools") or ["price", "volume"])
        merged["tools"] = list(dict.fromkeys(chosen_tools))

        merged["simulate_on_alert"] = bool(tooling.get("simulate_on_alert")) or bool(plan.get("simulate_on_alert"))
        merged["use_prediction_markets"] = "prediction_markets" in set(merged.get("tools") or [])
        merged["notification_style"] = self._normalize_notification_style(plan.get("notification_style"), "cycle")
        merged["manager_prompt"] = manager_prompt
        merged["runtime_tool_plan"] = plan
        return merged

    def _compose_notification_body(
        self,
        *,
        ticker: str,
        event_type: str,
        reasons: list[str],
        metric: Any,
        price_change: float,
        aggregate_sentiment: float,
        research_sources: list[str],
        narrative: str,
        simulation_context: dict[str, Any] | None,
        style: str,
    ) -> str:
        reason_line = "; ".join(reasons) if reasons else "monitoring update"
        normalized_style = self._normalize_notification_style(style, event_type)
        source_line = ", ".join(research_sources) if research_sources else "auto"
        event_label = "report" if event_type == "report" else "alert"
        if normalized_style == "short":
            simulation_line = ""
            if simulation_context:
                simulation_line = (
                    f" Sim outlook: {simulation_context['expected_return_pct']:+.2f}% exp, "
                    f"{simulation_context['downside_prob_3pct'] * 100:.0f}% chance of >3% downside."
                )
            lines = [
                f"{ticker} {event_label}: ${float(metric.price):.2f} ({price_change:+.2f}%), sentiment {aggregate_sentiment:+.2f}.",
                f"Why now: {reason_line}.",
                f"Sources: {source_line}.",
            ]
            if simulation_line:
                lines.append(simulation_line)
            return " ".join(lines)[:480]

        sim_long = ""
        if simulation_context:
            sim_long = (
                f"Simulation: expected {simulation_context['expected_return_pct']:+.2f}% "
                f"with {simulation_context['downside_prob_3pct'] * 100:.0f}% probability of >3% downside."
            )
        body_lines = [
            f"{ticker} {event_type.upper()} broker update",
            f"Price: ${float(metric.price):.2f} ({price_change:+.2f}%)",
            f"Sentiment: {aggregate_sentiment:+.2f}",
            f"Sources: {source_line}",
            f"What triggered this: {reason_line}",
        ]
        if sim_long:
            body_lines.append(sim_long)
        body_lines.extend(
            [
                f"Broker note: {narrative}",
                "Risk note: monitoring signal only, not financial advice.",
            ]
        )
        body = "\n".join([line for line in body_lines if line]).strip()
        return body[:1300]

    async def _synthesize_narrative(
        self,
        ticker: str,
        raw_context: str,
        *,
        style: str = "auto",
        manager_prompt: str = "",
        event_type: str = "cycle",
    ) -> str:
        style_token = self._normalize_notification_style(style, event_type)
        if style_token == "short":
            style_instruction = (
                "Return exactly 2-3 natural sentences like a personal broker text update. "
                "Include signal, risk, and one next-watch item."
            )
        elif style_token == "long":
            style_instruction = (
                "Return a readable broker-style brief with labeled sections: Setup, What Changed, Risk, Next Step."
            )
        else:
            style_instruction = (
                "Return 3-5 clear sentences in a personal broker tone, concise and human, no jargon dumping."
            )
        prompt = (
            "You are a personal broker assistant writing client updates for a beginner investor. "
            f"{style_instruction} "
            "Write in plain English. Be direct. Avoid hype and avoid generic filler. "
            f"Ticker: {ticker}. "
            f"Manager instruction: {manager_prompt or 'none provided'}. "
            f"Context: {raw_context}"
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
            f"{ticker} update: signals are mixed and require confirmation. "
            "Main risk is fast sentiment reversal on new headlines. "
            "Next step: watch price reaction and volume follow-through before acting."
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

    def _parse_timestamp(self, raw: Any) -> datetime | None:
        if not raw:
            return None
        try:
            stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            return stamp
        except Exception:
            return None

    async def _get_notification_preferences(self, user_id: str | None) -> dict[str, Any]:
        if not user_id:
            return {}

        cached = self._notification_pref_cache.get(user_id)
        if cached:
            cached_at = cached.get("cached_at")
            if isinstance(cached_at, datetime) and (datetime.now(timezone.utc) - cached_at) < timedelta(minutes=2):
                return dict(cached.get("data") or {})

        client = get_supabase()
        if client is None:
            return {}

        try:
            row = (
                client.table("notification_preferences")
                .select("*")
                .eq("user_id", user_id)
                .single()
                .execute()
                .data
            ) or {}
            data = {
                "phone_number": str(row.get("phone_number") or "").strip() or None,
                "email": str(row.get("email") or "").strip() or None,
                "preferred_channel": str(row.get("preferred_channel") or "push").strip().lower() or "push",
                "alert_frequency": str(row.get("alert_frequency") or "realtime").strip().lower() or "realtime",
                "price_alerts": bool(row.get("price_alerts", True)),
                "volume_alerts": bool(row.get("volume_alerts", True)),
                "simulation_summary": bool(row.get("simulation_summary", True)),
                "quiet_start": str(row.get("quiet_start") or "22:00:00"),
                "quiet_end": str(row.get("quiet_end") or "07:00:00"),
            }
            self._notification_pref_cache[user_id] = {"cached_at": datetime.now(timezone.utc), "data": data}
            return data
        except Exception:
            return {}

    def _clock_minutes(self, value: Any, default: str) -> int:
        token = str(value or default).strip()
        parts = token.split(":")
        if len(parts) < 2:
            parts = default.split(":")
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except Exception:
            hour, minute = 0, 0
        hour = max(0, min(23, hour))
        minute = max(0, min(59, minute))
        return (hour * 60) + minute

    def _is_quiet_hours(self, notification_prefs: dict[str, Any], now: datetime) -> bool:
        if not notification_prefs:
            return False
        start = self._clock_minutes(notification_prefs.get("quiet_start"), "22:00:00")
        end = self._clock_minutes(notification_prefs.get("quiet_end"), "07:00:00")
        current = (now.hour * 60) + now.minute
        if start == end:
            return False
        if start < end:
            return start <= current < end
        return current >= start or current < end

    def _notification_allowed(self, agent: dict[str, Any], notification_prefs: dict[str, Any], now: datetime) -> tuple[bool, str | None]:
        if not notification_prefs:
            return True, None

        frequency = str(notification_prefs.get("alert_frequency") or "realtime").strip().lower()
        min_interval_seconds = {
            "realtime": 0,
            "hourly": 3600,
            "daily": 86400,
        }.get(frequency, 0)
        if min_interval_seconds <= 0:
            return True, None

        last_notified = self._parse_timestamp(agent.get("last_alert_at"))
        if last_notified and (now - last_notified) < timedelta(seconds=min_interval_seconds):
            return False, f"frequency_{frequency}"
        return True, None

    def _resolve_channels(self, tooling: dict[str, Any], notification_prefs: dict[str, Any]) -> list[str]:
        channels = [str(item).strip().lower() for item in (tooling.get("notify_channels") or []) if str(item).strip()]
        if bool(tooling.get("notify_channels_from_trigger")) and channels:
            return channels

        preferred = str(notification_prefs.get("preferred_channel") or "").strip().lower()
        if preferred == "sms":
            return ["twilio"]
        if preferred in {"push", "email"}:
            # Email delivery is not wired yet, so "email" falls back to push integration.
            return ["poke"]
        return channels or ["twilio", "poke"]

    def _agent_tooling_config(self, agent: dict[str, Any]) -> dict[str, Any]:
        triggers = agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {}
        triggers = triggers or {}

        allowed_tools = {
            "price",
            "volume",
            "sentiment",
            "news",
            "prediction_markets",
            "deep_research",
            "simulation",
        }
        allowed_sources = {"perplexity", "x", "reddit", "prediction_markets", "deep"}
        allowed_channels = {"twilio", "poke"}
        tool_mode = str(triggers.get("tool_mode") or "auto").strip().lower()
        if tool_mode not in {"auto", "manual"}:
            tool_mode = "auto"

        tools: list[str] = []
        raw_tools = triggers.get("tools")
        if tool_mode == "manual" and isinstance(raw_tools, list):
            for item in raw_tools:
                token = str(item).strip().lower()
                if token in allowed_tools and token not in tools:
                    tools.append(token)

        if not tools:
            tools = ["price", "volume"]
            if any(key in triggers for key in {"sentiment_bearish_threshold", "sentiment_bullish_threshold", "x_bearish_threshold"}):
                tools.extend(["sentiment", "news"])
            if bool(agent.get("auto_simulate")) or bool(triggers.get("simulate_on_alert")):
                tools.append("simulation")

        research_sources: list[str] = []
        raw_sources = triggers.get("research_sources")
        if isinstance(raw_sources, list):
            for item in raw_sources:
                token = str(item).strip().lower()
                if token in allowed_sources and token not in research_sources:
                    research_sources.append(token)

        if not research_sources and any(token in tools for token in {"sentiment", "news", "prediction_markets", "deep_research"}):
            research_sources = ["perplexity", "x", "reddit"]

        notify_channels: list[str] = []
        notify_channels_from_trigger = False
        raw_channels = triggers.get("notify_channels")
        if isinstance(raw_channels, list):
            for item in raw_channels:
                token = str(item).strip().lower()
                if token in allowed_channels and token not in notify_channels:
                    notify_channels.append(token)
        if notify_channels:
            notify_channels_from_trigger = True
        if not notify_channels:
            notify_channels = ["twilio", "poke"]

        poll_interval = triggers.get("poll_interval_seconds")
        try:
            poll_interval_seconds = int(float(poll_interval))
        except Exception:
            poll_interval_seconds = self.settings.tracker_poll_interval_seconds
        poll_interval_seconds = max(30, min(3600, poll_interval_seconds))

        report_interval = triggers.get("report_interval_seconds")
        try:
            report_interval_seconds = int(float(report_interval))
        except Exception:
            report_interval_seconds = poll_interval_seconds
        report_interval_seconds = max(30, min(86400, report_interval_seconds))

        schedule_mode = str(triggers.get("schedule_mode") or "realtime").strip().lower()
        if schedule_mode not in {"realtime", "hourly", "daily", "custom"}:
            schedule_mode = "realtime"
        custom_time_enabled = bool(triggers.get("custom_time_enabled"))
        if schedule_mode == "realtime":
            poll_interval_seconds = 120
            report_interval_seconds = 120
        elif schedule_mode == "hourly":
            poll_interval_seconds = 3600
            report_interval_seconds = 3600
        elif schedule_mode == "daily":
            poll_interval_seconds = 86400
            report_interval_seconds = 86400
        elif schedule_mode == "custom" and custom_time_enabled:
            poll_interval_seconds = 86400
            report_interval_seconds = 86400

        daily_run_time_raw = str(triggers.get("daily_run_time") or "").strip()
        has_valid_daily_time = bool(re.match(r"^\d{1,2}:\d{2}$", daily_run_time_raw))
        if schedule_mode == "daily":
            daily_run_time = daily_run_time_raw if has_valid_daily_time else "09:30"
        elif schedule_mode == "custom" and custom_time_enabled:
            daily_run_time = daily_run_time_raw if has_valid_daily_time else "09:30"
        else:
            daily_run_time = ""

        timezone_name = str(triggers.get("timezone") or "America/New_York").strip() or "America/New_York"
        try:
            ZoneInfo(timezone_name)
        except Exception:
            timezone_name = "America/New_York"

        start_at = self._parse_timestamp(triggers.get("start_at"))

        baseline_mode = str(triggers.get("baseline_mode") or "prev_close").strip().lower()
        if baseline_mode not in {"prev_close", "last_check", "last_alert", "session_open"}:
            baseline_mode = "prev_close"

        report_mode_raw = str(triggers.get("report_mode") or "hybrid").strip().lower()
        report_mode_aliases = {
            "alerts": "triggers_only",
            "alerts_only": "triggers_only",
            "trigger_only": "triggers_only",
            "report": "periodic",
            "reports": "periodic",
            "mixed": "hybrid",
        }
        report_mode = report_mode_aliases.get(report_mode_raw, report_mode_raw)
        if report_mode not in {"triggers_only", "periodic", "hybrid"}:
            report_mode = "hybrid"

        timeframe = str(triggers.get("research_timeframe") or "7d")
        notify_phone = str(triggers.get("notify_phone") or "").strip() or None
        simulate_on_alert = bool(triggers.get("simulate_on_alert")) or bool(agent.get("auto_simulate"))
        notification_style = self._normalize_notification_style(triggers.get("notification_style"), "cycle")

        return {
            "tools": tools,
            "research_sources": research_sources,
            "research_source_lock": bool(triggers.get("research_source_lock", False)),
            "notify_channels": notify_channels,
            "notify_channels_from_trigger": notify_channels_from_trigger,
            "poll_interval_seconds": poll_interval_seconds,
            "report_interval_seconds": report_interval_seconds,
            "report_mode": report_mode,
            "research_timeframe": timeframe,
            "notify_phone": notify_phone,
            "simulate_on_alert": simulate_on_alert,
            "use_prediction_markets": "prediction_markets" in tools,
            "tool_mode": tool_mode,
            "baseline_mode": baseline_mode,
            "schedule_mode": schedule_mode,
            "daily_run_time": daily_run_time,
            "custom_time_enabled": custom_time_enabled,
            "timezone": timezone_name,
            "start_at": start_at.isoformat() if start_at else None,
            "notification_style": notification_style,
        }

    def _agent_poll_due(self, agent: dict[str, Any], tooling: dict[str, Any], now: datetime) -> bool:
        start_at = self._parse_timestamp(tooling.get("start_at"))
        if start_at and now < start_at:
            return False

        schedule_mode = str(tooling.get("schedule_mode") or "realtime").strip().lower()
        custom_time_enabled = bool(tooling.get("custom_time_enabled"))
        if schedule_mode == "daily" or (schedule_mode == "custom" and custom_time_enabled):
            return self._clock_schedule_due(
                last_stamp=agent.get("last_checked_at"),
                tooling=tooling,
                now=now,
                start_at=start_at,
            )

        poll_interval_seconds = int(tooling.get("poll_interval_seconds") or self.settings.tracker_poll_interval_seconds)
        raw = agent.get("last_checked_at")
        if not raw:
            return True
        try:
            stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
        except Exception:
            return True
        return (now - stamp) >= timedelta(seconds=max(30, poll_interval_seconds))

    def _clock_schedule_due(
        self,
        *,
        last_stamp: Any,
        tooling: dict[str, Any],
        now: datetime,
        start_at: datetime | None,
    ) -> bool:
        daily_run_time = str(tooling.get("daily_run_time") or "").strip()
        if not re.match(r"^\d{1,2}:\d{2}$", daily_run_time):
            return False
        timezone_name = str(tooling.get("timezone") or "America/New_York").strip() or "America/New_York"
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = timezone.utc

        hour = int(daily_run_time.split(":")[0])
        minute = int(daily_run_time.split(":")[1])
        now_local = now.astimezone(tz)
        run_today_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        run_today_utc = run_today_local.astimezone(timezone.utc)

        if start_at and run_today_utc < start_at:
            run_today_utc = run_today_utc + timedelta(days=1)
        if now < run_today_utc:
            return False

        prior = self._parse_timestamp(last_stamp)
        if prior is None:
            return True
        return prior < run_today_utc

    def _agent_report_due(
        self,
        agent: dict[str, Any],
        tooling: dict[str, Any],
        report_interval_seconds: int,
        now: datetime,
    ) -> bool:
        schedule_mode = str(tooling.get("schedule_mode") or "realtime").strip().lower()
        custom_time_enabled = bool(tooling.get("custom_time_enabled"))
        if schedule_mode == "daily" or (schedule_mode == "custom" and custom_time_enabled):
            return self._clock_schedule_due(
                last_stamp=agent.get("last_alert_at"),
                tooling=tooling,
                now=now,
                start_at=self._parse_timestamp(tooling.get("start_at")),
            )

        raw = agent.get("last_alert_at")
        if not raw:
            return True
        try:
            stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
        except Exception:
            return True
        return (now - stamp) >= timedelta(seconds=max(30, report_interval_seconds))

    def _resolve_agent_price_change(
        self,
        *,
        agent: dict[str, Any],
        metric: Any,
        observed_poll_change: float,
        tooling: dict[str, Any],
    ) -> tuple[float, str, float | None]:
        baseline_mode = str(tooling.get("baseline_mode") or "prev_close")
        current_price = float(metric.price or 0.0)
        if current_price <= 0:
            return observed_poll_change, "prev_close", None

        if baseline_mode in {"prev_close", "session_open"}:
            # MarketMetric.change_percent is provider-based move vs prior close/session open baseline.
            try:
                return float(metric.change_percent or 0.0), baseline_mode, None
            except Exception:
                return observed_poll_change, baseline_mode, None

        if baseline_mode == "last_check":
            baseline = agent.get("last_price")
            try:
                base = float(baseline)
            except Exception:
                base = 0.0
            if base > 0:
                return ((current_price - base) / base) * 100.0, baseline_mode, base
            return observed_poll_change, baseline_mode, None

        if baseline_mode == "last_alert":
            triggers = agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {}
            baseline = triggers.get("last_alert_price") if isinstance(triggers, dict) else None
            try:
                base = float(baseline)
            except Exception:
                base = 0.0
            if base > 0:
                return ((current_price - base) / base) * 100.0, baseline_mode, base
            return observed_poll_change, baseline_mode, None

        return observed_poll_change, "prev_close", None

    async def _update_agent_thesis(
        self,
        *,
        agent: dict[str, Any],
        metric: Any,
        event_type: str,
        reasons: list[str],
        narrative: str,
        price_change: float,
        research: dict[str, Any],
        simulation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        aggregate_sentiment = float(research.get("aggregate_sentiment", 0.0) or 0.0)
        stance_score = max(-1.0, min(1.0, (aggregate_sentiment * 0.7) + (price_change / 15.0)))
        confidence = max(0.05, min(0.98, 0.35 + (abs(aggregate_sentiment) * 0.35) + min(0.28, abs(price_change) / 12.0)))
        thesis = {
            "drivers": reasons[:5],
            "aggregate_sentiment": round(aggregate_sentiment, 4),
            "price_change_pct": round(price_change, 4),
            "recommendation": str(research.get("recommendation") or "hold"),
            "simulation": simulation_context or {},
            "narrative_excerpt": narrative[:280],
        }
        summary = (
            f"Stance {stance_score:+.2f} ({'bullish' if stance_score >= 0 else 'bearish'}) "
            f"confidence {confidence:.2f}; drivers: {', '.join(reasons[:3]) or 'none'}."
        )
        return tracker_repo.upsert_thesis(
            user_id=agent.get("user_id"),
            agent_id=str(agent.get("id")),
            symbol=str(metric.ticker),
            stance_score=stance_score,
            confidence=confidence,
            thesis=thesis,
            summary=summary,
            last_event_type=event_type,
        )

    async def _resolve_notification_phone(
        self,
        user_id: str | None,
        agent: dict[str, Any],
        tooling: dict[str, Any],
        notification_prefs: dict[str, Any] | None = None,
    ) -> str | None:
        from_triggers = str(tooling.get("notify_phone") or "").strip()
        if from_triggers:
            return from_triggers
        from_prefs = str((notification_prefs or {}).get("phone_number") or "").strip()
        if from_prefs:
            return from_prefs
        if not user_id:
            return self.settings.twilio_default_to_number or None

        client = get_supabase()
        if client is None:
            return self.settings.twilio_default_to_number or None
        try:
            profile = (
                client.table("profiles")
                .select("phone_number,phone,sms_number")
                .eq("id", user_id)
                .single()
                .execute()
                .data
            ) or {}
            for key in ("phone_number", "phone", "sms_number"):
                value = str(profile.get(key) or "").strip()
                if value:
                    return value
        except Exception:
            return self.settings.twilio_default_to_number or None
        return self.settings.twilio_default_to_number or None

    def _quick_simulation_context(self, price_change: float, aggregate_sentiment: float, base_price: float) -> dict[str, Any]:
        horizon_steps = 40
        simulations = 120
        drift = float((aggregate_sentiment * 0.0007) + (price_change * 0.0003))
        sigma = float(max(0.004, min(0.06, 0.012 + abs(price_change) * 0.0012)))
        shocks = self._rng.normal(loc=drift, scale=sigma, size=(simulations, horizon_steps))
        cumulative = shocks.sum(axis=1)
        projected_prices = base_price * np.exp(cumulative)
        projected_returns_pct = ((projected_prices - base_price) / max(1e-6, base_price)) * 100

        expected_return = float(np.mean(projected_returns_pct))
        downside_prob = float(np.mean(projected_returns_pct <= -3.0))
        upside_prob = float(np.mean(projected_returns_pct >= 3.0))
        worst_case = float(np.percentile(projected_returns_pct, 5))
        best_case = float(np.percentile(projected_returns_pct, 95))
        return {
            "expected_return_pct": round(expected_return, 2),
            "upside_prob_3pct": round(upside_prob, 3),
            "downside_prob_3pct": round(downside_prob, 3),
            "p05_return_pct": round(worst_case, 2),
            "p95_return_pct": round(best_case, 2),
            "horizon_steps": horizon_steps,
            "paths": simulations,
        }

    async def _maybe_launch_simulation_session(self, metric: Any, user_id: str | None) -> str | None:
        if self.orchestrator is None:
            return None
        try:
            from app.schemas import SimulationStartRequest

            req = SimulationStartRequest(
                ticker=str(metric.ticker),
                duration_seconds=90,
                initial_price=float(metric.price),
                starting_cash=100_000,
                volatility=0.02 + min(0.02, abs(float(metric.change_percent or 0.0)) / 100),
                user_id=user_id,
                inference_runtime="modal",
                agents=[],
            )
            state = await self.orchestrator.start(req)
            return state.session_id
        except Exception:
            return None

    async def _get_research_snapshot(
        self,
        ticker: str,
        timeframe: str = "7d",
        sources: list[str] | None = None,
        tools: list[str] | None = None,
        include_prediction_markets: bool = False,
        strict_sources: bool = False,
        manager_prompt: str = "",
        market_state: dict[str, Any] | None = None,
        event_hint: str = "cycle",
    ) -> dict[str, Any]:
        source_key = ",".join(sorted([str(item).lower() for item in (sources or [])])) or "default"
        tool_key = ",".join(sorted([str(item).lower() for item in (tools or [])])) or "default"
        key = f"{ticker.upper()}:{timeframe}:{source_key}:{tool_key}:{int(include_prediction_markets)}:{int(strict_sources)}"
        cached = self._research_cache.get(key)
        if cached:
            at = cached.get("cached_at")
            if isinstance(at, datetime) and (datetime.now(timezone.utc) - at) < timedelta(minutes=5):
                return dict(cached.get("data") or {})
        mcp_payload = await collect_tracker_research_via_mcp(
            self.settings,
            ticker=ticker.upper(),
            manager_prompt=manager_prompt,
            tools=list(tools or []),
            sources=list(sources or []),
            market_state=market_state or {},
            timeframe=timeframe,
            event_hint=event_hint,
        )
        if isinstance(mcp_payload, dict) and mcp_payload:
            payload = {
                "aggregate_sentiment": float(mcp_payload.get("aggregate_sentiment", 0.0) or 0.0),
                "recommendation": str(mcp_payload.get("recommendation") or "hold"),
                "breakdown": mcp_payload.get("breakdown") if isinstance(mcp_payload.get("breakdown"), dict) else {},
                "breakdown_summaries": (
                    mcp_payload.get("breakdown_summaries")
                    if isinstance(mcp_payload.get("breakdown_summaries"), dict)
                    else {}
                ),
                "prediction_markets": (
                    mcp_payload.get("prediction_markets")
                    if isinstance(mcp_payload.get("prediction_markets"), list)
                    else []
                ),
                "investigation": str(mcp_payload.get("investigation") or ""),
                "mcp_debug": mcp_payload.get("mcp_debug") if isinstance(mcp_payload.get("mcp_debug"), dict) else {},
            }
            self._research_cache[key] = {"cached_at": datetime.now(timezone.utc), "data": payload}
            try:
                print(
                    f"[tracker-mcp-debug][{ticker.upper()}] "
                    f"{json.dumps(payload.get('mcp_debug') or {}, ensure_ascii=True)[:6000]}",
                    flush=True,
                )
            except Exception:
                pass
            return payload
        try:
            data = await run_research_with_source_selection(
                ResearchRequest(
                    ticker=ticker.upper(),
                    timeframe=timeframe,
                    include_prediction_markets=include_prediction_markets,
                ),
                self.settings,
                sources=sources,
                strict_sources=strict_sources,
            )
            payload = {
                "aggregate_sentiment": data.aggregate_sentiment,
                "recommendation": data.recommendation,
                "breakdown": {item.source: item.score for item in data.source_breakdown},
                "breakdown_summaries": {item.source: str(item.summary or "")[:320] for item in data.source_breakdown},
                "prediction_markets": list(data.prediction_markets or []),
                "investigation": "",
                "mcp_debug": {},
            }
            self._research_cache[key] = {"cached_at": datetime.now(timezone.utc), "data": payload}
            return payload
        except Exception:
            return {
                "aggregate_sentiment": 0.0,
                "recommendation": "hold",
                "breakdown": {},
                "breakdown_summaries": {},
                "prediction_markets": [],
                "investigation": "",
                "mcp_debug": {},
            }

    async def _get_deep_research_snapshot(self, ticker: str) -> dict[str, Any]:
        key = f"deep:{ticker.upper()}"
        cached = self._research_cache.get(key)
        if cached:
            at = cached.get("cached_at")
            if isinstance(at, datetime) and (datetime.now(timezone.utc) - at) < timedelta(minutes=20):
                return dict(cached.get("data") or {})
        try:
            payload = await run_deep_research(ticker.upper(), self.settings)
            data = payload if isinstance(payload, dict) else {}
            self._research_cache[key] = {"cached_at": datetime.now(timezone.utc), "data": data}
            return data
        except Exception:
            return {}

    async def _evaluate_agent(
        self,
        agent: dict[str, Any],
        metric: Any,
        price_change: float,
        volume_ratio: float,
    ) -> dict[str, Any] | None:
        triggers = agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {}
        triggers = triggers or {}
        now = datetime.now(timezone.utc)

        def _as_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except Exception:
                return default

        tooling = self._agent_tooling_config(agent)
        tooling = await self._resolve_runtime_tooling(
            agent=agent,
            tooling=tooling,
            metric=metric,
            price_change=price_change,
            volume_ratio=volume_ratio,
        )
        notification_prefs = await self._get_notification_preferences(agent.get("user_id"))
        price_alerts_enabled = bool(notification_prefs.get("price_alerts", True))
        volume_alerts_enabled = bool(notification_prefs.get("volume_alerts", True))
        simulation_summary_enabled = bool(notification_prefs.get("simulation_summary", True))
        report_mode = str(tooling.get("report_mode") or "hybrid")
        report_interval_seconds = int(tooling.get("report_interval_seconds") or tooling.get("poll_interval_seconds") or 120)
        scheduled_report_due = report_mode in {"periodic", "hybrid"} and self._agent_report_due(
            agent,
            tooling,
            report_interval_seconds,
            now,
        )
        effective_price_change, baseline_mode, baseline_price = self._resolve_agent_price_change(
            agent=agent,
            metric=metric,
            observed_poll_change=price_change,
            tooling=tooling,
        )
        tools = set(tooling.get("tools") or [])
        requires_research = bool(
            tools.intersection({"sentiment", "news", "prediction_markets", "deep_research"})
            or any(key in triggers for key in {"sentiment_bearish_threshold", "sentiment_bullish_threshold", "x_bearish_threshold"})
        )

        research = {
            "aggregate_sentiment": 0.0,
            "recommendation": "hold",
            "breakdown": {},
            "breakdown_summaries": {},
            "prediction_markets": [],
        }
        if requires_research:
            research = await self._get_research_snapshot(
                metric.ticker,
                timeframe=str(tooling.get("research_timeframe") or "7d"),
                sources=list(tooling.get("research_sources") or []),
                tools=list(tools),
                include_prediction_markets=bool(tooling.get("use_prediction_markets")),
                strict_sources=bool(tooling.get("research_source_lock")),
                manager_prompt=str(tooling.get("manager_prompt") or ""),
                market_state={
                    "ticker": str(metric.ticker),
                    "price": float(metric.price or 0.0),
                    "change_percent": float(effective_price_change),
                    "volume_ratio": float(volume_ratio),
                },
                event_hint="cycle",
            )
        deep_research_payload: dict[str, Any] = {}
        if "deep_research" in tools:
            deep_research_payload = await self._get_deep_research_snapshot(metric.ticker)
            if deep_research_payload:
                research = {**research, "deep_research": deep_research_payload}

        aggregate = float(research.get("aggregate_sentiment", 0.0))
        x_score = float((research.get("breakdown") or {}).get("X API", 0.0))

        bearish_threshold = _as_float(triggers.get("sentiment_bearish_threshold"), -0.25)
        bullish_threshold = _as_float(triggers.get("sentiment_bullish_threshold"), 0.25)
        x_bearish = _as_float(triggers.get("x_bearish_threshold"), -0.25)
        price_threshold = _as_float(triggers.get("price_change_pct"), 2.0)
        volume_spike_threshold = _as_float(triggers.get("volume_spike_ratio"), 1.8)

        trigger_reasons: list[str] = []
        if "price" in tools and price_alerts_enabled and abs(effective_price_change) >= price_threshold:
            trigger_reasons.append(
                f"price move {effective_price_change:.2f}% >= {price_threshold:.2f}% "
                f"(baseline={baseline_mode})"
            )
        volume_spike = volume_ratio >= volume_spike_threshold
        if "volume" in tools and volume_alerts_enabled and volume_spike:
            trigger_reasons.append(f"volume spike exceeded ratio {volume_spike_threshold:.2f}")
        if "sentiment" in tools and aggregate <= bearish_threshold:
            trigger_reasons.append(f"broad sentiment bearish ({aggregate:.2f})")
        if "sentiment" in tools and aggregate >= bullish_threshold:
            trigger_reasons.append(f"broad sentiment bullish ({aggregate:.2f})")
        if "sentiment" in tools and x_score <= x_bearish:
            trigger_reasons.append(f"X sentiment deeply bearish ({x_score:.2f})")
        if "prediction_markets" in tools and (research.get("prediction_markets") or []):
            top_market = (research.get("prediction_markets") or [])[0]
            top_score = float(top_market.get("relevance_score", 0.0) or 0.0)
            if top_score >= 0.45:
                trigger_reasons.append(f"prediction-market signal elevated ({top_score:.2f})")
        if "deep_research" in tools and deep_research_payload and scheduled_report_due:
            trigger_reasons.append("deep-research context refreshed")

        quick_investigation = str(research.get("investigation") or "").strip()
        if (
            "news" in tools
            and not quick_investigation
            and (abs(effective_price_change) >= (price_threshold * 0.75) or volume_spike or scheduled_report_due)
        ):
            quick_investigation = await self._investigate_with_perplexity(
                metric.ticker,
                (
                    f"price_change={effective_price_change:.2f}% volume_ratio={volume_ratio:.2f} "
                    f"sentiment={aggregate:.2f} reasons={'; '.join(trigger_reasons) or 'none'}"
                ),
            )
        if quick_investigation and trigger_reasons and "news catalysts detected" not in trigger_reasons:
            trigger_reasons.append("news catalysts detected")

        await log_agent_activity(
            module="tracker",
            agent_name=str(agent.get("name") or f"{metric.ticker} Associate"),
            action=f"Evaluated multi-factor signals for {metric.ticker}",
            status="running",
            user_id=agent.get("user_id"),
            details={
                "agent_id": agent.get("id"),
                "symbol": metric.ticker,
                "description": "Checked configured tools for this agent across price/volume/research/simulation.",
                "price_change_pct": round(effective_price_change, 3),
                "baseline_mode": baseline_mode,
                "baseline_price": baseline_price,
                "volume_spike": bool(volume_spike),
                "volume_ratio": round(volume_ratio, 3),
                "aggregate_sentiment": round(aggregate, 3),
                "x_sentiment": round(x_score, 3),
                "tools": sorted(tools),
                "research_sources": tooling.get("research_sources") or [],
                "trigger_matches": trigger_reasons,
                "report_mode": report_mode,
                "scheduled_report_due": scheduled_report_due,
                "notification_style": tooling.get("notification_style"),
                "runtime_tool_plan": tooling.get("runtime_tool_plan"),
            },
        )

        if report_mode == "periodic":
            trigger_reasons = []

        if trigger_reasons and self._agent_in_cooldown(agent):
            if not scheduled_report_due:
                tracker_repo.create_run(
                    user_id=agent.get("user_id"),
                    agent_id=str(agent.get("id")),
                    symbol=metric.ticker,
                    run_type="noop",
                    trigger_reasons=trigger_reasons,
                    tools_used=list(tools),
                    research_sources=list(tooling.get("research_sources") or []),
                    market_snapshot={
                        "price": metric.price,
                        "price_change_pct": round(effective_price_change, 4),
                        "baseline_mode": baseline_mode,
                        "baseline_price": baseline_price,
                        "volume": metric.volume,
                        "volume_ratio": round(volume_ratio, 4),
                    },
                    research_snapshot=research,
                    decision={"event_type": "noop", "cooldown": True},
                    note="Trigger met but cooldown suppressed alert.",
                )
                tracker_repo.update_agent(
                    user_id=agent.get("user_id"),
                    agent_id=str(agent.get("id")),
                    updates={"last_checked_at": now.isoformat(), "last_price": float(metric.price)},
                )
                return None
            trigger_reasons = []

        event_type: str | None = None
        reasons: list[str] = []
        if trigger_reasons:
            event_type = "alert"
            reasons = list(trigger_reasons)
        elif scheduled_report_due:
            event_type = "report"
            reasons = [f"scheduled report ({report_interval_seconds}s cadence)"]
            if quick_investigation:
                reasons.append("news context refreshed")

        if event_type is None:
            tracker_repo.create_run(
                user_id=agent.get("user_id"),
                agent_id=str(agent.get("id")),
                symbol=metric.ticker,
                run_type="noop",
                trigger_reasons=[],
                tools_used=list(tools),
                research_sources=list(tooling.get("research_sources") or []),
                market_snapshot={
                    "price": metric.price,
                    "price_change_pct": round(effective_price_change, 4),
                    "baseline_mode": baseline_mode,
                    "baseline_price": baseline_price,
                    "volume": metric.volume,
                    "volume_ratio": round(volume_ratio, 4),
                },
                research_snapshot=research,
                decision={"event_type": "noop"},
                note="No trigger or periodic report due this cycle.",
            )
            await self._update_agent_thesis(
                agent=agent,
                metric=metric,
                event_type="noop",
                reasons=[],
                narrative="No event emitted this cycle.",
                price_change=effective_price_change,
                research=research,
                simulation_context=None,
            )
            tracker_repo.update_agent(
                user_id=agent.get("user_id"),
                agent_id=str(agent.get("id")),
                updates={"last_checked_at": now.isoformat(), "last_price": float(metric.price)},
            )
            return None

        prior_alerts = tracker_repo.list_alerts(
            user_id=agent.get("user_id"),
            agent_id=str(agent.get("id")),
            limit=6,
        )
        prior_context_rows = tracker_repo.list_alert_context(
            user_id=agent.get("user_id"),
            agent_id=str(agent.get("id")),
            limit=4,
        )
        recent_alert_context: list[dict[str, Any]] = []
        for item in prior_alerts:
            recent_alert_context.append(
                {
                    "created_at": item.get("created_at"),
                    "trigger_reason": str(item.get("trigger_reason") or ""),
                    "narrative": str(item.get("narrative") or "")[:240],
                    "market_snapshot": item.get("market_snapshot") if isinstance(item.get("market_snapshot"), dict) else {},
                }
            )
        prior_alert_line = " | ".join(
            [
                f"{str(row.get('created_at') or '')}: {str(row.get('trigger_reason') or '')}"
                for row in recent_alert_context[:4]
                if str(row.get("trigger_reason") or "").strip()
            ]
        ) or "none"
        prior_context_line = " | ".join(
            [
                str(item.get("context_summary") or "").strip()
                for item in prior_context_rows[:3]
                if str(item.get("context_summary") or "").strip()
            ]
        ) or "none"
        source_scores = research.get("breakdown") if isinstance(research.get("breakdown"), dict) else {}
        source_summaries = (
            research.get("breakdown_summaries")
            if isinstance(research.get("breakdown_summaries"), dict)
            else {}
        )
        source_score_tokens: list[str] = []
        for label, score in list(source_scores.items())[:4]:
            try:
                source_score_tokens.append(f"{str(label)}={float(score):+.2f}")
            except Exception:
                continue
        source_score_line = " | ".join(source_score_tokens) or "none"
        source_summary_line = " | ".join(
            [f"{str(label)}: {str(summary)[:120]}" for label, summary in list(source_summaries.items())[:3] if str(summary).strip()]
        ) or "none"
        prediction_line = "none"
        prediction_markets = research.get("prediction_markets") if isinstance(research.get("prediction_markets"), list) else []
        if prediction_markets:
            top_market = prediction_markets[0] if isinstance(prediction_markets[0], dict) else {}
            top_question = str(top_market.get("question") or top_market.get("title") or "").strip()
            top_relevance = float(top_market.get("relevance_score", 0.0) or 0.0)
            if top_question:
                prediction_line = f"{top_question[:140]} (rel={top_relevance:.2f})"
        context = (
            f"Agent={agent.get('name')} symbol={metric.ticker} price={metric.price} "
            f"change={effective_price_change:.2f}% baseline={baseline_mode} "
            f"aggregate_sentiment={aggregate:.2f} x_score={x_score:.2f} "
            f"source_scores={source_score_line} prediction_market={prediction_line} "
            f"source_notes={source_summary_line} "
            f"matches={'; '.join(reasons)} prior_alerts={prior_alert_line} "
            f"prior_analysis={prior_context_line}"
        )
        if quick_investigation:
            context = f"{context} | catalyst_note={quick_investigation}"
        narrative = await self._synthesize_narrative(
            metric.ticker,
            context,
            style=str(tooling.get("notification_style") or "auto"),
            manager_prompt=str(tooling.get("manager_prompt") or ""),
            event_type=event_type,
        )

        simulation_context: dict[str, Any] | None = None
        launched_session_id: str | None = None
        simulation_requested = bool(tooling.get("simulate_on_alert")) or ("simulation" in tools)
        if ("simulation" in tools and simulation_summary_enabled) or simulation_requested:
            simulation_context = self._quick_simulation_context(
                price_change=effective_price_change,
                aggregate_sentiment=aggregate,
                base_price=float(metric.price),
            )
            if simulation_requested and event_type == "alert":
                launched_session_id = await self._maybe_launch_simulation_session(
                    metric=metric,
                    user_id=agent.get("user_id"),
                )

        channels = self._resolve_channels(tooling, notification_prefs)
        notification_allowed, block_reason = self._notification_allowed(agent, notification_prefs, now)
        if event_type == "report" and not notification_allowed:
            tracker_repo.create_run(
                user_id=agent.get("user_id"),
                agent_id=str(agent.get("id")),
                symbol=metric.ticker,
                run_type="report_skipped",
                trigger_reasons=reasons,
                tools_used=list(tools),
                research_sources=list(tooling.get("research_sources") or []),
                market_snapshot={
                    "price": metric.price,
                    "price_change_pct": round(effective_price_change, 4),
                    "baseline_mode": baseline_mode,
                    "baseline_price": baseline_price,
                    "volume": metric.volume,
                    "volume_ratio": round(volume_ratio, 4),
                },
                research_snapshot=research,
                decision={"notification_allowed": False, "block_reason": block_reason, "event_type": event_type},
                note="Periodic report skipped due to notification preferences.",
            )
            await self._update_agent_thesis(
                agent=agent,
                metric=metric,
                event_type="report_skipped",
                reasons=reasons,
                narrative=narrative,
                price_change=effective_price_change,
                research=research,
                simulation_context=simulation_context,
            )
            tracker_repo.update_agent(
                user_id=agent.get("user_id"),
                agent_id=str(agent.get("id")),
                updates={"last_checked_at": now.isoformat(), "last_price": float(metric.price)},
            )
            return None

        title = f"TickerMaster {'Report' if event_type == 'report' else 'Alert'}: {metric.ticker}"
        body = self._compose_notification_body(
            ticker=metric.ticker,
            event_type=event_type,
            reasons=reasons,
            metric=metric,
            price_change=effective_price_change,
            aggregate_sentiment=aggregate,
            research_sources=list(tooling.get("research_sources") or []),
            narrative=narrative,
            simulation_context=simulation_context,
            style=str(tooling.get("notification_style") or "auto"),
        )
        if notification_allowed:
            notification = await dispatch_alert_notification(
                settings=self.settings,
                title=title,
                body=body,
                link=f"https://localhost:5173?tab=tracker&ticker={metric.ticker}",
                preferred_channels=channels,
                to_number=await self._resolve_notification_phone(
                    user_id=agent.get("user_id"),
                    agent=agent,
                    tooling=tooling,
                    notification_prefs=notification_prefs,
                ),
                metadata={
                    "ticker": metric.ticker,
                    "price": metric.price,
                    "change_percent": round(effective_price_change, 2),
                    "reason": "; ".join(reasons),
                    "agent_id": agent.get("id"),
                    "event_type": event_type,
                },
            )
        else:
            notification = {
                "channels": channels,
                "delivered": False,
                "skipped": True,
                "reason": block_reason,
                "twilio": {"attempted": False, "delivered": False},
                "poke": {"attempted": False, "delivered": False},
            }

        alert = tracker_repo.create_alert(
            symbol=metric.ticker,
            trigger_reason="; ".join(reasons),
            narrative=narrative,
            market_snapshot={
                "price": metric.price,
                "change_percent": round(effective_price_change, 2),
                "volume": metric.volume,
                "volume_ratio": round(volume_ratio, 3),
                "sentiment": aggregate,
                "x_sentiment": x_score,
                "baseline_mode": baseline_mode,
                "baseline_price": baseline_price,
            },
            investigation_data={
                "research": research,
                "investigation": quick_investigation,
                "simulation": simulation_context,
                "simulation_session_id": launched_session_id,
                "notification": notification,
                "tooling": tooling,
                "runtime_tool_plan": tooling.get("runtime_tool_plan"),
                "notification_preferences": notification_prefs,
                "event_type": event_type,
                "baseline_mode": baseline_mode,
            },
            user_id=agent.get("user_id"),
            agent_id=agent.get("id"),
            simulation_id=launched_session_id,
            poke_sent=bool(notification.get("delivered")),
        )
        alert_context_payload = {
            "event_type": event_type,
            "reasons": reasons,
            "analysis_context": context,
            "market_snapshot": {
                "price": metric.price,
                "change_percent": round(effective_price_change, 2),
                "volume": metric.volume,
                "volume_ratio": round(volume_ratio, 3),
                "baseline_mode": baseline_mode,
                "baseline_price": baseline_price,
            },
            "research_snapshot": research,
            "quick_investigation": quick_investigation,
            "recent_alerts": recent_alert_context,
            "recent_analysis_context": prior_context_rows,
            "simulation": simulation_context or {},
            "simulation_session_id": launched_session_id,
            "tooling": tooling,
        }
        tracker_repo.create_alert_context(
            user_id=agent.get("user_id"),
            agent_id=str(agent.get("id")),
            symbol=metric.ticker,
            alert_id=str(alert.get("id")) if alert.get("id") else None,
            event_type=event_type,
            context_summary=f"{'; '.join(reasons)} | price={float(metric.price):.2f}"[:300],
            context_payload=alert_context_payload,
            simulation_requested=simulation_requested,
        )
        if agent.get("user_id") and agent.get("id"):
            try:
                await asyncio.to_thread(
                    append_alert_context_csv,
                    user_id=str(agent.get("user_id")),
                    agent_id=str(agent.get("id")),
                    symbol=str(metric.ticker),
                    event_type=event_type,
                    generated_at=now.isoformat(),
                    alert_id=str(alert.get("id")) if alert.get("id") else None,
                    context_summary=f"{'; '.join(reasons)} | price={float(metric.price):.2f}"[:300],
                    simulation_requested=simulation_requested,
                    context_payload=alert_context_payload,
                )
            except Exception:
                pass
            try:
                memory_export = await asyncio.to_thread(
                    append_agent_memory_documents,
                    user_id=str(agent.get("user_id")),
                    agent_id=str(agent.get("id")),
                    symbol=str(metric.ticker),
                    generated_at=now.isoformat(),
                    event_type=event_type,
                    manager_instruction=str(tooling.get("manager_prompt") or ""),
                    agent_response=str(narrative or ""),
                    context_payload={
                        "reasons": reasons,
                        "research": research,
                        "quick_investigation": quick_investigation,
                        "simulation": simulation_context or {},
                        "runtime_tool_plan": tooling.get("runtime_tool_plan"),
                        "market_snapshot": {
                            "price": metric.price,
                            "change_percent": round(effective_price_change, 2),
                            "volume": metric.volume,
                            "volume_ratio": round(volume_ratio, 3),
                        },
                    },
                    mcp_debug=(
                        research.get("mcp_debug")
                        if isinstance(research.get("mcp_debug"), dict)
                        else {}
                    ),
                )
                if isinstance(memory_export, dict):
                    alert_context_payload["memory_export"] = memory_export
            except Exception:
                pass
        update_payload = {
            "last_checked_at": now.isoformat(),
            "last_alert_at": now.isoformat(),
            "last_price": float(metric.price),
        }
        if event_type == "alert":
            update_payload["total_alerts"] = int(agent.get("total_alerts") or 0) + 1
            existing_triggers = agent.get("triggers") if isinstance(agent.get("triggers"), dict) else {}
            update_payload["triggers"] = {
                **(existing_triggers or {}),
                "last_alert_price": float(metric.price),
            }
        tracker_repo.update_agent(user_id=agent.get("user_id"), agent_id=str(agent.get("id")), updates=update_payload)

        tracker_repo.create_run(
            user_id=agent.get("user_id"),
            agent_id=str(agent.get("id")),
            symbol=metric.ticker,
            run_type=event_type,
            trigger_reasons=reasons,
            tools_used=list(tools),
            research_sources=list(tooling.get("research_sources") or []),
            market_snapshot={
                "price": metric.price,
                "price_change_pct": round(effective_price_change, 4),
                "baseline_mode": baseline_mode,
                "baseline_price": baseline_price,
                "volume": metric.volume,
                "volume_ratio": round(volume_ratio, 4),
            },
            research_snapshot=research,
            simulation_snapshot={
                "context": simulation_context or {},
                "session_id": launched_session_id,
            },
            decision={
                "notification": notification,
                "event_type": event_type,
            },
            note=narrative[:400],
        )
        await self._update_agent_thesis(
            agent=agent,
            metric=metric,
            event_type=event_type,
            reasons=reasons,
            narrative=narrative,
            price_change=effective_price_change,
            research=research,
            simulation_context=simulation_context,
        )

        await log_agent_activity(
            module="tracker",
            agent_name=str(agent.get("name") or f"{metric.ticker} Associate"),
            action=f"Published associate {event_type} for {metric.ticker}",
            status="success",
            user_id=agent.get("user_id"),
            details={
                "agent_id": agent.get("id"),
                "symbol": metric.ticker,
                "description": "Generated narrative from configured tools and persisted tracker output.",
                "event_type": event_type,
                "trigger_reason": "; ".join(reasons),
                "narrative": narrative[:300],
                "notification_delivered": bool(notification.get("delivered")),
                "channels": channels,
                "simulation_session_id": launched_session_id,
                "baseline_mode": baseline_mode,
                "notification_style": tooling.get("notification_style"),
                "runtime_tool_plan": tooling.get("runtime_tool_plan"),
            },
        )
        return {
            "ticker": metric.ticker,
            "agent_id": agent.get("id"),
            "agent_name": agent.get("name"),
            "event_type": event_type,
            "reason": "; ".join(reasons),
            "analysis": narrative,
            "change_percent": round(effective_price_change, 2),
            "price": metric.price,
            "alert_id": alert.get("id"),
            "alert_time": alert.get("created_at"),
            "notification": notification,
            "simulation": simulation_context,
            "simulation_session_id": launched_session_id,
            "baseline_mode": baseline_mode,
        }

    async def poll_once(self) -> TrackerSnapshot:
        now = datetime.now(timezone.utc)

        symbol_agents: dict[str, list[dict[str, Any]]] = defaultdict(list)
        agents = tracker_repo.list_agents(user_id=None)
        for agent in agents:
            if str(agent.get("status", "")).lower() != "active":
                continue
            symbol = str(agent.get("symbol", "")).upper()
            if symbol:
                symbol_agents[symbol].append(agent)

        symbols = set(self.list_watchlist())
        symbols.update(symbol_agents.keys())
        metrics = await asyncio.to_thread(fetch_watchlist_metrics, sorted(symbols))

        alerts_triggered: List[Dict[str, Any]] = []
        for metric in metrics:
            attached_agents = symbol_agents.get(metric.ticker, [])
            previous = self._previous.get(metric.ticker)
            previous_price = float(previous.get("price", 0.0)) if previous else 0.0
            previous_volume = float(previous.get("volume", 0.0)) if previous else 0.0
            price_change = ((metric.price - previous_price) / previous_price) * 100 if previous_price else 0.0
            observed_volume_ratio = (float(metric.volume or 0.0) / previous_volume) if previous_volume > 0 else 1.0
            generic_volume_spike = observed_volume_ratio >= 1.8

            if attached_agents:
                evaluation_tasks: list[asyncio.Task] = []
                for agent in attached_agents:
                    tooling = self._agent_tooling_config(agent)
                    if not self._agent_poll_due(agent, tooling, now):
                        continue
                    if self._agent_in_cooldown(agent) and str(tooling.get("report_mode") or "hybrid") == "triggers_only":
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

                    await log_agent_activity(
                        module="tracker",
                        agent_name=str(agent.get("name") or f"{metric.ticker} Tracker"),
                        action=f"Monitoring {metric.ticker}: price {metric.price:.2f}, move {price_change:.2f}%",
                        status="running",
                        user_id=agent.get("user_id"),
                        details={
                            "agent_id": agent.get("id"),
                            "symbol": metric.ticker,
                            "description": "Scanning configured tools and thresholds.",
                            "change_percent": round(price_change, 3),
                            "volume": metric.volume,
                            "volume_ratio": round(observed_volume_ratio, 3),
                            "tools": tooling.get("tools"),
                        },
                    )
                    evaluation_tasks.append(
                        asyncio.create_task(
                            self._evaluate_agent(
                                agent=agent,
                                metric=metric,
                                price_change=price_change,
                                volume_ratio=observed_volume_ratio,
                            )
                        )
                    )

                if evaluation_tasks:
                    results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception) or result is None:
                            continue
                        alerts_triggered.append(result)
            else:
                await log_agent_activity(
                    module="tracker",
                    agent_name=f"{metric.ticker} Tracker",
                    action=f"Price check {metric.ticker} @ {metric.price}",
                    status="running",
                    details={
                        "symbol": metric.ticker,
                        "description": "Price scan completed with no dedicated deployed agent found.",
                        "change_percent": round(price_change, 3),
                        "volume": metric.volume,
                        "volume_ratio": round(observed_volume_ratio, 3),
                    },
                )

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

            # Keep a baseline crawler for unassigned watchlist symbols.
            if (not attached_agents) and (abs(price_change) > 2.0 or generic_volume_spike):
                trigger_context = (
                    f"price_change={price_change:.2f}% volume={metric.volume} prior_volume={previous_volume:.0f}"
                )
                why = await self._investigate_with_perplexity(metric.ticker, trigger_context)
                synthesis = await self._synthesize_narrative(metric.ticker, why, style="short")
                notification = await dispatch_alert_notification(
                    settings=self.settings,
                    title=f"TickerMaster Alert: {metric.ticker}",
                    body=synthesis[:220],
                    link=f"https://localhost:5173?tab=tracker&ticker={metric.ticker}",
                    preferred_channels=["twilio", "poke"],
                    metadata={
                        "ticker": metric.ticker,
                        "price": metric.price,
                        "change_percent": round(price_change, 2),
                        "reason": trigger_context,
                    },
                )
                event = {
                    "ticker": metric.ticker,
                    "reason": trigger_context,
                    "investigation": why,
                    "analysis": synthesis,
                    "change_percent": round(price_change, 2),
                    "price": metric.price,
                    "notification": notification,
                }
                alerts_triggered.append(event)
                tracker_repo.create_alert(
                    symbol=metric.ticker,
                    trigger_reason=trigger_context,
                    narrative=synthesis,
                    market_snapshot={
                        "price": metric.price,
                        "change_percent": round(price_change, 2),
                        "volume": metric.volume,
                        "volume_ratio": round(observed_volume_ratio, 3),
                    },
                    investigation_data={"investigation": why, "notification": notification},
                    user_id=None,
                    poke_sent=bool(notification.get("delivered")),
                )
                await log_agent_activity(
                    module="tracker",
                    agent_name=f"{metric.ticker} Tracker",
                    action=f"Baseline crawler alert fired for {metric.ticker}",
                    status="success",
                    details={
                        "symbol": metric.ticker,
                        "description": "Independent crawler detected unusual move and sent notification.",
                        "reason": trigger_context,
                        "notification_delivered": bool(notification.get("delivered")),
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
