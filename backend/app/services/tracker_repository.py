from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.services.database import get_supabase


class TrackerRepository:
    def __init__(self) -> None:
        self._agents: dict[str, dict[str, Any]] = {}
        self._alerts: list[dict[str, Any]] = []
        self._alert_context: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._runs: list[dict[str, Any]] = []
        self._thesis: dict[str, dict[str, Any]] = {}

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_agent(
        self,
        user_id: str | None,
        symbol: str,
        name: str,
        triggers: dict[str, Any],
        auto_simulate: bool,
        *,
        strict_persistence: bool = False,
    ) -> dict[str, Any]:
        client = get_supabase()
        payload = {
            "user_id": user_id,
            "symbol": symbol.upper(),
            "name": name,
            "status": "active",
            "triggers": triggers,
            "auto_simulate": auto_simulate,
        }
        if strict_persistence and (not client or not user_id):
            raise RuntimeError("Supabase client or user_id missing; cannot persist tracker agent.")
        if client and user_id:
            try:
                data = client.table("tracker_agents").insert(payload).execute().data
                if data:
                    return data[0]
            except Exception as exc:
                if strict_persistence:
                    raise RuntimeError(f"Failed to persist tracker agent: {exc}") from exc

        if strict_persistence:
            raise RuntimeError("Tracker agent was not persisted to Supabase.")

        agent_id = str(uuid.uuid4())
        agent = {
            "id": agent_id,
            **payload,
            "total_alerts": 0,
            "last_alert_at": None,
            "created_at": self._now(),
            "updated_at": self._now(),
        }
        self._agents[agent_id] = agent
        return agent

    def create_history(
        self,
        *,
        user_id: str | None,
        agent_id: str,
        event_type: str,
        raw_prompt: str | None = None,
        parsed_intent: dict[str, Any] | None = None,
        trigger_snapshot: dict[str, Any] | None = None,
        tool_outputs: dict[str, Any] | None = None,
        note: str | None = None,
        strict_persistence: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "agent_id": agent_id,
            "user_id": user_id,
            "event_type": event_type,
            "raw_prompt": raw_prompt,
            "parsed_intent": parsed_intent or {},
            "trigger_snapshot": trigger_snapshot or {},
            "tool_outputs": tool_outputs or {},
            "note": note,
        }

        client = get_supabase()
        if strict_persistence and (not client or not user_id):
            raise RuntimeError("Supabase client or user_id missing; cannot persist tracker agent history.")
        if client and user_id:
            try:
                data = client.table("tracker_agent_history").insert(payload).execute().data
                if data:
                    return data[0]
            except Exception as exc:
                if strict_persistence:
                    raise RuntimeError(f"Failed to persist tracker agent history: {exc}") from exc

        if strict_persistence:
            raise RuntimeError("Tracker agent history was not persisted to Supabase.")

        row = {
            "id": str(uuid.uuid4()),
            **payload,
            "created_at": self._now(),
        }
        self._history.append(row)
        return row

    def list_history(self, user_id: str | None, agent_id: str, limit: int = 20) -> list[dict[str, Any]]:
        client = get_supabase()
        if client and user_id:
            try:
                data = (
                    client.table("tracker_agent_history")
                    .select("*")
                    .eq("user_id", user_id)
                    .eq("agent_id", agent_id)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                    .data
                )
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        data = [item for item in self._history if item.get("agent_id") == agent_id and ((not user_id) or item.get("user_id") == user_id)]
        return sorted(data, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    def create_run(
        self,
        *,
        user_id: str | None,
        agent_id: str,
        symbol: str,
        run_type: str,
        trigger_reasons: list[str] | None = None,
        tools_used: list[str] | None = None,
        research_sources: list[str] | None = None,
        market_snapshot: dict[str, Any] | None = None,
        research_snapshot: dict[str, Any] | None = None,
        simulation_snapshot: dict[str, Any] | None = None,
        decision: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "agent_id": agent_id,
            "symbol": symbol.upper(),
            "run_type": run_type,
            "trigger_reasons": trigger_reasons or [],
            "tools_used": tools_used or [],
            "research_sources": research_sources or [],
            "market_snapshot": market_snapshot or {},
            "research_snapshot": research_snapshot or {},
            "simulation_snapshot": simulation_snapshot or {},
            "decision": decision or {},
            "note": note,
        }

        client = get_supabase()
        if client and user_id:
            try:
                data = client.table("tracker_agent_runs").insert(payload).execute().data
                if data:
                    return data[0]
            except Exception:
                pass

        row = {
            "id": str(uuid.uuid4()),
            **payload,
            "created_at": self._now(),
        }
        self._runs.append(row)
        return row

    def list_runs(self, user_id: str | None, agent_id: str, limit: int = 50) -> list[dict[str, Any]]:
        client = get_supabase()
        if client and user_id:
            try:
                data = (
                    client.table("tracker_agent_runs")
                    .select("*")
                    .eq("user_id", user_id)
                    .eq("agent_id", agent_id)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                    .data
                )
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        data = [item for item in self._runs if item.get("agent_id") == agent_id and ((not user_id) or item.get("user_id") == user_id)]
        return sorted(data, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    def upsert_thesis(
        self,
        *,
        user_id: str | None,
        agent_id: str,
        symbol: str,
        stance_score: float,
        confidence: float,
        thesis: dict[str, Any] | None = None,
        summary: str | None = None,
        last_event_type: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "agent_id": agent_id,
            "symbol": symbol.upper(),
            "stance_score": float(max(-1.0, min(1.0, stance_score))),
            "confidence": float(max(0.0, min(1.0, confidence))),
            "thesis": thesis or {},
            "summary": summary,
            "last_event_type": last_event_type,
            "updated_at": self._now(),
        }

        client = get_supabase()
        if client and user_id:
            try:
                # Upsert by agent id keeps one evolving thesis per agent.
                data = (
                    client.table("tracker_agent_thesis")
                    .upsert(payload, on_conflict="agent_id")
                    .execute()
                    .data
                )
                if data:
                    return data[0]
            except Exception:
                pass

        local = {
            "id": str(uuid.uuid4()),
            **payload,
            "created_at": self._thesis.get(agent_id, {}).get("created_at", self._now()),
        }
        self._thesis[agent_id] = local
        return local

    def get_thesis(self, user_id: str | None, agent_id: str) -> dict[str, Any] | None:
        client = get_supabase()
        if client and user_id:
            try:
                data = (
                    client.table("tracker_agent_thesis")
                    .select("*")
                    .eq("user_id", user_id)
                    .eq("agent_id", agent_id)
                    .single()
                    .execute()
                    .data
                )
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        row = self._thesis.get(agent_id)
        if not row:
            return None
        if user_id and row.get("user_id") != user_id:
            return None
        return row

    def list_agents(self, user_id: str | None) -> list[dict[str, Any]]:
        client = get_supabase()
        if client:
            try:
                query = client.table("tracker_agents").select("*").neq("status", "deleted").order("created_at", desc=True)
                if user_id:
                    query = query.eq("user_id", user_id)
                data = query.execute().data
                if isinstance(data, list):
                    if user_id:
                        return data
                    local = [a for a in self._agents.values() if a.get("status") != "deleted"]
                    known_ids = {str(item.get("id")) for item in data}
                    merged = data + [item for item in local if str(item.get("id")) not in known_ids]
                    return merged
            except Exception:
                pass
        return [a for a in self._agents.values() if (not user_id or a.get("user_id") == user_id) and a.get("status") != "deleted"]

    def update_agent(self, user_id: str | None, agent_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        client = get_supabase()
        updates = {**updates, "updated_at": self._now()}
        if client and user_id:
            try:
                data = (
                    client.table("tracker_agents")
                    .update(updates)
                    .eq("id", agent_id)
                    .eq("user_id", user_id)
                    .execute()
                    .data
                )
                if data:
                    return data[0]
            except Exception:
                pass

        agent = self._agents.get(agent_id)
        if not agent or (user_id and agent.get("user_id") != user_id):
            return None
        agent.update(updates)
        return agent

    def get_agent(self, user_id: str | None, agent_id: str) -> dict[str, Any] | None:
        client = get_supabase()
        if client and user_id:
            try:
                data = (
                    client.table("tracker_agents")
                    .select("*")
                    .eq("id", agent_id)
                    .eq("user_id", user_id)
                    .single()
                    .execute()
                    .data
                )
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        agent = self._agents.get(agent_id)
        if not agent:
            return None
        if user_id and agent.get("user_id") != user_id:
            return None
        return agent

    def delete_agent(self, user_id: str | None, agent_id: str) -> bool:
        out = self.update_agent(user_id=user_id, agent_id=agent_id, updates={"status": "deleted"})
        return out is not None

    def list_alerts(self, user_id: str | None, agent_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        client = get_supabase()
        if client and user_id:
            try:
                query = client.table("tracker_alerts").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit)
                if agent_id:
                    query = query.eq("agent_id", agent_id)
                data = query.execute().data
                if isinstance(data, list) and data:
                    return data
            except Exception:
                pass

        data = [a for a in self._alerts if ((not user_id) or a.get("user_id") == user_id) and (not agent_id or a.get("agent_id") == agent_id)]
        return sorted(data, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    def create_alert(
        self,
        symbol: str,
        trigger_reason: str,
        narrative: str | None,
        market_snapshot: dict[str, Any] | None = None,
        investigation_data: dict[str, Any] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        poke_sent: bool = False,
        simulation_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "agent_id": agent_id,
            "user_id": user_id,
            "symbol": symbol.upper(),
            "trigger_reason": trigger_reason,
            "narrative": narrative,
            "market_snapshot": market_snapshot or {},
            "investigation_data": investigation_data or {},
            "poke_sent": poke_sent,
            "simulation_id": simulation_id,
        }

        client = get_supabase()
        if client:
            try:
                data = client.table("tracker_alerts").insert(payload).execute().data
                if data:
                    return data[0]
            except Exception:
                pass

        row = {
            "id": str(uuid.uuid4()),
            **payload,
            "created_at": self._now(),
        }
        self._alerts.append(row)
        return row

    def create_alert_context(
        self,
        *,
        user_id: str | None,
        agent_id: str,
        symbol: str,
        event_type: str,
        context_payload: dict[str, Any],
        alert_id: str | None = None,
        context_summary: str | None = None,
        simulation_requested: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "agent_id": agent_id,
            "alert_id": alert_id,
            "symbol": symbol.upper(),
            "event_type": event_type,
            "context_summary": context_summary,
            "context_payload": context_payload or {},
            "simulation_requested": bool(simulation_requested),
        }

        client = get_supabase()
        if client and user_id:
            try:
                data = client.table("tracker_alert_context").insert(payload).execute().data
                if data:
                    return data[0]
            except Exception:
                pass

        row = {
            "id": str(uuid.uuid4()),
            **payload,
            "created_at": self._now(),
        }
        self._alert_context.append(row)
        return row

    def list_alert_context(self, user_id: str | None, agent_id: str, limit: int = 40) -> list[dict[str, Any]]:
        client = get_supabase()
        if client and user_id:
            try:
                data = (
                    client.table("tracker_alert_context")
                    .select("*")
                    .eq("user_id", user_id)
                    .eq("agent_id", agent_id)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                    .data
                )
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        data = [
            item
            for item in self._alert_context
            if item.get("agent_id") == agent_id and ((not user_id) or item.get("user_id") == user_id)
        ]
        return sorted(data, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]


tracker_repo = TrackerRepository()
