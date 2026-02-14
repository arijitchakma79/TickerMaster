from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.services.database import get_supabase


class TrackerRepository:
    def __init__(self) -> None:
        self._agents: dict[str, dict[str, Any]] = {}
        self._alerts: list[dict[str, Any]] = []

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_agent(self, user_id: str | None, symbol: str, name: str, triggers: dict[str, Any], auto_simulate: bool) -> dict[str, Any]:
        client = get_supabase()
        payload = {
            "user_id": user_id,
            "symbol": symbol.upper(),
            "name": name,
            "status": "active",
            "triggers": triggers,
            "auto_simulate": auto_simulate,
        }
        if client and user_id:
            try:
                data = client.table("tracker_agents").insert(payload).execute().data
                if data:
                    return data[0]
            except Exception:
                pass

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


tracker_repo = TrackerRepository()
