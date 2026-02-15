from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Optional

from app.config import Settings
from app.schemas import AgentConfig
from app.services.modal_integration import modal_cron_health, spin_modal_sandbox
from app.services.persona_training import infer_persona_params

# Simple, explicit "public profile" blobs used to seed persona inference.
# In a fuller implementation, this can be replaced with a real SEC EDGAR fetcher.
_SEC13F_PUBLIC_PROFILES: dict[str, dict[str, Any]] = {
    "BlackRock Macro Core": {
        "source": "SEC 13F",
        "summary": "Large diversified allocator; low turnover; prefers liquid mega-caps; risk managed sizing.",
        "signals": {"top10_concentration": 0.27, "turnover": 0.06},
    },
    "Citadel Execution Desk": {
        "source": "SEC 13F",
        "summary": "High-frequency / systematic desk proxy; higher turnover; reacts to momentum and news skew.",
        "signals": {"top10_concentration": 0.35, "turnover": 0.16},
    },
    "Jane Street Microflow": {
        "source": "SEC 13F",
        "summary": "Market-making / microstructure proxy; fast reaction; sizes dynamically; disciplined risk.",
        "signals": {"top10_concentration": 0.46, "turnover": 0.14},
    },
    "Vanguard Index Sentinel": {
        "source": "SEC 13F",
        "summary": "Index allocator proxy; lowest turnover; passive bias; tight risk, steady sizing.",
        "signals": {"top10_concentration": 0.31, "turnover": 0.05},
    },
}


def sec13f_public_profile(persona_name: str) -> Optional[dict[str, Any]]:
    """Return a public-profile blob for a known persona (or None if unknown)."""
    return _SEC13F_PUBLIC_PROFILES.get(str(persona_name or "").strip())


async def calibrate_personas(
    settings: Settings,
    agents: Iterable[AgentConfig],
    *,
    prefer_modal: bool = True,
    max_concurrency: int = 2,
) -> List[AgentConfig]:
    """
    For any agent with a known SEC13F-derived public profile, infer persona params via Modal (preferred)
    and return updated AgentConfig objects. Unknown agents pass through unchanged.
    """

    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    agent_list = list(agents)
    results: list[AgentConfig] = [agent for agent in agent_list]

    async def _calibrate_one(index: int, agent: AgentConfig) -> None:
        profile = sec13f_public_profile(agent.name)
        if not profile:
            return
        async with semaphore:
            inferred = await infer_persona_params(
                settings,
                persona_name=agent.name,
                public_profile=profile,
                prefer_modal=prefer_modal,
            )
        # Keep the original name stable (some LLMs may try to rename).
        inferred.name = agent.name
        results[index] = inferred

    tasks = [asyncio.create_task(_calibrate_one(i, agent)) for i, agent in enumerate(agent_list)]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return results


__all__ = [
    "calibrate_personas",
    "modal_cron_health",
    "sec13f_public_profile",
    "spin_modal_sandbox",
]

