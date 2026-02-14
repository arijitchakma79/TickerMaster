from __future__ import annotations

from typing import Any, Dict

import httpx

from app.config import Settings


_SERIES = {
    "fed_funds_rate": ("FEDFUNDS", "Federal Funds Rate", "%"),
    "cpi_yoy": ("CPIAUCSL", "CPI YoY", "%"),
    "unemployment": ("UNRATE", "Unemployment Rate", "%"),
    "ten_year_treasury": ("DGS10", "10Y Treasury Yield", "%"),
    "vix": ("VIXCLS", "VIX", ""),
}


async def _latest_observation(series_id: str, settings: Settings) -> float | None:
    if not settings.fred_api_key:
        return None

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 24,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            observations = resp.json().get("observations", [])
        for item in observations:
            raw = str(item.get("value", "."))
            if raw and raw != ".":
                return float(raw)
    except Exception:
        return None
    return None


async def get_macro_indicators(settings: Settings) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for key, (series_id, name, unit) in _SERIES.items():
        value = await _latest_observation(series_id, settings)
        output[key] = {"name": name, "unit": unit, "value": value}
    return output
