from __future__ import annotations

import asyncio
import csv
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np

from app.config import get_settings
from app.schemas import CandlestickPoint, MarketMetric, TickerLookup

_LAST_METRIC_CACHE: dict[str, MarketMetric] = {}
_LAST_CANDLES_CACHE: dict[str, List[CandlestickPoint]] = {}
_LAST_ADVANCED_CACHE: dict[str, tuple[float, Dict[str, Any]]] = {}
_ADVANCED_CACHE_TTL_SECONDS = 15 * 60
_SEC_TICKER_CACHE: tuple[datetime, list[tuple[str, str]]] | None = None
_SEC_TICKER_CACHE_TTL = timedelta(hours=24)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            clean = value.strip()
            if not clean:
                return None
            clean = clean.replace(",", "").replace("$", "").replace("%", "")
            value = clean
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_metric_quality_valid(metric: MarketMetric | None) -> bool:
    if metric is None:
        return False
    price = _safe_float(metric.price)
    if price is None or price <= 0:
        return False
    volume = _safe_int(metric.volume)
    market_cap = _safe_float(metric.market_cap)
    change = _safe_float(metric.change_percent) or 0.0
    # Require at least one additional signal so invalid symbols with zeroed quotes do not pass.
    return bool((volume and volume > 0) or (market_cap and market_cap > 0) or abs(change) > 0.0001)


def _synthetic_metric(symbol: str) -> MarketMetric:
    seed = abs(hash(symbol)) % 10_000
    rng = np.random.default_rng(seed)
    price = float(rng.uniform(20, 500))
    change = float(rng.uniform(-3, 3))
    return MarketMetric(
        ticker=symbol,
        price=round(price, 2),
        change_percent=round(change, 2),
        pe_ratio=round(float(rng.uniform(8, 40)), 2),
        beta=round(float(rng.uniform(0.7, 2.1)), 2),
        volume=int(rng.uniform(2_000_000, 150_000_000)),
        market_cap=float(rng.uniform(5e9, 3e12)),
    )


def _synthetic_candles(symbol: str, count: int = 30) -> List[CandlestickPoint]:
    now = datetime.now(timezone.utc)
    baseline = abs(hash(symbol)) % 300 + 50
    points: List[CandlestickPoint] = []
    for idx in range(count):
        px = baseline + np.sin(idx / 4) * 4
        ts = (now - timedelta(days=(count - idx))).replace(microsecond=0)
        points.append(
            CandlestickPoint(
                timestamp=ts.isoformat(),
                open=round(px - 0.5, 2),
                high=round(px + 1.3, 2),
                low=round(px - 1.2, 2),
                close=round(px + 0.6, 2),
                volume=2_000_000 + idx * 12000,
            )
        )
    return points


def _settings():
    return get_settings()


def _is_symbol_candidate(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", value))


_US_LOOKUP_EXCHANGES = {
    "US",
    "NASDAQ",
    "NYSE",
    "NYSEARCA",
    "AMEX",
    "BATS",
    "IEX",
    "NMS",
    "NYQ",
    "PCX",
    "ASE",
}

_COMMON_COMPANY_ALIASES = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "NVIDA": "NVDA",
    "TESLA": "TSLA",
    "AMAZON": "AMZN",
    "META": "META",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "NETFLIX": "NFLX",
    "PALANTIR": "PLTR",
    "SP500": "SPY",
    "S&P 500": "SPY",
    "S P 500": "SPY",
}


def _normalize_lookup_symbol(raw: Any) -> str:
    token = str(raw or "").upper().strip()
    if not token:
        return ""
    token = token.split(":")[-1].strip()
    token = token.replace(".", "-")
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"[-.]US$", "", token)
    token = token.strip("-")
    return token


def _normalize_lookup_text(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())


def _lookup_score(query: str, item: TickerLookup) -> int:
    clean_query = str(query or "").strip()
    if not clean_query:
        return 0
    query_norm = _normalize_lookup_text(clean_query)
    symbol = _normalize_lookup_symbol(item.ticker)
    symbol_norm = _normalize_lookup_text(symbol)
    name = str(item.name or symbol).strip()
    name_norm = _normalize_lookup_text(name)

    score = 0
    if clean_query.upper() == symbol:
        score = 100
    elif symbol.startswith(clean_query.upper()):
        score = 92
    elif symbol_norm == query_norm:
        score = 88
    elif symbol_norm.startswith(query_norm):
        score = 84
    elif name_norm == query_norm:
        score = 82
    elif name_norm.startswith(query_norm):
        score = 76
    elif query_norm and name_norm and query_norm in name_norm:
        score = 68

    exchange = str(item.exchange or "").upper()
    if exchange in _US_LOOKUP_EXCHANGES:
        score += 4
    elif exchange:
        score -= 4
    instrument_type = str(item.instrument_type or "").upper()
    if any(token in instrument_type for token in {"EQUITY", "COMMON"}):
        score += 3
    elif "ETF" in instrument_type:
        score += 2
    if any(token in instrument_type for token in {"WARRANT", "RIGHT", "PREFERRED", "PFD", "UNIT", "NOTE", "BOND"}):
        score -= 18

    if re.search(r"\d", symbol):
        score -= 10
    if len(symbol) > 5:
        score -= 6
    if "-" in symbol and not re.fullmatch(r"[A-Z]{1,5}-[A-Z]{1,2}", symbol):
        score -= 6
    name_upper = name.upper()
    if any(token in name_upper for token in {"PREFERRED", "PFD", "WARRANT", "RIGHT", "UNIT", "NOTE"}):
        score -= 10
    return score


def _rank_lookup_results(query: str, rows: List[TickerLookup], limit: int = 8) -> List[TickerLookup]:
    scored: dict[str, tuple[int, TickerLookup]] = {}
    for row in rows:
        symbol = _normalize_lookup_symbol(row.ticker)
        if not symbol or not _is_symbol_candidate(symbol):
            continue
        normalized = TickerLookup(
            ticker=symbol,
            name=str(row.name or symbol).strip() or symbol,
            exchange=str(row.exchange or "").strip() or None,
            instrument_type=str(row.instrument_type or "").strip() or None,
        )
        score = _lookup_score(query, normalized)
        if score <= 0:
            continue
        prev = scored.get(symbol)
        if prev is None or score > prev[0]:
            scored[symbol] = (score, normalized)

    ranked = sorted(
        scored.values(),
        key=lambda pair: (
            -pair[0],
            0 if str(pair[1].exchange or "").upper() in _US_LOOKUP_EXCHANGES else 1,
            len(pair[1].ticker),
            pair[1].ticker,
        ),
    )
    return [item for _, item in ranked[: max(1, limit)]]


def resolve_symbol_input(raw: str) -> str:
    clean = raw.strip()
    if not clean:
        raise ValueError("Ticker input is empty.")

    alias_key = re.sub(r"[^A-Za-z0-9&]+", " ", clean).strip().upper()
    if alias_key in _COMMON_COMPANY_ALIASES:
        return _COMMON_COMPANY_ALIASES[alias_key]

    direct = clean.upper().replace(".", "-")
    if _is_symbol_candidate(direct):
        # Plain alphabetic tokens longer than 5 characters are usually company names
        # ("NVIDIA"), not tradable symbols. Resolve them via provider search.
        if not re.fullmatch(r"[A-Z]{6,}", direct):
            return direct

    # Support formats like "Alaska Airlines (ALK)".
    grouped = re.search(r"\(([A-Za-z][A-Za-z0-9.-]{0,9})\)", clean)
    if grouped:
        grouped_symbol = grouped.group(1).upper().replace(".", "-")
        if _is_symbol_candidate(grouped_symbol):
            return grouped_symbol

    # Natural-language fallback: resolve via provider search.
    hits = _search_symbol_providers(clean, limit=8)
    if hits:
        return hits[0].ticker.upper().strip()

    raise ValueError(f"Could not resolve ticker symbol from input: {raw!r}")


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _period_days(period: str) -> int:
    mapping = {
        "1d": 1,
        "5d": 5,
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
        "ytd": 365,
        "max": 3650,
    }
    return mapping.get(period, 31)


def _alpaca_timeframe(interval: str) -> str:
    mapping = {
        "1m": "1Min",
        "2m": "2Min",
        "5m": "5Min",
        "15m": "15Min",
        "30m": "30Min",
        "60m": "1Hour",
        "90m": "1Hour",
        "1h": "1Hour",
        "1d": "1Day",
        "5d": "1Day",
        "1wk": "1Week",
        "1mo": "1Month",
        "3mo": "1Month",
    }
    return mapping.get(interval, "1Day")


def _finnhub_resolution(interval: str) -> str:
    mapping = {
        "1m": "1",
        "2m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
        "90m": "60",
        "1h": "60",
        "1d": "D",
        "5d": "D",
        "1wk": "W",
        "1mo": "M",
        "3mo": "M",
    }
    return mapping.get(interval, "D")


def _twelvedata_interval(interval: str) -> str:
    mapping = {
        "1m": "1min",
        "2m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "1h",
        "90m": "1h",
        "1h": "1h",
        "1d": "1day",
        "5d": "1day",
        "1wk": "1week",
        "1mo": "1month",
        "3mo": "1month",
    }
    return mapping.get(interval, "1day")


def _normalize_market_cap(value: Any) -> float | None:
    cap = _safe_float(value)
    if cap is None:
        return None
    # Finnhub profile often returns market cap in millions.
    return cap * 1_000_000 if 0 < cap < 10_000_000 else cap


def _alpaca_headers() -> dict[str, str] | None:
    settings = _settings()
    if not settings.alpaca_api_key or not settings.alpaca_api_secret:
        return None
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
        "Accept": "application/json",
    }


def _alpaca_get(path: str, params: dict[str, Any] | None = None, *, trading: bool = False) -> Any:
    headers = _alpaca_headers()
    if headers is None:
        return None

    settings = _settings()
    base = settings.alpaca_trading_url if trading else settings.alpaca_data_url
    url = f"{base.rstrip('/')}{path}"

    try:
        with httpx.Client(timeout=12.0, headers=headers) as client:
            resp = client.get(url, params=params)
            if resp.status_code in {401, 403, 404, 429}:
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


def _finnhub_get(path: str, params: dict[str, Any] | None = None) -> Any:
    settings = _settings()
    if not settings.finnhub_api_key:
        return None

    query = dict(params or {})
    query["token"] = settings.finnhub_api_key
    url = f"{settings.finnhub_api_url.rstrip('/')}{path}"

    try:
        with httpx.Client(timeout=12.0) as client:
            resp = client.get(url, params=query)
            if resp.status_code in {401, 403, 404, 429}:
                return None
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


def _twelvedata_get(path: str, params: dict[str, Any] | None = None) -> Any:
    settings = _settings()
    query = dict(params or {})
    query["apikey"] = settings.twelvedata_api_key or "demo"
    url = f"{settings.twelvedata_api_url.rstrip('/')}{path}"

    try:
        with httpx.Client(timeout=12.0) as client:
            resp = client.get(url, params=query)
            if resp.status_code in {401, 403, 404, 429}:
                return None
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("status") == "error":
                return None
            return payload
    except Exception:
        return None


def _twelvedata_statistics(symbol: str) -> dict[str, Any]:
    payload = _twelvedata_get("/statistics", {"symbol": symbol})
    return payload if isinstance(payload, dict) else {}


def _yahoo_quote(symbol: str) -> dict[str, Any]:
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_0) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        with httpx.Client(timeout=12.0, headers=headers) as client:
            resp = client.get(url, params={"symbols": symbol})
            if resp.status_code in {401, 403, 404, 429}:
                return {}
            resp.raise_for_status()
            payload = resp.json()
            rows = (
                payload.get("quoteResponse", {}).get("result", [])
                if isinstance(payload, dict)
                else []
            )
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                return rows[0]
    except Exception:
        return {}
    return {}


def _parse_twelvedata_datetime(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def _alpaca_snapshot(symbol: str) -> dict[str, Any] | None:
    payload = _alpaca_get(
        f"/v2/stocks/{symbol}/snapshot",
        params={"feed": _settings().alpaca_data_feed},
    )
    if isinstance(payload, dict) and "snapshot" in payload and isinstance(payload["snapshot"], dict):
        return payload["snapshot"]
    if isinstance(payload, dict) and (
        "latestTrade" in payload or "dailyBar" in payload or "prevDailyBar" in payload
    ):
        return payload
    return None


def _alpaca_metric(symbol: str) -> MarketMetric | None:
    snapshot = _alpaca_snapshot(symbol)
    if not snapshot:
        return None

    latest_trade = snapshot.get("latestTrade") or {}
    daily_bar = snapshot.get("dailyBar") or {}
    prev_daily_bar = snapshot.get("prevDailyBar") or {}

    price = _safe_float(latest_trade.get("p")) or _safe_float(daily_bar.get("c"))
    prev_close = _safe_float(prev_daily_bar.get("c")) or _safe_float(daily_bar.get("o")) or price

    if price is None or price <= 0:
        return None

    change_percent = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

    return MarketMetric(
        ticker=symbol,
        price=round(price, 2),
        change_percent=round(change_percent, 2),
        pe_ratio=None,
        beta=None,
        volume=_safe_int(daily_bar.get("v")),
        market_cap=None,
    )


def _finnhub_quote(symbol: str) -> dict[str, Any] | None:
    payload = _finnhub_get("/quote", {"symbol": symbol})
    if not isinstance(payload, dict):
        return None

    price = _safe_float(payload.get("c"))
    if price is None or price <= 0:
        return None

    prev = _safe_float(payload.get("pc")) or price
    change_percent = _safe_float(payload.get("dp"))
    if change_percent is None:
        change_percent = ((price - prev) / prev * 100) if prev else 0.0

    return {
        "price": price,
        "change_percent": change_percent,
        "volume": _safe_int(payload.get("v")),
    }


def _finnhub_basic_metrics(symbol: str) -> dict[str, Any]:
    payload = _finnhub_get("/stock/metric", {"symbol": symbol, "metric": "all"})
    metrics = payload.get("metric") if isinstance(payload, dict) else None
    return metrics if isinstance(metrics, dict) else {}


def _finnhub_profile(symbol: str) -> dict[str, Any]:
    payload = _finnhub_get("/stock/profile2", {"symbol": symbol})
    return payload if isinstance(payload, dict) else {}


def _finnhub_recommendation(symbol: str) -> dict[str, Any]:
    payload = _finnhub_get("/stock/recommendation", {"symbol": symbol})
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first
    return {}


def _finnhub_price_target(symbol: str) -> dict[str, Any]:
    payload = _finnhub_get("/stock/price-target", {"symbol": symbol})
    return payload if isinstance(payload, dict) else {}


def _finnhub_insider_transactions(symbol: str) -> list[dict[str, Any]]:
    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=365)).isoformat()
    payload = _finnhub_get(
        "/stock/insider-transactions",
        {
            "symbol": symbol,
            "from": start,
            "to": today.isoformat(),
        },
    )
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for row in rows[:30]:
        if not isinstance(row, dict):
            continue
        shares = _safe_float(row.get("share"))
        txn_price = _safe_float(row.get("transactionPrice"))
        value = (shares * txn_price) if shares is not None and txn_price is not None else None
        out.append(
            {
                "start_date": row.get("filingDate") or row.get("transactionDate"),
                "filer_name": row.get("name") or "Insider",
                "filer_relation": row.get("officerTitle") or row.get("position") or "Insider",
                "money_text": row.get("transactionCode") or row.get("transactionType") or "Insider transaction",
                "shares": shares,
                "value": value,
                "ownership": row.get("shareOwned") or row.get("change") or "",
            }
        )
    return out


def _alpaca_bars(symbol: str, period: str = "1mo", interval: str = "1d") -> List[CandlestickPoint]:
    timeframe = _alpaca_timeframe(interval)
    days = _period_days(period)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    if timeframe.endswith("Min"):
        limit = min(10_000, max(1000, days * 390))
    else:
        limit = min(10_000, max(200, days * 2))

    payload = _alpaca_get(
        f"/v2/stocks/{symbol}/bars",
        params={
            "feed": _settings().alpaca_data_feed,
            "timeframe": timeframe,
            "start": _iso_z(start),
            "end": _iso_z(end),
            "adjustment": "raw",
            "sort": "asc",
            "limit": limit,
        },
    )
    bars = payload.get("bars") if isinstance(payload, dict) else None
    if not isinstance(bars, list):
        return []

    out: List[CandlestickPoint] = []
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        close = _safe_float(bar.get("c"))
        if close is None:
            continue
        open_ = _safe_float(bar.get("o")) or close
        high = _safe_float(bar.get("h")) or close
        low = _safe_float(bar.get("l")) or close
        volume = _safe_float(bar.get("v")) or 0.0

        ts = bar.get("t")
        if isinstance(ts, str):
            try:
                timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
            except ValueError:
                continue
        else:
            continue

        out.append(
            CandlestickPoint(
                timestamp=timestamp,
                open=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=float(volume),
            )
        )
    return out


def _finnhub_candles(symbol: str, period: str = "1mo", interval: str = "1d") -> List[CandlestickPoint]:
    days = _period_days(period)
    end = int(datetime.now(timezone.utc).timestamp())
    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    payload = _finnhub_get(
        "/stock/candle",
        {
            "symbol": symbol,
            "resolution": _finnhub_resolution(interval),
            "from": start,
            "to": end,
        },
    )
    if not isinstance(payload, dict) or payload.get("s") != "ok":
        return []

    closes = payload.get("c") or []
    opens = payload.get("o") or []
    highs = payload.get("h") or []
    lows = payload.get("l") or []
    volumes = payload.get("v") or []
    timestamps = payload.get("t") or []

    size = min(len(closes), len(opens), len(highs), len(lows), len(volumes), len(timestamps))
    out: List[CandlestickPoint] = []
    for idx in range(size):
        close = _safe_float(closes[idx])
        if close is None:
            continue
        open_ = _safe_float(opens[idx]) or close
        high = _safe_float(highs[idx]) or close
        low = _safe_float(lows[idx]) or close
        volume = _safe_float(volumes[idx]) or 0.0
        ts = _safe_int(timestamps[idx])
        if ts is None:
            continue

        out.append(
            CandlestickPoint(
                timestamp=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                open=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=float(volume),
            )
        )
    return out


def _twelvedata_candles(symbol: str, period: str = "1mo", interval: str = "1d") -> List[CandlestickPoint]:
    days = _period_days(period)
    if interval == "1wk":
        outputsize = min(5000, max(30, (days // 7) + 24))
    elif interval in {"1mo", "3mo"}:
        outputsize = min(5000, max(24, (days // 30) + 24))
    else:
        outputsize = min(5000, max(60, days + 30))

    payload = _twelvedata_get(
        "/time_series",
        {
            "symbol": symbol,
            "interval": _twelvedata_interval(interval),
            "outputsize": outputsize,
            "format": "JSON",
        },
    )
    values = payload.get("values") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return []

    out: List[CandlestickPoint] = []
    # Twelve Data returns newest-first rows.
    for row in reversed(values):
        if not isinstance(row, dict):
            continue

        close = _safe_float(row.get("close"))
        if close is None:
            continue
        open_ = _safe_float(row.get("open")) or close
        high = _safe_float(row.get("high")) or close
        low = _safe_float(row.get("low")) or close
        volume = _safe_float(row.get("volume")) or 0.0
        timestamp = _parse_twelvedata_datetime(row.get("datetime"))
        if timestamp is None:
            continue

        out.append(
            CandlestickPoint(
                timestamp=timestamp,
                open=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=float(volume),
            )
        )

    return out


def _stooq_symbol(symbol: str) -> str:
    return f"{symbol.lower()}.us"


def _stooq_group_key(ts: datetime, interval: str) -> tuple[int, int]:
    if interval == "1mo":
        return ts.year, ts.month
    year, week, _ = ts.isocalendar()
    return year, week


def _stooq_candles(symbol: str, period: str = "1mo", interval: str = "1d") -> List[CandlestickPoint]:
    url = f"https://stooq.com/q/d/l/?s={_stooq_symbol(symbol)}&i=d"
    try:
        with httpx.Client(timeout=12.0) as client:
            resp = client.get(url)
            if resp.status_code >= 400:
                return []
            text = resp.text
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(text.splitlines()):
        if not isinstance(row, dict):
            continue
        raw_date = str(row.get("Date") or "").strip()
        try:
            date_value = datetime.strptime(raw_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        close = _safe_float(row.get("Close"))
        if close is None:
            continue

        open_ = _safe_float(row.get("Open")) or close
        high = _safe_float(row.get("High")) or close
        low = _safe_float(row.get("Low")) or close
        volume = _safe_float(row.get("Volume")) or 0.0

        rows.append(
            {
                "dt": date_value,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    if not rows:
        return []

    rows.sort(key=lambda item: item["dt"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=_period_days(period))
    filtered = [row for row in rows if row["dt"] >= cutoff]
    if not filtered:
        filtered = rows[-max(20, min(400, _period_days(period))):]

    if interval == "1d":
        return [
            CandlestickPoint(
                timestamp=row["dt"].isoformat(),
                open=round(float(row["open"]), 2),
                high=round(float(row["high"]), 2),
                low=round(float(row["low"]), 2),
                close=round(float(row["close"]), 2),
                volume=float(row["volume"]),
            )
            for row in filtered
        ]

    grouped: list[list[dict[str, Any]]] = []
    current_bucket: tuple[int, int] | None = None
    current_rows: list[dict[str, Any]] = []

    for row in filtered:
        bucket = _stooq_group_key(row["dt"], interval)
        if current_bucket is None or bucket == current_bucket:
            current_bucket = bucket
            current_rows.append(row)
            continue
        grouped.append(current_rows)
        current_bucket = bucket
        current_rows = [row]

    if current_rows:
        grouped.append(current_rows)

    out: list[CandlestickPoint] = []
    for bucket_rows in grouped:
        first = bucket_rows[0]
        last = bucket_rows[-1]
        high = max(float(item["high"]) for item in bucket_rows)
        low = min(float(item["low"]) for item in bucket_rows)
        volume = sum(float(item["volume"]) for item in bucket_rows)
        out.append(
            CandlestickPoint(
                timestamp=last["dt"].isoformat(),
                open=round(float(first["open"]), 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(float(last["close"]), 2),
                volume=volume,
            )
        )
    return out


def _alpaca_asset_by_symbol(symbol: str) -> dict[str, Any] | None:
    payload = _alpaca_get(f"/v2/assets/{symbol}", trading=True)
    return payload if isinstance(payload, dict) else None


def _finnhub_search(query: str, limit: int = 8) -> List[TickerLookup]:
    payload = _finnhub_get("/search", {"q": query})
    rows = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []

    out: List[TickerLookup] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = _normalize_lookup_symbol(row.get("symbol"))
        if not _is_symbol_candidate(symbol):
            continue
        if not symbol or symbol in seen:
            continue

        description = str(row.get("description") or symbol).strip() or symbol
        exchange = str(row.get("exchange") or row.get("mic") or "").strip() or None
        instrument_type = str(row.get("type") or "").strip() or None

        out.append(
            TickerLookup(
                ticker=symbol,
                name=description,
                exchange=exchange,
                instrument_type=instrument_type,
            )
        )
        seen.add(symbol)
        if len(out) >= limit:
            break

    return out


def _load_sec_company_directory() -> list[tuple[str, str]]:
    global _SEC_TICKER_CACHE
    now = datetime.now(timezone.utc)
    if _SEC_TICKER_CACHE and (now - _SEC_TICKER_CACHE[0]) < _SEC_TICKER_CACHE_TTL:
        return _SEC_TICKER_CACHE[1]

    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": "TickerMaster/1.0 (research@treehacks.dev)",
        "Accept": "application/json",
    }
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        with httpx.Client(timeout=15.0, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return _SEC_TICKER_CACHE[1] if _SEC_TICKER_CACHE else []

    source_rows = list(payload.values()) if isinstance(payload, dict) else payload
    if not isinstance(source_rows, list):
        return _SEC_TICKER_CACHE[1] if _SEC_TICKER_CACHE else []
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        symbol = _normalize_lookup_symbol(row.get("ticker"))
        if not _is_symbol_candidate(symbol) or symbol in seen:
            continue
        title = str(row.get("title") or symbol).strip() or symbol
        rows.append((symbol, title))
        seen.add(symbol)
    _SEC_TICKER_CACHE = (now, rows)
    return rows


def _sec_company_search(query: str, limit: int = 8) -> List[TickerLookup]:
    clean_query = str(query or "").strip()
    if not clean_query:
        return []
    directory = _load_sec_company_directory()
    if not directory:
        return []

    ranked: list[tuple[int, TickerLookup]] = []
    for symbol, title in directory:
        item = TickerLookup(
            ticker=symbol,
            name=title,
            exchange="US",
            instrument_type="Common Stock",
        )
        score = _lookup_score(clean_query, item)
        if score <= 0:
            continue
        ranked.append((score, item))

    ranked.sort(key=lambda pair: (-pair[0], len(pair[1].ticker), pair[1].ticker))
    out: list[TickerLookup] = []
    seen: set[str] = set()
    for _, item in ranked:
        if item.ticker in seen:
            continue
        out.append(item)
        seen.add(item.ticker)
        if len(out) >= limit:
            break
    return out


def _twelvedata_search(query: str, limit: int = 8) -> List[TickerLookup]:
    payload = _twelvedata_get(
        "/symbol_search",
        {"symbol": query, "outputsize": min(max(limit * 4, 12), 80)},
    )
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []

    out: List[TickerLookup] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = _normalize_lookup_symbol(row.get("symbol"))
        if not _is_symbol_candidate(symbol) or symbol in seen:
            continue
        name = str(row.get("instrument_name") or row.get("name") or symbol).strip() or symbol
        exchange = str(row.get("exchange") or "").strip() or None
        instrument_type = str(row.get("instrument_type") or "").strip() or None
        out.append(
            TickerLookup(
                ticker=symbol,
                name=name,
                exchange=exchange,
                instrument_type=instrument_type,
            )
        )
        seen.add(symbol)
        if len(out) >= limit:
            break
    return out


def _yahoo_search(query: str, limit: int = 8) -> List[TickerLookup]:
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_0) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        with httpx.Client(timeout=12.0, headers=headers) as client:
            resp = client.get(
                url,
                params={
                    "q": query,
                    "quotesCount": min(max(limit * 6, 20), 100),
                    "newsCount": 0,
                    "enableFuzzyQuery": "true",
                    "lang": "en-US",
                    "region": "US",
                },
            )
            if resp.status_code in {401, 403, 404, 429}:
                return []
            resp.raise_for_status()
            payload = resp.json()
    except Exception:
        return []

    rows = payload.get("quotes") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []

    out: List[TickerLookup] = []
    seen: set[str] = set()
    allowed_types = {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}
    for row in rows:
        if not isinstance(row, dict):
            continue
        quote_type = str(row.get("quoteType") or "").upper()
        if quote_type and quote_type not in allowed_types:
            continue
        symbol = _normalize_lookup_symbol(row.get("symbol"))
        if not _is_symbol_candidate(symbol) or symbol in seen:
            continue
        name = str(row.get("shortname") or row.get("longname") or symbol).strip() or symbol
        exchange = str(row.get("exchange") or row.get("exchDisp") or "").strip() or None
        out.append(
            TickerLookup(
                ticker=symbol,
                name=name,
                exchange=exchange,
                instrument_type=(quote_type or None),
            )
        )
        seen.add(symbol)
        if len(out) >= limit:
            break
    return out


def _search_symbol_providers(query: str, limit: int = 8) -> List[TickerLookup]:
    raw: List[TickerLookup] = []
    raw.extend(_sec_company_search(query, max(limit * 2, 16)))
    raw.extend(_finnhub_search(query, max(limit * 2, 8)))
    raw.extend(_twelvedata_search(query, max(limit * 2, 8)))
    raw.extend(_yahoo_search(query, max(limit * 2, 8)))
    return _rank_lookup_results(query, raw, limit=limit)


def _metric_from_finnhub(symbol: str) -> MarketMetric | None:
    quote = _finnhub_quote(symbol)
    if not quote:
        return None

    basic = _finnhub_basic_metrics(symbol)
    profile = _finnhub_profile(symbol)

    pe_ratio = _safe_float(basic.get("peTTM")) or _safe_float(basic.get("peBasicExclExtraTTM"))
    beta = _safe_float(basic.get("beta"))
    market_cap = _normalize_market_cap(basic.get("marketCapitalization")) or _normalize_market_cap(
        profile.get("marketCapitalization")
    )

    return MarketMetric(
        ticker=symbol,
        price=round(float(quote["price"]), 2),
        change_percent=round(float(quote["change_percent"]), 2),
        pe_ratio=round(pe_ratio, 2) if pe_ratio is not None and not math.isnan(pe_ratio) else None,
        beta=round(beta, 2) if beta is not None and not math.isnan(beta) else None,
        volume=_safe_int(quote.get("volume") or basic.get("10DayAverageTradingVolume")),
        market_cap=market_cap,
    )


def fetch_metric(ticker: str) -> MarketMetric:
    symbol = resolve_symbol_input(ticker)

    # Primary provider: Finnhub.
    finnhub_metric = _metric_from_finnhub(symbol)
    if finnhub_metric is not None:
        _LAST_METRIC_CACHE[symbol] = finnhub_metric
        return finnhub_metric

    # Backup provider: Alpaca (real-time snapshot).
    alpaca_metric = _alpaca_metric(symbol)
    if alpaca_metric is not None:
        basic = _finnhub_basic_metrics(symbol)
        profile = _finnhub_profile(symbol)
        pe_ratio = _safe_float(basic.get("peTTM")) or _safe_float(basic.get("peBasicExclExtraTTM"))
        beta = _safe_float(basic.get("beta"))
        market_cap = _normalize_market_cap(basic.get("marketCapitalization")) or _normalize_market_cap(
            profile.get("marketCapitalization")
        )

        out = MarketMetric(
            ticker=alpaca_metric.ticker,
            price=alpaca_metric.price,
            change_percent=alpaca_metric.change_percent,
            pe_ratio=round(pe_ratio, 2) if pe_ratio is not None and not math.isnan(pe_ratio) else None,
            beta=round(beta, 2) if beta is not None and not math.isnan(beta) else None,
            volume=alpaca_metric.volume,
            market_cap=market_cap,
        )
        _LAST_METRIC_CACHE[symbol] = out
        return out

    cached = _LAST_METRIC_CACHE.get(symbol)
    if cached is not None:
        return cached

    raise RuntimeError(f"No quote data available for {symbol}")


def fetch_watchlist_metrics(tickers: List[str]) -> List[MarketMetric]:
    items: List[MarketMetric] = []
    for ticker in tickers:
        try:
            items.append(fetch_metric(ticker))
        except Exception:
            continue
    return items


async def search_tickers(query: str, limit: int = 8) -> List[TickerLookup]:
    clean_query = query.strip()
    if not clean_query:
        return []

    out: List[TickerLookup] = []
    seen: set[str] = set()

    # Primary: Alpaca exact symbol lookup.
    symbol_candidate = clean_query.upper().strip()
    if symbol_candidate.replace(".", "").isalnum() and len(symbol_candidate) <= 8:
        asset = await asyncio.to_thread(_alpaca_asset_by_symbol, symbol_candidate)
        if isinstance(asset, dict):
            asset_symbol = str(asset.get("symbol") or "").upper().strip()
            if asset_symbol:
                out.append(
                    TickerLookup(
                        ticker=asset_symbol,
                        name=str(asset.get("name") or asset_symbol),
                        exchange=str(asset.get("exchange") or "").strip() or None,
                        instrument_type=str(asset.get("asset_class") or "").strip() or None,
                    )
                )
                seen.add(asset_symbol)

    # Backup: provider search API chain with ranking.
    if len(out) < limit:
        backup = await asyncio.to_thread(_search_symbol_providers, clean_query, max(limit * 2, 8))
        for item in backup:
            if item.ticker in seen:
                continue
            out.append(item)
            seen.add(item.ticker)
            if len(out) >= limit:
                break

    return out[:limit]


def fetch_candles(ticker: str, period: str = "1mo", interval: str = "1d") -> List[CandlestickPoint]:
    symbol = resolve_symbol_input(ticker)
    cache_key = f"{symbol}:{period}:{interval}"

    # Primary provider: Finnhub candles.
    finnhub = _finnhub_candles(symbol, period=period, interval=interval)
    if finnhub:
        _LAST_CANDLES_CACHE[cache_key] = finnhub
        return finnhub

    # Backup provider: Alpaca bars.
    alpaca = _alpaca_bars(symbol, period=period, interval=interval)
    if alpaca:
        _LAST_CANDLES_CACHE[cache_key] = alpaca
        return alpaca

    # Backup provider: Twelve Data candles.
    twelvedata = _twelvedata_candles(symbol, period=period, interval=interval)
    if twelvedata:
        _LAST_CANDLES_CACHE[cache_key] = twelvedata
        return twelvedata

    # Backup provider: Stooq historical bars (daily feed with local aggregation).
    stooq = _stooq_candles(symbol, period=period, interval=interval)
    if stooq:
        _LAST_CANDLES_CACHE[cache_key] = stooq
        return stooq

    cached = _LAST_CANDLES_CACHE.get(cache_key)
    if cached:
        return cached

    raise RuntimeError(f"No candle data available for {symbol}")


def fetch_sp500_returns_window(period: str = "10y") -> Tuple[np.ndarray, float]:
    bars = _alpaca_bars("SPY", period=period, interval="1d")
    if not bars:
        bars = _finnhub_candles("SPY", period=period, interval="1d")
    if not bars:
        bars = _twelvedata_candles("SPY", period=period, interval="1d")
    if not bars:
        bars = _stooq_candles("SPY", period=period, interval="1d")

    closes = [float(point.close) for point in bars if point.close is not None]

    if len(closes) < 3:
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=0.0004, scale=0.012, size=2500)
    else:
        arr = np.asarray(closes, dtype=float)
        returns = np.diff(arr) / arr[:-1]

    if len(returns) == 0:
        returns = np.array([0.0])

    annualized_vol = float(np.std(returns) * np.sqrt(252))
    return returns, annualized_vol


def _recommendation_label(row: dict[str, Any]) -> str | None:
    if not row:
        return None

    strong_buy = _safe_int(row.get("strongBuy")) or 0
    buy = _safe_int(row.get("buy")) or 0
    hold = _safe_int(row.get("hold")) or 0
    sell = _safe_int(row.get("sell")) or 0
    strong_sell = _safe_int(row.get("strongSell")) or 0

    bullish = strong_buy + buy
    bearish = strong_sell + sell

    if bullish > bearish + hold:
        return "buy"
    if bearish > bullish + hold:
        return "sell"
    if hold > 0:
        return "hold"
    return None


def fetch_advanced_stock_data(ticker: str) -> Dict[str, Any]:
    symbol = resolve_symbol_input(ticker)
    now_ts = datetime.now(timezone.utc).timestamp()
    cached = _LAST_ADVANCED_CACHE.get(symbol)
    if cached and (now_ts - cached[0]) <= _ADVANCED_CACHE_TTL_SECONDS:
        return cached[1]

    profile = _finnhub_profile(symbol)
    basic = _finnhub_basic_metrics(symbol)
    recommendation_row = _finnhub_recommendation(symbol)
    price_target = _finnhub_price_target(symbol)
    tw_stats = _twelvedata_statistics(symbol)
    yahoo_quote = _yahoo_quote(symbol)

    try:
        metric = fetch_metric(symbol)
    except Exception:
        quote = _finnhub_quote(symbol) or {}
        quote_price = _safe_float(quote.get("price"))
        metric = MarketMetric(
            ticker=symbol,
            price=round(quote_price, 2) if quote_price is not None else 0.0,
            change_percent=round(float(_safe_float(quote.get("change_percent")) or 0.0), 2),
            pe_ratio=None,
            beta=None,
            volume=_safe_int(quote.get("volume")),
            market_cap=None,
        )

    insider = _finnhub_insider_transactions(symbol)
    if not insider:
        insider = _sec_form4_fallback(symbol)

    trailing_pe = (
        _safe_float(basic.get("peTTM"))
        or _safe_float(basic.get("peBasicExclExtraTTM"))
        or metric.pe_ratio
        or _safe_float(tw_stats.get("pe_ratio"))
        or _safe_float(tw_stats.get("trailing_pe"))
        or _safe_float(yahoo_quote.get("trailingPE"))
    )
    forward_pe = (
        _safe_float(basic.get("peForward"))
        or _safe_float(tw_stats.get("forward_pe"))
        or _safe_float(yahoo_quote.get("forwardPE"))
    )
    eps_trailing = (
        _safe_float(basic.get("epsTTM"))
        or _safe_float(tw_stats.get("eps"))
        or _safe_float(tw_stats.get("eps_ttm"))
        or _safe_float(yahoo_quote.get("epsTrailingTwelveMonths"))
    )
    eps_forward = _safe_float(basic.get("epsForward")) or _safe_float(yahoo_quote.get("epsForward"))
    dividend_yield = _safe_float(basic.get("dividendYieldIndicatedAnnual"))
    fifty_two_week_high = _safe_float(basic.get("52WeekHigh")) or _safe_float(yahoo_quote.get("fiftyTwoWeekHigh"))
    fifty_two_week_low = _safe_float(basic.get("52WeekLow")) or _safe_float(yahoo_quote.get("fiftyTwoWeekLow"))
    avg_volume = _safe_float(basic.get("10DayAverageTradingVolume")) or _safe_float(
        basic.get("3MonthAverageTradingVolume")
    ) or _safe_float(yahoo_quote.get("averageDailyVolume3Month"))

    sec_shares_outstanding = _sec_latest_shares_outstanding(symbol)
    sec_eps = _sec_latest_eps_diluted(symbol)

    market_cap_value = (
        metric.market_cap
        or _normalize_market_cap(profile.get("marketCapitalization"))
        or _safe_float(tw_stats.get("market_capitalization"))
        or _safe_float(yahoo_quote.get("marketCap"))
    )
    if market_cap_value is None and sec_shares_outstanding is not None and metric.price and metric.price > 0:
        market_cap_value = float(sec_shares_outstanding) * float(metric.price)

    eps_trailing_value = eps_trailing or sec_eps
    trailing_pe_value = trailing_pe
    if trailing_pe_value is None and eps_trailing_value is not None and eps_trailing_value > 0 and metric.price and metric.price > 0:
        trailing_pe_value = float(metric.price) / float(eps_trailing_value)

    out = {
        "ticker": symbol,
        "current_price": metric.price if metric.price and metric.price > 0 else None,
        "change_percent": metric.change_percent if metric.price and metric.price > 0 else None,
        "company_name": (profile.get("name") or yahoo_quote.get("longName") or yahoo_quote.get("shortName") or symbol),
        "exchange": profile.get("exchange") or profile.get("mic") or yahoo_quote.get("fullExchangeName") or yahoo_quote.get("exchange"),
        "sector": profile.get("finnhubIndustry"),
        "industry": profile.get("finnhubIndustry"),
        "website": profile.get("weburl"),
        "description": profile.get("description"),
        "market_cap": market_cap_value,
        "beta": (
            metric.beta
            if metric.beta is not None
            else (_safe_float(basic.get("beta")) or _safe_float(yahoo_quote.get("beta")))
        ),
        "trailing_pe": trailing_pe_value,
        "forward_pe": forward_pe,
        "eps_trailing": eps_trailing_value,
        "eps_forward": eps_forward,
        "dividend_yield": dividend_yield,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
        "avg_volume": avg_volume,
        "volume": metric.volume if metric.volume is not None else _safe_int(yahoo_quote.get("regularMarketVolume")),
        "recommendation": _recommendation_label(recommendation_row) or str(yahoo_quote.get("recommendationKey") or "").strip().lower() or None,
        "target_mean_price": (
            _safe_float(price_target.get("targetMean"))
            or _safe_float(tw_stats.get("target_price"))
            or _safe_float(yahoo_quote.get("targetMeanPrice"))
        ),
        "insider_transactions": insider,
    }
    key_fields = [
        out.get("market_cap"),
        out.get("trailing_pe"),
        out.get("forward_pe"),
        out.get("eps_trailing"),
        out.get("target_mean_price"),
    ]
    if any(value is not None for value in key_fields):
        _LAST_ADVANCED_CACHE[symbol] = (now_ts, out)
        return out
    if cached:
        return cached[1]
    return out


def _sec_form4_fallback(symbol: str) -> list[dict[str, Any]]:
    ua = {"User-Agent": "TickerMaster/1.0 (research@treehacks.dev)"}
    try:
        with httpx.Client(timeout=12.0) as client:
            tickers = client.get("https://www.sec.gov/files/company_tickers.json", headers=ua).json()
            cik = None
            if isinstance(tickers, dict):
                for value in tickers.values():
                    if isinstance(value, dict) and str(value.get("ticker", "")).upper() == symbol:
                        cik_num = value.get("cik_str")
                        cik = f"{int(cik_num):010d}" if cik_num is not None else None
                        break
            if not cik:
                return []
            submissions = client.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=ua).json()
            recent = (submissions.get("filings") or {}).get("recent") or {}
            forms = recent.get("form") or []
            dates = recent.get("filingDate") or []
            accessions = recent.get("accessionNumber") or []
            out: list[dict[str, Any]] = []
            for idx, form in enumerate(forms):
                if str(form) != "4":
                    continue
                out.append(
                    {
                        "start_date": dates[idx] if idx < len(dates) else None,
                        "filer_name": "SEC Form 4",
                        "filer_relation": "Insider Filing",
                        "money_text": "Ownership change disclosure",
                        "shares": None,
                        "value": None,
                        "ownership": accessions[idx] if idx < len(accessions) else "",
                    }
                )
                if len(out) >= 20:
                    break
            return out
    except Exception:
        return []


def _sec_cik_for_symbol(symbol: str) -> str | None:
    ua = {"User-Agent": "TickerMaster/1.0 (research@treehacks.dev)"}
    try:
        with httpx.Client(timeout=12.0) as client:
            tickers = client.get("https://www.sec.gov/files/company_tickers.json", headers=ua).json()
            if not isinstance(tickers, dict):
                return None
            for value in tickers.values():
                if not isinstance(value, dict):
                    continue
                if str(value.get("ticker", "")).upper() != symbol:
                    continue
                cik_num = value.get("cik_str")
                if cik_num is None:
                    return None
                return f"{int(cik_num):010d}"
    except Exception:
        return None
    return None


def _sec_company_facts(cik: str) -> dict[str, Any]:
    ua = {"User-Agent": "TickerMaster/1.0 (research@treehacks.dev)"}
    try:
        with httpx.Client(timeout=12.0) as client:
            payload = client.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=ua).json()
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _sec_latest_numeric_fact(facts: dict[str, Any], taxonomy: str, tag: str, units: list[str]) -> float | None:
    taxonomy_block = facts.get("facts", {}).get(taxonomy, {})
    tag_block = taxonomy_block.get(tag, {}) if isinstance(taxonomy_block, dict) else {}
    units_block = tag_block.get("units", {}) if isinstance(tag_block, dict) else {}
    if not isinstance(units_block, dict):
        return None

    latest_value: float | None = None
    latest_sort_key = ""
    for unit in units:
        rows = units_block.get(unit)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = _safe_float(row.get("val"))
            if value is None:
                continue
            filed = str(row.get("filed") or "")
            end = str(row.get("end") or "")
            sort_key = f"{filed}|{end}"
            if sort_key >= latest_sort_key:
                latest_sort_key = sort_key
                latest_value = value
    return latest_value


def _sec_latest_shares_outstanding(symbol: str) -> float | None:
    cik = _sec_cik_for_symbol(symbol)
    if not cik:
        return None
    facts = _sec_company_facts(cik)
    return (
        _sec_latest_numeric_fact(facts, "dei", "EntityCommonStockSharesOutstanding", ["shares"])
        or _sec_latest_numeric_fact(facts, "us-gaap", "CommonStockSharesOutstanding", ["shares"])
    )


def _sec_latest_eps_diluted(symbol: str) -> float | None:
    cik = _sec_cik_for_symbol(symbol)
    if not cik:
        return None
    facts = _sec_company_facts(cik)
    return _sec_latest_numeric_fact(facts, "us-gaap", "EarningsPerShareDiluted", ["USD/shares"])
