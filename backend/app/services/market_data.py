from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np

from app.schemas import CandlestickPoint, MarketMetric, TickerLookup

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
YAHOO_ALLOWED_TYPES = {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}

_LAST_METRIC_CACHE: dict[str, MarketMetric] = {}
_LAST_CANDLES_CACHE: dict[str, List[CandlestickPoint]] = {}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _yahoo_quote(symbol: str) -> Dict[str, Any] | None:
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": symbol}
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            results = resp.json().get("quoteResponse", {}).get("result", [])
            if results:
                return results[0]
    except Exception:
        return None
    return None


def _normalize_interval(interval: str) -> str:
    allowed = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}
    return interval if interval in allowed else "1d"


def _normalize_range(period: str) -> str:
    allowed = {
        "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
    }
    return period if period in allowed else "1mo"


def _yahoo_chart(symbol: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any] | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": _normalize_range(period), "interval": _normalize_interval(interval)}
    try:
        with httpx.Client(timeout=12.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json().get("chart", {})
    except Exception:
        return None


def _yahoo_quote_summary(symbol: str, modules: list[str]) -> Dict[str, Any] | None:
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
    params = {"modules": ",".join(modules)}
    try:
        with httpx.Client(timeout=14.0) as client:
            resp = client.get(url, params=params, headers={"User-Agent": "TickerMaster/1.0 (research@treehacks.dev)"})
            resp.raise_for_status()
            data = resp.json().get("quoteSummary", {})
            results = data.get("result") or []
            if results and isinstance(results[0], dict):
                return results[0]
    except Exception:
        return None
    return None


def _stooq_candles(symbol: str, period: str = "1mo") -> List[CandlestickPoint]:
    stooq_symbol = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    try:
        with httpx.Client(timeout=12.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
    except Exception:
        return []

    if len(lines) <= 1:
        return []

    period_map = {
        "5d": 5,
        "1mo": 22,
        "3mo": 66,
        "6mo": 132,
        "1y": 252,
        "2y": 504,
        "5y": 1260,
    }
    take = period_map.get(period, 66)

    out: List[CandlestickPoint] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        date_s, open_s, high_s, low_s, close_s, vol_s = parts[:6]
        try:
            dt = datetime.fromisoformat(date_s).replace(tzinfo=timezone.utc)
            out.append(
                CandlestickPoint(
                    timestamp=dt.isoformat(),
                    open=round(float(open_s), 2),
                    high=round(float(high_s), 2),
                    low=round(float(low_s), 2),
                    close=round(float(close_s), 2),
                    volume=float(vol_s),
                )
            )
        except Exception:
            continue

    return out[-take:] if out else []


def _stooq_latest_quote(symbol: str) -> Dict[str, Any] | None:
    points = _stooq_candles(symbol, period="1mo")
    if not points:
        return None
    latest = points[-1]
    prev = points[-2] if len(points) > 1 else latest
    prev_close = prev.close or latest.close
    change = ((latest.close - prev_close) / prev_close * 100) if prev_close else 0.0
    return {
        "price": latest.close,
        "change_percent": change,
        "volume": int(latest.volume),
    }


def fetch_metric(ticker: str) -> MarketMetric:
    symbol = ticker.upper().strip()
    quote = _yahoo_quote(symbol)
    if not quote:
        stooq = _stooq_latest_quote(symbol)
        if stooq:
            out = MarketMetric(
                ticker=symbol,
                price=round(float(stooq["price"]), 2),
                change_percent=round(float(stooq["change_percent"]), 2),
                pe_ratio=None,
                beta=None,
                volume=_safe_int(stooq.get("volume")),
                market_cap=None,
            )
            _LAST_METRIC_CACHE[symbol] = out
            return out
        cached = _LAST_METRIC_CACHE.get(symbol)
        if cached:
            return cached
        raise RuntimeError(f"No quote data available for {symbol}")

    price = _safe_float(quote.get("regularMarketPrice"))
    prev = _safe_float(quote.get("regularMarketPreviousClose")) or price
    if price is None:
        stooq = _stooq_latest_quote(symbol)
        if stooq:
            out = MarketMetric(
                ticker=symbol,
                price=round(float(stooq["price"]), 2),
                change_percent=round(float(stooq["change_percent"]), 2),
                pe_ratio=None,
                beta=None,
                volume=_safe_int(stooq.get("volume")),
                market_cap=None,
            )
            _LAST_METRIC_CACHE[symbol] = out
            return out
        cached = _LAST_METRIC_CACHE.get(symbol)
        if cached:
            return cached
        raise RuntimeError(f"No price data available for {symbol}")

    change_percent = ((price - prev) / prev * 100) if prev else 0.0
    pe_ratio = _safe_float(quote.get("trailingPE"))
    beta = _safe_float(quote.get("beta"))
    volume = _safe_int(quote.get("regularMarketVolume"))
    market_cap = _safe_float(quote.get("marketCap"))

    out = MarketMetric(
        ticker=symbol,
        price=round(price, 2),
        change_percent=round(change_percent, 2),
        pe_ratio=round(pe_ratio, 2) if pe_ratio is not None and not math.isnan(pe_ratio) else None,
        beta=round(beta, 2) if beta is not None and not math.isnan(beta) else None,
        volume=volume,
        market_cap=market_cap,
    )
    _LAST_METRIC_CACHE[symbol] = out
    return out


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

    params = {
        "q": clean_query,
        "quotesCount": max(1, min(limit * 3, 50)),
        "newsCount": 0,
        "enableFuzzyQuery": "true",
    }
    headers = {
        "User-Agent": "TickerMaster/1.0 (+https://tickermaster.local)",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
            response = await client.get(YAHOO_SEARCH_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return []

    seen = set()
    out: List[TickerLookup] = []
    for item in payload.get("quotes", []):
        symbol = str(item.get("symbol", "")).upper().strip()
        if not symbol or symbol in seen:
            continue

        quote_type = str(item.get("quoteType", "")).upper().strip()
        if quote_type and quote_type not in YAHOO_ALLOWED_TYPES:
            continue

        name = item.get("shortname") or item.get("longname") or symbol
        if not isinstance(name, str) or not name.strip():
            name = symbol

        exchange = item.get("exchDisp") or item.get("exchange")
        exchange_name = str(exchange).strip() if isinstance(exchange, str) else None
        instrument_type = quote_type or None

        out.append(
            TickerLookup(
                ticker=symbol,
                name=name.strip(),
                exchange=exchange_name or None,
                instrument_type=instrument_type,
            )
        )
        seen.add(symbol)
        if len(out) >= limit:
            break

    return out


def fetch_candles(ticker: str, period: str = "1mo", interval: str = "1d") -> List[CandlestickPoint]:
    symbol = ticker.upper().strip()
    cache_key = f"{symbol}:{period}:{interval}"
    chart = _yahoo_chart(symbol, period=period, interval=interval)
    if not chart:
        stooq = _stooq_candles(symbol, period=period)
        if stooq:
            _LAST_CANDLES_CACHE[cache_key] = stooq
            return stooq
        cached = _LAST_CANDLES_CACHE.get(cache_key)
        if cached:
            return cached
        raise RuntimeError(f"No candle data available for {symbol}")

    results = chart.get("result") or []
    if not results:
        stooq = _stooq_candles(symbol, period=period)
        if stooq:
            _LAST_CANDLES_CACHE[cache_key] = stooq
            return stooq
        cached = _LAST_CANDLES_CACHE.get(cache_key)
        if cached:
            return cached
        raise RuntimeError(f"No candle data available for {symbol}")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    points: List[CandlestickPoint] = []
    for idx, ts in enumerate(timestamps):
        if idx >= len(closes):
            break
        c = closes[idx]
        if c is None:
            continue
        o = opens[idx] if idx < len(opens) and opens[idx] is not None else c
        h = highs[idx] if idx < len(highs) and highs[idx] is not None else c
        l = lows[idx] if idx < len(lows) and lows[idx] is not None else c
        v = volumes[idx] if idx < len(volumes) and volumes[idx] is not None else 0
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        points.append(
            CandlestickPoint(
                timestamp=dt.isoformat(),
                open=round(float(o), 2),
                high=round(float(h), 2),
                low=round(float(l), 2),
                close=round(float(c), 2),
                volume=float(v),
            )
        )

    if points:
        _LAST_CANDLES_CACHE[cache_key] = points
        return points
    stooq = _stooq_candles(symbol, period=period)
    if stooq:
        _LAST_CANDLES_CACHE[cache_key] = stooq
        return stooq
    cached = _LAST_CANDLES_CACHE.get(cache_key)
    if cached:
        return cached
    raise RuntimeError(f"No candle data available for {symbol}")


def fetch_sp500_returns_window(period: str = "10y") -> Tuple[np.ndarray, float]:
    chart = _yahoo_chart("^GSPC", period=period, interval="1d")
    closes: List[float] = []

    if chart and chart.get("result"):
        result = chart["result"][0]
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        closes = [float(c) for c in (quote.get("close") or []) if c is not None]

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


def fetch_advanced_stock_data(ticker: str) -> Dict[str, Any]:
    symbol = ticker.upper().strip()
    modules = [
        "price",
        "summaryDetail",
        "financialData",
        "defaultKeyStatistics",
        "summaryProfile",
        "calendarEvents",
        "recommendationTrend",
        "insiderTransactions",
        "majorHoldersBreakdown",
        "upgradeDowngradeHistory",
    ]
    summary = _yahoo_quote_summary(symbol, modules) or {}
    quote = _yahoo_quote(symbol) or {}

    insider: list[dict[str, Any]] = []

    if not insider:
        insider_raw = ((summary.get("insiderTransactions") or {}).get("transactions") or [])
        for row in insider_raw[:30]:
            if not isinstance(row, dict):
                continue
            insider.append(
                {
                    "start_date": ((row.get("startDate") or {}).get("fmt") if isinstance(row.get("startDate"), dict) else None),
                    "filer_name": (row.get("filerName") or row.get("filer") or ""),
                    "filer_relation": (row.get("filerRelation") or ""),
                    "money_text": (row.get("moneyText") or ""),
                    "shares": ((row.get("shares") or {}).get("raw") if isinstance(row.get("shares"), dict) else row.get("shares")),
                    "value": ((row.get("value") or {}).get("raw") if isinstance(row.get("value"), dict) else row.get("value")),
                    "ownership": (row.get("ownership") or ""),
                }
            )
    if not insider:
        insider = _sec_form4_fallback(symbol)

    def _get_fmt(section: str, key: str) -> Any:
        block = summary.get(section) or {}
        value = block.get(key)
        if isinstance(value, dict):
            return value.get("raw", value.get("fmt"))
        return value

    return {
        "ticker": symbol,
        "company_name": (quote.get("longName") or quote.get("shortName") or symbol),
        "exchange": quote.get("fullExchangeName"),
        "sector": (summary.get("summaryProfile") or {}).get("sector"),
        "industry": (summary.get("summaryProfile") or {}).get("industry"),
        "website": (summary.get("summaryProfile") or {}).get("website"),
        "description": (summary.get("summaryProfile") or {}).get("longBusinessSummary"),
        "market_cap": (_get_fmt("summaryDetail", "marketCap") or quote.get("marketCap")),
        "beta": _get_fmt("summaryDetail", "beta"),
        "trailing_pe": _get_fmt("summaryDetail", "trailingPE"),
        "forward_pe": _get_fmt("summaryDetail", "forwardPE"),
        "eps_trailing": _get_fmt("defaultKeyStatistics", "trailingEps"),
        "eps_forward": _get_fmt("defaultKeyStatistics", "forwardEps"),
        "dividend_yield": _get_fmt("summaryDetail", "dividendYield"),
        "fifty_two_week_high": _get_fmt("summaryDetail", "fiftyTwoWeekHigh"),
        "fifty_two_week_low": _get_fmt("summaryDetail", "fiftyTwoWeekLow"),
        "avg_volume": _get_fmt("summaryDetail", "averageVolume"),
        "volume": quote.get("regularMarketVolume"),
        "recommendation": _get_fmt("financialData", "recommendationKey"),
        "target_mean_price": _get_fmt("financialData", "targetMeanPrice"),
        "insider_transactions": insider,
    }


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
