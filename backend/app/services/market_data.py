from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np
import yfinance as yf

from app.schemas import CandlestickPoint, MarketMetric, TickerLookup

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
YAHOO_ALLOWED_TYPES = {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}


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


def fetch_metric(ticker: str) -> MarketMetric:
    symbol = ticker.upper().strip()
    asset = yf.Ticker(symbol)

    history = asset.history(period="2d", interval="1d", auto_adjust=False)
    info = asset.info if hasattr(asset, "info") else {}

    if history.empty:
        # Offline fallback with deterministic synthetic numbers by ticker hash
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

    closes = history["Close"].dropna().tolist()
    latest_price = float(closes[-1])
    previous_close = float(closes[-2]) if len(closes) > 1 else latest_price
    change_percent = ((latest_price - previous_close) / previous_close * 100) if previous_close else 0

    pe_ratio = _safe_float(info.get("trailingPE"))
    beta = _safe_float(info.get("beta"))
    volume = _safe_int(info.get("volume"))
    market_cap = _safe_float(info.get("marketCap"))

    return MarketMetric(
        ticker=symbol,
        price=round(latest_price, 2),
        change_percent=round(change_percent, 2),
        pe_ratio=round(pe_ratio, 2) if pe_ratio is not None and not math.isnan(pe_ratio) else None,
        beta=round(beta, 2) if beta is not None and not math.isnan(beta) else None,
        volume=volume,
        market_cap=market_cap,
    )


def fetch_watchlist_metrics(tickers: List[str]) -> List[MarketMetric]:
    return [fetch_metric(ticker) for ticker in tickers]


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
    frame = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)

    if frame.empty:
        now = datetime.now(timezone.utc)
        baseline = abs(hash(symbol)) % 300 + 50
        points: List[CandlestickPoint] = []
        for idx in range(30):
            px = baseline + np.sin(idx / 4) * 4
            points.append(
                CandlestickPoint(
                    timestamp=(now.replace(microsecond=0)).isoformat(),
                    open=round(px - 0.5, 2),
                    high=round(px + 1.3, 2),
                    low=round(px - 1.2, 2),
                    close=round(px + 0.6, 2),
                    volume=2_000_000 + idx * 12000,
                )
            )
        return points

    points = []
    for ts, row in frame.iterrows():
        points.append(
            CandlestickPoint(
                timestamp=ts.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
                open=round(float(row["Open"]), 2),
                high=round(float(row["High"]), 2),
                low=round(float(row["Low"]), 2),
                close=round(float(row["Close"]), 2),
                volume=float(row["Volume"]),
            )
        )
    return points


def fetch_sp500_returns_window(period: str = "10y") -> Tuple[np.ndarray, float]:
    frame = yf.download("^GSPC", period=period, interval="1d", progress=False, auto_adjust=False)
    if frame.empty:
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=0.0004, scale=0.012, size=2500)
    else:
        close = frame["Close"].dropna()
        returns = close.pct_change().dropna().to_numpy()

    if len(returns) == 0:
        returns = np.array([0.0])

    annualized_vol = float(np.std(returns) * np.sqrt(252))
    return returns, annualized_vol
