from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.schemas import ResearchRequest
from app.services.market_data import fetch_candles, fetch_metric, search_tickers
from app.services.sentiment import run_research

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/analyze")
async def analyze_research(payload: ResearchRequest, request: Request):
    settings = request.app.state.settings
    return await run_research(payload, settings)


@router.get("/candles/{ticker}")
async def candles(ticker: str, period: str = "1mo", interval: str = "1d"):
    return {"ticker": ticker.upper(), "points": fetch_candles(ticker, period=period, interval=interval)}


@router.get("/quote/{ticker}")
async def quote(ticker: str):
    return fetch_metric(ticker).model_dump()


@router.get("/search/tickers")
async def ticker_lookup(
    query: str = Query(..., min_length=1, max_length=80),
    limit: int = Query(8, ge=1, le=20),
):
    results = await search_tickers(query, limit=limit)
    return {"query": query, "results": [item.model_dump() for item in results]}
