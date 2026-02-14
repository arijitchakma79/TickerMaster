from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import ResearchRequest
from app.services.market_data import fetch_candles
from app.services.sentiment import run_research

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/analyze")
async def analyze_research(payload: ResearchRequest, request: Request):
    settings = request.app.state.settings
    return await run_research(payload, settings)


@router.get("/candles/{ticker}")
async def candles(ticker: str, period: str = "1mo", interval: str = "1d"):
    return {"ticker": ticker.upper(), "points": fetch_candles(ticker, period=period, interval=interval)}
