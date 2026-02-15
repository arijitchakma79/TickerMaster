from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SourceLink(BaseModel):
    source: str
    title: str
    url: str


class MarketMetric(BaseModel):
    ticker: str
    price: float
    change_percent: float
    pe_ratio: Optional[float] = None
    beta: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None


class TickerLookup(BaseModel):
    ticker: str
    name: str
    exchange: Optional[str] = None
    instrument_type: Optional[str] = None


class CandlestickPoint(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class ResearchRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=12)
    timeframe: str = "7d"
    include_prediction_markets: bool = True


class SentimentBreakdown(BaseModel):
    source: str
    sentiment: Literal["bullish", "neutral", "bearish"]
    score: float
    summary: str
    links: List[SourceLink] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    ticker: str
    generated_at: str
    aggregate_sentiment: float
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    narratives: List[str]
    source_breakdown: List[SentimentBreakdown]
    prediction_markets: List[Dict[str, Any]] = Field(default_factory=list)
    tool_links: List[SourceLink] = Field(default_factory=list)


class AgentConfig(BaseModel):
    name: str
    personality: Literal["quant_momentum", "fundamental_value", "retail_reactive"]
    model: str = "meta-llama/llama-3.1-8b-instruct"
    strategy_prompt: str = Field(default="", max_length=1200)
    aggressiveness: float = Field(0.5, ge=0, le=1)
    risk_limit: float = Field(0.5, ge=0, le=1)
    trade_size: int = Field(10, ge=1, le=1000)
    active: bool = True


class SimulationStartRequest(BaseModel):
    ticker: str = "AAPL"
    target_tickers: List[str] = Field(default_factory=list)
    user_id: Optional[str] = None
    duration_seconds: int = Field(180, ge=30, le=3600)
    initial_price: float = Field(185.0, gt=0)
    starting_cash: float = Field(100_000, gt=0)
    volatility: float = Field(0.02, ge=0.001, le=0.3)
    inference_runtime: Literal["direct", "modal"] = "direct"
    agents: List[AgentConfig] = Field(default_factory=list)


class TradeRecord(BaseModel):
    timestamp: str
    session_id: str
    ticker: str
    side: Literal["buy", "sell"]
    agent: str
    quantity: int
    price: float
    slippage_bps: float
    rationale: str


class SimulationState(BaseModel):
    session_id: str
    ticker: str
    tickers: List[str] = Field(default_factory=list)
    running: bool
    paused: bool
    tick: int
    current_price: float
    market_prices: Dict[str, float] = Field(default_factory=dict)
    volatility: float
    started_at: str
    ends_at: str
    crash_mode: bool
    recent_news: List[str] = Field(default_factory=list)
    trades: List[TradeRecord] = Field(default_factory=list)
    portfolios: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    order_book: Dict[str, List[Dict[str, float]]] = Field(default_factory=dict)


class AlertConfig(BaseModel):
    ticker: str
    threshold_percent: float = Field(2.0, ge=0.1, le=25)
    direction: Literal["up", "down", "either"] = "either"


class TrackerWatchlistRequest(BaseModel):
    tickers: List[str]


class TrackerSnapshot(BaseModel):
    generated_at: datetime
    tickers: List[MarketMetric]
    alerts_triggered: List[Dict[str, Any]] = Field(default_factory=list)


class ChatRequest(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    model: str
    generated_at: str


class ResearchChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    ticker: Optional[str] = None
    timeframe: str = "7d"
    include_deep: bool = False
    auto_fetch_if_missing: bool = True


class ResearchChatResponse(BaseModel):
    ticker: str
    response: str
    model: str
    generated_at: str
    context_refreshed: bool = True
    sources: List[str] = Field(default_factory=list)
