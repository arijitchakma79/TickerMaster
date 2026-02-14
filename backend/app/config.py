from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List


def _load_dotenv(path: str = ".env") -> None:
    candidates = [
        Path(path),
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[3] / ".env",
    ]
    seen: set[Path] = set()
    for env_path in candidates:
        if env_path in seen or not env_path.exists():
            continue
        seen.add(env_path)
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [token.strip() for token in raw.split(",") if token.strip()] or default


@dataclass
class Settings:
    app_name: str = "TickerMaster API"
    environment: str = "development"
    log_level: str = "INFO"

    frontend_origins: List[str] = field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])

    supabase_url: str = ""
    supabase_key: str = ""
    supabase_service_key: str = ""
    database_url: str = ""

    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_data_url: str = "https://data.alpaca.markets"
    alpaca_trading_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_feed: str = "iex"

    finnhub_api_key: str = ""
    finnhub_api_url: str = "https://finnhub.io/api/v1"

    openai_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct"

    perplexity_api_key: str = ""
    perplexity_model: str = "sonar"

    x_api_bearer_token: str = ""
    x_consumer_key: str = ""
    x_consumer_secret: str = ""
    x_access_token: str = ""
    x_access_token_secret: str = ""

    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "TickerMaster/1.0"

    kalshi_api_key: str = ""
    kalshi_api_secret: str = ""

    polymarket_api_key: str = ""
    polymarket_clob_url: str = "https://clob.polymarket.com"
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"

    fred_api_key: str = ""

    browserbase_api_key: str = ""
    browserbase_project_id: str = ""

    modal_token_id: str = ""
    modal_token_secret: str = ""

    poke_recipe_enabled: bool = True
    poke_api_key: str = ""
    poke_recipe_slug: str = ""
    poke_kitchen_url: str = "https://poke.com/kitchen"

    cerebras_api_key: str = ""
    nvidia_nim_api_key: str = ""

    tracker_poll_interval_seconds: int = 60
    default_watchlist: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        app_name=_env("APP_NAME", "TickerMaster API"),
        environment=_env("ENVIRONMENT", "development"),
        log_level=_env("LOG_LEVEL", "INFO"),
        frontend_origins=_env_list("FRONTEND_ORIGINS", ["http://localhost:5173", "http://localhost:3000"]),
        supabase_url=_env("SUPABASE_URL"),
        supabase_key=_env("SUPABASE_KEY"),
        supabase_service_key=_env("SUPABASE_SERVICE_KEY"),
        database_url=_env("DATABASE_URL"),
        alpaca_api_key=_env("ALPACA_API_KEY") or _env("APCA_API_KEY_ID"),
        alpaca_api_secret=_env("ALPACA_API_SECRET") or _env("APCA_API_SECRET_KEY"),
        alpaca_data_url=_env("ALPACA_DATA_URL", "https://data.alpaca.markets"),
        alpaca_trading_url=_env("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets"),
        alpaca_data_feed=_env("ALPACA_DATA_FEED", "iex"),
        finnhub_api_key=_env("FINNHUB_API_KEY"),
        finnhub_api_url=_env("FINNHUB_API_URL", "https://finnhub.io/api/v1"),
        openai_api_key=_env("OPENAI_API_KEY"),
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        openrouter_model=_env("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        perplexity_api_key=_env("PERPLEXITY_API_KEY"),
        perplexity_model=_env("PERPLEXITY_MODEL", "sonar"),
        x_api_bearer_token=_env("X_API_BEARER_TOKEN") or _env("X_BEARER_TOKEN"),
        x_consumer_key=_env("X_CONSUMER_KEY"),
        x_consumer_secret=_env("X_CONSUMER_SECRET"),
        x_access_token=_env("X_ACCESS_TOKEN"),
        x_access_token_secret=_env("X_ACCESS_TOKEN_SECRET"),
        reddit_client_id=_env("REDDIT_CLIENT_ID"),
        reddit_client_secret=_env("REDDIT_CLIENT_SECRET"),
        reddit_user_agent=_env("REDDIT_USER_AGENT", "TickerMaster/1.0"),
        kalshi_api_key=_env("KALSHI_API_KEY"),
        kalshi_api_secret=_env("KALSHI_API_SECRET"),
        polymarket_api_key=_env("POLYMARKET_API_KEY"),
        polymarket_clob_url=_env("POLYMARKET_CLOB_URL", "https://clob.polymarket.com"),
        polymarket_gamma_url=_env("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com"),
        fred_api_key=_env("FRED_API_KEY"),
        browserbase_api_key=_env("BROWSERBASE_API_KEY"),
        browserbase_project_id=_env("BROWSERBASE_PROJECT_ID"),
        modal_token_id=_env("MODAL_TOKEN_ID"),
        modal_token_secret=_env("MODAL_TOKEN_SECRET"),
        poke_recipe_enabled=_env_bool("POKE_RECIPE_ENABLED", True),
        poke_api_key=_env("POKE_API_KEY"),
        poke_recipe_slug=_env("POKE_RECIPE_SLUG"),
        poke_kitchen_url=_env("POKE_KITCHEN_URL", "https://poke.com/kitchen"),
        cerebras_api_key=_env("CEREBRAS_API_KEY"),
        nvidia_nim_api_key=_env("NVIDIA_NIM_API_KEY"),
        tracker_poll_interval_seconds=_env_int("TRACKER_POLL_INTERVAL_SECONDS", 60),
        default_watchlist=_env_list("DEFAULT_WATCHLIST", ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]),
    )
