from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List
from urllib.parse import urlparse


def _load_dotenv(path: str = ".env") -> None:
    candidates = [Path(path)]
    resolved = Path(__file__).resolve()
    # Probe parent directories safely without assuming a fixed depth.
    for parent in resolved.parents:
        candidates.append(parent / ".env")
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
    supabase_avatar_bucket: str = "avatars"
    supabase_tracker_exports_bucket: str = "tracker-exports"
    supabase_tracker_memory_bucket: str = "tracker-memory"
    database_url: str = ""

    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_data_url: str = "https://data.alpaca.markets"
    alpaca_trading_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_feed: str = "iex"

    finnhub_api_key: str = ""
    finnhub_api_url: str = "https://finnhub.io/api/v1"
    twelvedata_api_key: str = ""
    twelvedata_api_url: str = "https://api.twelvedata.com"

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
    x_api_enabled: bool = True
    x_api_max_results: int = 10
    x_api_timeout_seconds: int = 10
    x_api_cache_ttl_seconds: int = 300
    x_api_max_calls_per_window: int = 20
    x_api_window_seconds: int = 900
    x_api_min_seconds_between_calls: int = 2
    x_api_allow_oauth1_fallback: bool = False

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
    modal_simulation_app_name: str = "tickermaster-simulation"
    modal_sandbox_timeout_seconds: int = 600
    modal_sandbox_idle_timeout_seconds: int = 120
    modal_sandbox_python_version: str = "3.11"
    modal_inference_function_name: str = "agent_inference"
    modal_inference_timeout_seconds: int = 15

    poke_recipe_enabled: bool = True
    poke_api_key: str = ""
    poke_recipe_slug: str = ""
    poke_api_url: str = "https://poke.com/api/v1"
    poke_kitchen_url: str = "https://poke.com/kitchen"

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_whatsapp_from_number: str = ""
    twilio_use_whatsapp: bool = False
    twilio_default_to_number: str = ""

    cerebras_api_key: str = ""
    nvidia_nim_api_key: str = ""

    tracker_poll_interval_seconds: int = 300
    default_watchlist: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    mcp_tracker_router_enabled: bool = True
    mcp_tracker_server_command: str = ""
    mcp_tracker_timeout_seconds: int = 6


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_settings(settings: Settings) -> None:
    if settings.environment.lower() == "production":
        required = {
            "SUPABASE_URL": settings.supabase_url,
            "SUPABASE_KEY": settings.supabase_key,
            "SUPABASE_SERVICE_KEY": settings.supabase_service_key,
        }
        missing = [key for key, value in required.items() if not str(value or "").strip()]
        if missing:
            raise RuntimeError(f"Missing required production environment variables: {', '.join(sorted(missing))}")

    invalid_origins = [origin for origin in settings.frontend_origins if not _is_http_url(origin)]
    if invalid_origins:
        raise RuntimeError(f"Invalid FRONTEND_ORIGINS entries: {', '.join(invalid_origins)}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    settings = Settings(
        app_name=_env("APP_NAME", "TickerMaster API"),
        environment=_env("ENVIRONMENT", "development"),
        log_level=_env("LOG_LEVEL", "INFO"),
        frontend_origins=_env_list("FRONTEND_ORIGINS", ["http://localhost:5173", "http://localhost:3000"]),
        supabase_url=_env("SUPABASE_URL"),
        supabase_key=_env("SUPABASE_KEY"),
        supabase_service_key=_env("SUPABASE_SERVICE_KEY"),
        supabase_avatar_bucket=_env("SUPABASE_AVATAR_BUCKET", "avatars"),
        supabase_tracker_exports_bucket=_env("SUPABASE_TRACKER_EXPORTS_BUCKET", "tracker-exports"),
        supabase_tracker_memory_bucket=_env("SUPABASE_TRACKER_MEMORY_BUCKET", "tracker-memory"),
        database_url=_env("DATABASE_URL"),
        alpaca_api_key=_env("ALPACA_API_KEY") or _env("APCA_API_KEY_ID"),
        alpaca_api_secret=_env("ALPACA_API_SECRET") or _env("APCA_API_SECRET_KEY"),
        alpaca_data_url=_env("ALPACA_DATA_URL", "https://data.alpaca.markets"),
        alpaca_trading_url=_env("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets"),
        alpaca_data_feed=_env("ALPACA_DATA_FEED", "iex"),
        finnhub_api_key=_env("FINNHUB_API_KEY"),
        finnhub_api_url=_env("FINNHUB_API_URL", "https://finnhub.io/api/v1"),
        twelvedata_api_key=_env("TWELVEDATA_API_KEY"),
        twelvedata_api_url=_env("TWELVEDATA_API_URL", "https://api.twelvedata.com"),
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
        x_api_enabled=_env_bool("X_API_ENABLED", True),
        x_api_max_results=_env_int("X_API_MAX_RESULTS", 10),
        x_api_timeout_seconds=_env_int("X_API_TIMEOUT_SECONDS", 10),
        x_api_cache_ttl_seconds=_env_int("X_API_CACHE_TTL_SECONDS", 300),
        x_api_max_calls_per_window=_env_int("X_API_MAX_CALLS_PER_WINDOW", 20),
        x_api_window_seconds=_env_int("X_API_WINDOW_SECONDS", 900),
        x_api_min_seconds_between_calls=_env_int("X_API_MIN_SECONDS_BETWEEN_CALLS", 2),
        x_api_allow_oauth1_fallback=_env_bool("X_API_ALLOW_OAUTH1_FALLBACK", False),
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
        modal_simulation_app_name=_env("MODAL_SIMULATION_APP_NAME", "tickermaster-simulation"),
        modal_sandbox_timeout_seconds=_env_int("MODAL_SANDBOX_TIMEOUT_SECONDS", 600),
        modal_sandbox_idle_timeout_seconds=_env_int("MODAL_SANDBOX_IDLE_TIMEOUT_SECONDS", 120),
        modal_sandbox_python_version=_env("MODAL_SANDBOX_PYTHON_VERSION", "3.11"),
        modal_inference_function_name=_env("MODAL_INFERENCE_FUNCTION_NAME", "agent_inference"),
        modal_inference_timeout_seconds=_env_int("MODAL_INFERENCE_TIMEOUT_SECONDS", 15),
        poke_recipe_enabled=_env_bool("POKE_RECIPE_ENABLED", True),
        poke_api_key=_env("POKE_API_KEY"),
        poke_recipe_slug=_env("POKE_RECIPE_SLUG"),
        poke_api_url=_env("POKE_API", "https://poke.com/api/v1"),
        poke_kitchen_url=_env("POKE_KITCHEN_URL", "https://poke.com/kitchen"),
        twilio_account_sid=_env("TWILIO_ACCOUNT_SID"),
        twilio_auth_token=_env("TWILIO_AUTH_TOKEN"),
        twilio_from_number=_env("TWILIO_FROM_NUMBER"),
        twilio_whatsapp_from_number=_env("TWILIO_WHATSAPP_FROM_NUMBER") or _env("TWILIO_WHATSAPP_FROM"),
        twilio_use_whatsapp=_env_bool("TWILIO_USE_WHATSAPP", False),
        twilio_default_to_number=_env("TWILIO_DEFAULT_TO_NUMBER"),
        cerebras_api_key=_env("CEREBRAS_API_KEY"),
        nvidia_nim_api_key=_env("NVIDIA_NIM_API_KEY"),
        tracker_poll_interval_seconds=_env_int("TRACKER_POLL_INTERVAL_SECONDS", 300),
        default_watchlist=_env_list("DEFAULT_WATCHLIST", ["AAPL", "MSFT", "NVDA"]),
        mcp_tracker_router_enabled=_env_bool("MCP_TRACKER_ROUTER_ENABLED", True),
        mcp_tracker_server_command=_env("MCP_TRACKER_SERVER_COMMAND"),
        mcp_tracker_timeout_seconds=_env_int("MCP_TRACKER_TIMEOUT_SECONDS", 6),
    )
    _validate_settings(settings)
    return settings
