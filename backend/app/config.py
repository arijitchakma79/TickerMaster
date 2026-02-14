from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "TickerMaster API"
    environment: str = "development"
    log_level: str = "INFO"

    frontend_origins: List[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    openai_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct"

    perplexity_api_key: str = ""
    perplexity_model: str = "sonar"

    x_api_bearer_token: str = ""

    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "TickerMaster/1.0"

    kalshi_api_key: str = ""
    kalshi_api_secret: str = ""

    polymarket_api_key: str = ""

    modal_token_id: str = ""
    modal_token_secret: str = ""

    poke_recipe_enabled: bool = True
    poke_recipe_slug: str = ""
    poke_kitchen_url: str = "https://poke.com/kitchen"

    cerebras_api_key: str = ""
    nvidia_nim_api_key: str = ""

    tracker_poll_interval_seconds: int = 60
    default_watchlist: List[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"])


@lru_cache
def get_settings() -> Settings:
    return Settings()
