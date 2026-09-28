"""Configuration management for Trading Intelligence System.

Loads configuration from environment variables and .env file with strong typing
and validation via Pydantic.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # General Environment
    app_name: str = "Trading Intelligence System"
    app_env: Literal["development", "paper", "production", "test"] = "development"
    debug: bool = False

    # Structured Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    # Market Data
    data_provider: str = "binance"
    market_symbols: str | list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    ws_base_url: str = "wss://stream.binance.com:9443/ws"
    rest_base_url: str = "https://api.binance.com"
    stale_data_timeout_seconds: float = 15.0
    max_reconnect_attempts: int = 10
    initial_reconnect_delay_seconds: float = 1.0
    max_reconnect_delay_seconds: float = 30.0

    # PostgreSQL / TimescaleDB
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "trading_db"
    postgres_pool_size: int = 10

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Local Storage Paths
    raw_storage_path: Path = Path("data/storage/raw")
    cleaned_storage_path: Path = Path("data/storage/cleaned")
    parquet_storage_path: Path = Path("data/storage/parquet")

    @field_validator("market_symbols", mode="after")
    @classmethod
    def parse_market_symbols(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            # Check if it is a JSON array string
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                import json

                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(s).strip().upper() for s in parsed]
                except Exception:
                    pass
            return [s.strip().upper() for s in value.split(",") if s.strip()]
        return [str(s).upper() for s in value]

    @property
    def postgres_dsn(self) -> str:
        """Construct PostgreSQL async DSN for asyncpg."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_dsn(self) -> str:
        """Construct PostgreSQL synchronous DSN for migrations/MLflow."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
