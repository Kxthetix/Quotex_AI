"""FastAPI application for system health, telemetry, and market state inspection."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from configs.config import get_settings
from data.ingestion.pipeline import IngestionPipeline
from monitoring.health import SystemHealthReport
from monitoring.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Global singleton pipeline instance for API access
_pipeline: IngestionPipeline | None = None


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


def set_pipeline(pipeline: IngestionPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)
    logger.info("api_server_started", app_name=settings.app_name, env=settings.app_env)
    yield
    logger.info("api_server_shutting_down")


app = FastAPI(
    title="Trading Intelligence System API",
    description="Operational telemetry, health monitoring, and market data state API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=SystemHealthReport)
async def health_check():
    """System health check endpoint providing latency percentiles and breaker status."""
    pipeline = get_pipeline()
    report = pipeline.get_health()
    return report


@app.get("/api/v1/market/state/{symbol}")
async def get_market_state(symbol: str) -> dict[str, Any]:
    """Retrieve current in-memory market state for a given symbol."""
    pipeline = get_pipeline()
    clean_sym = symbol.upper()
    trade = await pipeline.state_store.get_latest_trade(clean_sym)
    candle = await pipeline.state_store.get_latest_candle(clean_sym, timeframe="1m")
    active_candle = pipeline.candle_builder.get_active_candle(clean_sym)

    return {
        "symbol": clean_sym,
        "latest_trade": trade.model_dump() if trade else None,
        "latest_closed_candle": candle.model_dump() if candle else None,
        "in_flight_candle": active_candle.model_dump() if active_candle else None,
    }


@app.post("/api/v1/market/breaker/reset")
async def reset_circuit_breaker() -> dict[str, str]:
    """Manually reset the data-quality circuit breaker after inspection."""
    pipeline = get_pipeline()
    pipeline.quality_engine.reset_breaker()
    return {"message": "Data quality circuit breaker has been reset successfully."}
