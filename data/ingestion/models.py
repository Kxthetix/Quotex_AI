"""Core domain models for market data events, candles, and metrics."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class MarketEvent(BaseModel):
    """Raw market event representing a tick, trade, orderbook update, or kline."""

    source: str = Field(..., description="Source exchange or provider (e.g., 'binance')")
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTCUSDT')")
    event_type: str = Field(..., description="Type of event (e.g., 'trade', 'kline', 'bookTicker')")
    event_time: datetime = Field(..., description="Timestamp of the event from the exchange")
    receive_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when event was received locally",
    )
    payload: dict[str, Any] = Field(..., description="Original raw payload from the exchange")

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("event_time", "receive_time")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class NormalizedTrade(BaseModel):
    """Standardized representation of an individual executed trade."""

    trade_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    symbol: str
    price: float = Field(..., gt=0)
    quantity: float = Field(..., gt=0)
    side: str = Field(default="unknown")  # 'buy', 'sell', or 'unknown'
    timestamp: datetime
    receive_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("timestamp", "receive_time")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class Candle(BaseModel):
    """OHLCV candlestick data entity."""

    symbol: str = Field(..., description="Trading pair symbol")
    timeframe: str = Field(default="1m", description="Candle timeframe e.g. 1m, 5m, 1h, 1d")
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    start_time: datetime = Field(..., description="Candle start timestamp (inclusive)")
    end_time: datetime = Field(
        ..., description="Candle end timestamp (exclusive/inclusive boundary)"
    )
    trades_count: int = Field(default=0, ge=0)
    is_closed: bool = Field(default=True, description="Whether the candle is closed/finalized")

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("start_time", "end_time")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class LatencyMetrics(BaseModel):
    """Latency percentile metrics in milliseconds."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    sample_count: int
    min_ms: float
    max_ms: float
