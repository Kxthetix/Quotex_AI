"""Core domain models for market data events, candles, and metrics."""

from datetime import UTC, datetime
from enum import StrEnum
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


class IngestionRunStatus(StrEnum):
    STARTED = "STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class IngestionRun(BaseModel):
    """Execution run ledger for tracking historical and bulk ingestion jobs."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    provider: str
    symbol: str
    timeframe: str = "1m"
    requested_start: datetime | None = None
    requested_end: datetime | None = None
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    status: IngestionRunStatus = IngestionRunStatus.STARTED
    records_received: int = 0
    records_accepted: int = 0
    records_rejected: int = 0
    duplicates: int = 0
    errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completion_time: datetime | None = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()


class CandleCorrectionRecord(BaseModel):
    """Audit record generated when a late/out-of-order trade adjusts an already-closed candle."""

    correction_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    timeframe: str
    start_time: datetime
    original_open: float
    original_high: float
    original_low: float
    original_close: float
    original_volume: float
    original_trades: int
    revised_high: float
    revised_low: float
    revised_close: float
    revised_volume: float
    revised_trades: int
    late_trade_id: str
    late_trade_price: float
    late_trade_quantity: float
    late_trade_timestamp: datetime
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CandleGapReason(StrEnum):
    NETWORK_OUTAGE = "NETWORK_OUTAGE"
    PROVIDER_DATA_DROPOUT = "PROVIDER_DATA_DROPOUT"
    MARKET_SESSION_CLOSED = "MARKET_SESSION_CLOSED"
    NO_TRADE_PERIOD = "NO_TRADE_PERIOD"
    UNKNOWN = "UNKNOWN"


class CandleGapRecord(BaseModel):
    """Structured record of a detected sequence gap between consecutive candles."""

    gap_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    timeframe: str
    gap_start: datetime
    gap_end: datetime
    missing_candles_count: int
    reason: CandleGapReason = CandleGapReason.UNKNOWN
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_failure: bool = True
