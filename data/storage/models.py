"""SQLAlchemy models for PostgreSQL/TimescaleDB storage."""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RawMarketEventModel(Base):
    """Raw market event table, converted to TimescaleDB hypertable partitioned by receive_time."""

    __tablename__ = "raw_market_events"
    __table_args__ = {"schema": "market_data"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(String(64), nullable=False)
    symbol = Column(String(32), nullable=False, index=True)
    event_type = Column(String(32), nullable=False)
    event_time = Column(DateTime(timezone=True), nullable=False)
    receive_time = Column(
        DateTime(timezone=True),
        primary_key=True,
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    payload = Column(JSONB, nullable=False)


class CandleModel(Base):
    """Cleaned candlesticks table, converted to TimescaleDB hypertable partitioned by start_time."""

    __tablename__ = "candles"
    __table_args__ = {"schema": "market_data"}

    symbol = Column(String(32), primary_key=True, nullable=False)
    timeframe = Column(String(16), primary_key=True, nullable=False)
    start_time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    open = Column(Numeric(20, 8), nullable=False)
    high = Column(Numeric(20, 8), nullable=False)
    low = Column(Numeric(20, 8), nullable=False)
    close = Column(Numeric(20, 8), nullable=False)
    volume = Column(Numeric(24, 8), nullable=False)
    trades_count = Column(Integer, default=0, nullable=False)
    is_closed = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class NormalizedTradeModel(Base):
    """Cleaned individual trade ticks, converted to TimescaleDB hypertable partitioned by timestamp."""

    __tablename__ = "normalized_trades"
    __table_args__ = {"schema": "market_data"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), nullable=False, index=True)
    source = Column(String(64), nullable=False)
    symbol = Column(String(32), nullable=False, index=True)
    price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(24, 8), nullable=False)
    side = Column(String(16), nullable=False)
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    receive_time = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class DataQualityIncidentModel(Base):
    """Structured data quality violations and breaker trip incidents."""

    __tablename__ = "data_quality_incidents"
    __table_args__ = {"schema": "market_data"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    check_name = Column(String(64), nullable=False)
    category = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), nullable=False)
    symbol = Column(String(32), nullable=True, index=True)
    message = Column(Text, nullable=False)
    details = Column(JSONB, nullable=True)
    recorded_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )


class CandleCorrectionModel(Base):
    """Audit ledger recording when late/out-of-order ticks revise a closed candlestick."""

    __tablename__ = "candle_corrections"
    __table_args__ = {"schema": "market_data"}

    correction_id = Column(String(64), primary_key=True)
    symbol = Column(String(32), nullable=False, index=True)
    timeframe = Column(String(16), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    original_open = Column(Numeric(20, 8), nullable=False)
    original_high = Column(Numeric(20, 8), nullable=False)
    original_low = Column(Numeric(20, 8), nullable=False)
    original_close = Column(Numeric(20, 8), nullable=False)
    original_volume = Column(Numeric(24, 8), nullable=False)
    original_trades = Column(Integer, nullable=False)
    revised_high = Column(Numeric(20, 8), nullable=False)
    revised_low = Column(Numeric(20, 8), nullable=False)
    revised_close = Column(Numeric(20, 8), nullable=False)
    revised_volume = Column(Numeric(24, 8), nullable=False)
    revised_trades = Column(Integer, nullable=False)
    late_trade_id = Column(String(64), nullable=False)
    late_trade_price = Column(Numeric(20, 8), nullable=False)
    late_trade_quantity = Column(Numeric(24, 8), nullable=False)
    late_trade_timestamp = Column(DateTime(timezone=True), nullable=False)
    detected_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class IngestionRunModel(Base):
    """Ledger of batch ingestion and historical backfill jobs."""

    __tablename__ = "ingestion_runs"
    __table_args__ = {"schema": "market_data"}

    run_id = Column(String(64), primary_key=True)
    provider = Column(String(64), nullable=False, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    timeframe = Column(String(16), default="1m", nullable=False)
    requested_start = Column(DateTime(timezone=True), nullable=True)
    requested_end = Column(DateTime(timezone=True), nullable=True)
    actual_start = Column(DateTime(timezone=True), nullable=True)
    actual_end = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(32), nullable=False, index=True)
    records_received = Column(Integer, default=0, nullable=False)
    records_accepted = Column(Integer, default=0, nullable=False)
    records_rejected = Column(Integer, default=0, nullable=False)
    duplicates = Column(Integer, default=0, nullable=False)
    errors = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    completion_time = Column(DateTime(timezone=True), nullable=True)


class IngestionHealthLogModel(Base):
    """Ingestion health and operational metrics."""

    __tablename__ = "ingestion_health_log"
    __table_args__ = {"schema": "market_data"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False)
    p50_latency_ms = Column(Float, nullable=True)
    p95_latency_ms = Column(Float, nullable=True)
    p99_latency_ms = Column(Float, nullable=True)
    events_per_sec = Column(Float, nullable=True)
    disconnections_count = Column(Integer, default=0)
    details = Column(JSONB, nullable=True)
    recorded_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class FeatureVectorModel(Base):
    """Calculated versioned feature vectors table for Phase 3."""

    __tablename__ = "feature_vectors"
    __table_args__ = (
        Index("idx_features_lookup", "symbol", "timeframe", "feature_version", "timestamp"),
        {"schema": "market_data"},
    )

    symbol = Column(String(32), primary_key=True, nullable=False)
    timeframe = Column(String(16), primary_key=True, default="1m", nullable=False)
    feature_version = Column(String(32), primary_key=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    is_valid = Column(Boolean, nullable=False, default=True)
    values = Column(JSONB().with_variant(JSON, "sqlite"), nullable=False)
    validity = Column(JSONB().with_variant(JSON, "sqlite"), nullable=False)
    calculated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
