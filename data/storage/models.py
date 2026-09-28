"""SQLAlchemy models for PostgreSQL/TimescaleDB storage."""

from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    Numeric,
    String,
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
