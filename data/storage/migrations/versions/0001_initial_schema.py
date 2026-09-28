"""Initial market data schema and hypertables.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-28 23:55:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Schema creation
    op.execute("CREATE SCHEMA IF NOT EXISTS market_data;")

    # 1. Raw Market Events
    op.create_table(
        "raw_market_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "receive_time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id", "receive_time"),
        schema="market_data",
    )
    op.create_index(
        "idx_raw_events_symbol_time",
        "raw_market_events",
        ["symbol", sa.text("event_time DESC")],
        schema="market_data",
    )

    # 2. Candles
    op.create_table(
        "candles",
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("high", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("low", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("close", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("volume", sa.Numeric(precision=24, scale=8), nullable=False),
        sa.Column("trades_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_closed", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("symbol", "timeframe", "start_time"),
        schema="market_data",
    )
    op.create_index(
        "idx_candles_lookup",
        "candles",
        ["symbol", "timeframe", sa.text("start_time DESC")],
        schema="market_data",
    )

    # 3. Health log
    op.create_table(
        "ingestion_health_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("p50_latency_ms", sa.Float(), nullable=True),
        sa.Column("p95_latency_ms", sa.Float(), nullable=True),
        sa.Column("p99_latency_ms", sa.Float(), nullable=True),
        sa.Column("events_per_sec", sa.Float(), nullable=True),
        sa.Column("disconnections_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "recorded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="market_data",
    )


def downgrade() -> None:
    op.drop_table("ingestion_health_log", schema="market_data")
    op.drop_table("candles", schema="market_data")
    op.drop_table("raw_market_events", schema="market_data")
    op.execute("DROP SCHEMA IF EXISTS market_data CASCADE;")
