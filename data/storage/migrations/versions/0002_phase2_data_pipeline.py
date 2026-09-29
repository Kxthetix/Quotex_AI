"""Phase 2 data pipeline tables: normalized trades, data quality incidents, corrections, and ingestion runs.

Revision ID: 0002_phase2
Revises: 0001_initial
Create Date: 2026-09-29 00:20:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_phase2"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Normalized Trades
    op.create_table(
        "normalized_trades",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("trade_id", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=24, scale=8), nullable=False),
        sa.Column("side", sa.String(length=16), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "receive_time", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id", "timestamp"),
        schema="market_data",
    )
    op.create_index(
        "idx_trades_symbol_time",
        "normalized_trades",
        ["symbol", sa.text("timestamp DESC")],
        schema="market_data",
    )

    # 2. Data Quality Incidents
    op.create_table(
        "data_quality_incidents",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("check_name", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "recorded_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="market_data",
    )
    op.create_index(
        "idx_incidents_category_time",
        "data_quality_incidents",
        ["category", sa.text("recorded_at DESC")],
        schema="market_data",
    )
    op.create_index(
        "idx_incidents_symbol",
        "data_quality_incidents",
        ["symbol"],
        schema="market_data",
    )

    # 3. Candle Corrections
    op.create_table(
        "candle_corrections",
        sa.Column("correction_id", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("original_open", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("original_high", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("original_low", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("original_close", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("original_volume", sa.Numeric(precision=24, scale=8), nullable=False),
        sa.Column("original_trades", sa.Integer(), nullable=False),
        sa.Column("revised_high", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("revised_low", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("revised_close", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("revised_volume", sa.Numeric(precision=24, scale=8), nullable=False),
        sa.Column("revised_trades", sa.Integer(), nullable=False),
        sa.Column("late_trade_id", sa.String(length=64), nullable=False),
        sa.Column("late_trade_price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("late_trade_quantity", sa.Numeric(precision=24, scale=8), nullable=False),
        sa.Column("late_trade_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "detected_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("correction_id"),
        schema="market_data",
    )
    op.create_index(
        "idx_corrections_symbol_time",
        "candle_corrections",
        ["symbol", sa.text("start_time DESC")],
        schema="market_data",
    )

    # 4. Ingestion Runs
    op.create_table(
        "ingestion_runs",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False, server_default="1m"),
        sa.Column("requested_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("requested_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("records_received", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_accepted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("records_rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duplicates", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("errors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("completion_time", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("run_id"),
        schema="market_data",
    )
    op.create_index(
        "idx_runs_provider_symbol",
        "ingestion_runs",
        ["provider", "symbol", sa.text("created_at DESC")],
        schema="market_data",
    )


def downgrade() -> None:
    op.drop_table("ingestion_runs", schema="market_data")
    op.drop_table("candle_corrections", schema="market_data")
    op.drop_table("data_quality_incidents", schema="market_data")
    op.drop_table("normalized_trades", schema="market_data")
