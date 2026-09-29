"""Phase 3 feature engine tables: versioned feature vectors.

Revision ID: 0003_phase3
Revises: 0002_phase2
Create Date: 2026-09-29 01:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_phase3"
down_revision: str | None = "0002_phase2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Feature Vectors Table
    op.create_table(
        "feature_vectors",
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("timeframe", sa.String(length=16), nullable=False),
        sa.Column("feature_version", sa.String(length=32), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_valid", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("values", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("validity", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "calculated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("symbol", "timeframe", "feature_version", "timestamp"),
        schema="market_data",
    )

    # 2. Indexes for fast temporal and versioned queries
    op.create_index(
        "idx_features_lookup",
        "feature_vectors",
        ["symbol", "timeframe", "feature_version", sa.text("timestamp DESC")],
        schema="market_data",
    )


def downgrade() -> None:
    op.drop_index("idx_features_lookup", table_name="feature_vectors", schema="market_data")
    op.drop_table("feature_vectors", schema="market_data")
