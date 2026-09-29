"""Feature storage for Phase 3.

Provides high-performance Parquet serialization for offline research and ML model training,
as well as relational/TimescaleDB persistence for versioned feature vectors.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from data.storage.models import FeatureVectorModel
from features.models import FeatureValidity, FeatureVector
from monitoring.logging import get_logger

logger = get_logger(__name__)


def save_features_to_parquet(features: Sequence[FeatureVector], path: Path | str) -> None:
    """Save a sequence of FeatureVectors to a Parquet file.

    Flattens feature values into individual numeric columns while maintaining
    metadata (symbol, timeframe, feature_version, timestamp, is_valid) and validity flags.
    """
    if not features:
        logger.warning("No features provided to save_features_to_parquet.")
        return

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for f in features:
        rec = {
            "symbol": f.symbol,
            "timeframe": f.timeframe,
            "feature_version": f.feature_version,
            "timestamp": f.timestamp,
            "is_valid": f.is_valid,
            "calculated_at": f.calculated_at,
            "_validity_json": json.dumps({k: v.value for k, v in f.validity.items()}),
        }
        # Flatten numeric feature values
        for k, v in f.values.items():
            rec[k] = float(v) if v is not None else None
        records.append(rec)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["calculated_at"] = pd.to_datetime(df["calculated_at"], utc=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(target_path), compression="snappy")
    logger.info("Saved %d feature vectors to %s", len(features), target_path)


def load_features_from_parquet(
    path: Path | str,
    symbol: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> list[FeatureVector]:
    """Read feature vectors from a Parquet file with optional filtering."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Feature Parquet file not found: {target_path}")

    df = pq.read_table(str(target_path)).to_pandas()
    if df.empty:
        return []

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if symbol:
        df = df[df["symbol"] == symbol.upper()]
    if start_time:
        st = pd.to_datetime(start_time, utc=True)
        df = df[df["timestamp"] >= st]
    if end_time:
        et = pd.to_datetime(end_time, utc=True)
        df = df[df["timestamp"] <= et]

    meta_cols = {
        "symbol",
        "timeframe",
        "feature_version",
        "timestamp",
        "is_valid",
        "calculated_at",
        "_validity_json",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]

    vectors: list[FeatureVector] = []
    for _, row in df.iterrows():
        val_map = {}
        for col in feature_cols:
            val = row[col]
            val_map[col] = float(val) if pd.notna(val) else None

        validity_dict = {}
        if "_validity_json" in row and pd.notna(row["_validity_json"]):
            raw_val = json.loads(row["_validity_json"])
            for k, v in raw_val.items():
                validity_dict[k] = FeatureValidity(v)

        ts = row["timestamp"]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()

        calc_at = row.get("calculated_at", datetime.now(UTC))
        if hasattr(calc_at, "to_pydatetime"):
            calc_at = calc_at.to_pydatetime()

        vec = FeatureVector(
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            feature_version=row["feature_version"],
            timestamp=ts,
            values=val_map,
            validity=validity_dict,
            is_valid=bool(row["is_valid"]),
            calculated_at=calc_at,
        )
        vectors.append(vec)

    return vectors


def load_features_dataframe_from_parquet(path: Path | str) -> pd.DataFrame:
    """Read features into a tabular DataFrame for ML training and statistical analysis."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Feature Parquet file not found: {target_path}")

    df = pq.read_table(str(target_path)).to_pandas()
    if "_validity_json" in df.columns:
        df = df.drop(columns=["_validity_json"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


async def save_features_to_db(session: AsyncSession, features: Sequence[FeatureVector]) -> int:
    """Persist feature vectors into PostgreSQL/TimescaleDB market_data.feature_vectors."""
    if not features:
        return 0

    records = [
        {
            "symbol": f.symbol,
            "timeframe": f.timeframe,
            "feature_version": f.feature_version,
            "timestamp": f.timestamp,
            "is_valid": f.is_valid,
            "values": f.values,
            "validity": {k: v.value for k, v in f.validity.items()},
            "calculated_at": f.calculated_at,
        }
        for f in features
    ]

    is_postgres = session.bind.dialect.name == "postgresql"
    if is_postgres:
        stmt = insert(FeatureVectorModel).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "timeframe", "feature_version", "timestamp"],
            set_={
                "is_valid": stmt.excluded.is_valid,
                "values": stmt.excluded.values,
                "validity": stmt.excluded.validity,
                "calculated_at": stmt.excluded.calculated_at,
            },
        )
        await session.execute(stmt)
    else:
        # Fallback for SQLite / generic DB in tests
        for rec in records:
            model = FeatureVectorModel(**rec)
            await session.merge(model)

    await session.commit()
    return len(features)


async def load_features_from_db(
    session: AsyncSession,
    symbol: str,
    timeframe: str = "1m",
    feature_version: str = "v1.0.0",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> list[FeatureVector]:
    """Retrieve versioned feature vectors from the database."""
    stmt = (
        select(FeatureVectorModel)
        .where(
            FeatureVectorModel.symbol == symbol.upper(),
            FeatureVectorModel.timeframe == timeframe,
            FeatureVectorModel.feature_version == feature_version,
        )
        .order_by(FeatureVectorModel.timestamp.asc())
    )

    if start_time:
        stmt = stmt.where(FeatureVectorModel.timestamp >= start_time)
    if end_time:
        stmt = stmt.where(FeatureVectorModel.timestamp <= end_time)

    result = await session.execute(stmt)
    rows = result.scalars().all()

    vectors: list[FeatureVector] = []
    for r in rows:
        val_map = {k: (float(v) if v is not None else None) for k, v in r.values.items()}
        validity_map = {k: FeatureValidity(v) for k, v in r.validity.items()}
        vec = FeatureVector(
            symbol=r.symbol,
            timeframe=r.timeframe,
            feature_version=r.feature_version,
            timestamp=r.timestamp,
            values=val_map,
            validity=validity_map,
            is_valid=r.is_valid,
            calculated_at=r.calculated_at,
        )
        vectors.append(vec)

    return vectors
