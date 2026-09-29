"""Tests for Feature Storage (Phase 3 Section 40)."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from data.storage.features_store import (
    load_features_dataframe_from_parquet,
    load_features_from_db,
    load_features_from_parquet,
    save_features_to_db,
    save_features_to_parquet,
)
from data.storage.models import FeatureVectorModel
from features.engine import FeatureEngine
from features.models import FeatureValidity, FeatureVector
from tests.unit.features.conftest import generate_candles


def test_parquet_storage_roundtrip(tmp_path: Path):
    """Verify Parquet save, load, and dataframe extraction."""
    candles = generate_candles(count=30, seed=101)
    engine = FeatureEngine()
    vectors = engine.compute_historical_features(candles)

    file_path = tmp_path / "features_test.parquet"
    save_features_to_parquet(vectors, file_path)
    assert file_path.exists()

    # Load back as FeatureVector list
    loaded_vectors = load_features_from_parquet(file_path)
    assert len(loaded_vectors) == len(vectors)

    # Check first and last vector
    v_orig = vectors[-1]
    v_loaded = loaded_vectors[-1]
    assert v_orig.symbol == v_loaded.symbol
    assert v_orig.timestamp == v_loaded.timestamp
    assert v_orig.feature_version == v_loaded.feature_version
    assert v_orig.is_valid == v_loaded.is_valid

    for k in v_orig.values:
        if v_orig.values[k] is None:
            assert v_loaded.values[k] is None
        else:
            assert pytest.approx(v_orig.values[k]) == v_loaded.values[k]

    # Load as tabular DataFrame
    df = load_features_dataframe_from_parquet(file_path)
    assert len(df) == len(vectors)
    assert "return_1" in df.columns
    assert "symbol" in df.columns
    assert "timestamp" in df.columns


@pytest.mark.asyncio
async def test_database_storage_mocked_session():
    """Verify storing and loading feature vectors with SQLAlchemy PostgreSQL session."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.bind = MagicMock()
    mock_session.bind.dialect.name = "postgresql"

    ts1 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    ts2 = datetime(2026, 1, 1, 10, 1, 0, tzinfo=UTC)

    vecs = [
        FeatureVector(
            symbol="BTC/USDT",
            timeframe="1m",
            feature_version="v1.0.0",
            timestamp=ts1,
            values={"return_1": 0.01, "ema_20": 50000.0},
            validity={"return_1": FeatureValidity.VALID, "ema_20": FeatureValidity.VALID},
            is_valid=True,
        ),
        FeatureVector(
            symbol="BTC/USDT",
            timeframe="1m",
            feature_version="v1.0.0",
            timestamp=ts2,
            values={"return_1": 0.02, "ema_20": 50100.0},
            validity={"return_1": FeatureValidity.VALID, "ema_20": FeatureValidity.VALID},
            is_valid=True,
        ),
    ]

    # Test saving
    saved_count = await save_features_to_db(mock_session, vecs)
    assert saved_count == 2
    assert mock_session.execute.called
    assert mock_session.commit.called

    # Test loading
    mock_model_1 = FeatureVectorModel(
        symbol="BTC/USDT",
        timeframe="1m",
        feature_version="v1.0.0",
        timestamp=ts1,
        is_valid=True,
        values={"return_1": 0.01, "ema_20": 50000.0},
        validity={"return_1": "VALID", "ema_20": "VALID"},
        calculated_at=datetime.now(UTC),
    )
    mock_model_2 = FeatureVectorModel(
        symbol="BTC/USDT",
        timeframe="1m",
        feature_version="v1.0.0",
        timestamp=ts2,
        is_valid=True,
        values={"return_1": 0.02, "ema_20": 50100.0},
        validity={"return_1": "VALID", "ema_20": "VALID"},
        calculated_at=datetime.now(UTC),
    )

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_model_1, mock_model_2]
    mock_session.execute.return_value = mock_result

    loaded = await load_features_from_db(
        mock_session,
        symbol="BTC/USDT",
        timeframe="1m",
        feature_version="v1.0.0",
    )
    assert len(loaded) == 2
    assert loaded[0].values["return_1"] == 0.01
    assert loaded[1].values["return_1"] == 0.02
    assert loaded[0].validity["return_1"] == FeatureValidity.VALID
