"""Tests for Historical Batch vs. Incremental Live Consistency (Phase 3 Section 38, 39 & 42.K)."""

from features.config import FeatureEngineConfig
from features.engine import FeatureEngine
from features.registry import FeatureRegistry
from tests.unit.features.conftest import generate_candles


def test_batch_vs_live_consistency():
    """Verify that batch historical calculation and incremental live calculation yield identical results."""
    candles = generate_candles(count=150, seed=555)

    config = FeatureEngineConfig(version="v1.0.0")
    engine = FeatureEngine(config=config, registry=FeatureRegistry())

    # 1. Compute historical batch
    batch_vectors = engine.compute_historical_features(candles)
    assert len(batch_vectors) == 150
    latest_batch_vector = batch_vectors[-1]

    # 2. Compute incremental live using candle buffer
    live_vector = engine.compute_live_features(candles)

    # 3. Assert identity
    assert live_vector.timestamp == latest_batch_vector.timestamp
    assert live_vector.symbol == latest_batch_vector.symbol
    assert live_vector.feature_version == latest_batch_vector.feature_version
    assert live_vector.is_valid == latest_batch_vector.is_valid

    assert set(live_vector.values.keys()) == set(latest_batch_vector.values.keys())

    for k in live_vector.values:
        live_val = live_vector.values[k]
        batch_val = latest_batch_vector.values[k]
        if batch_val is None:
            assert live_val is None, f"Feature '{k}' mismatch: live is {live_val}, batch is None"
        else:
            assert live_val is not None, (
                f"Feature '{k}' mismatch: live is None, batch is {batch_val}"
            )
            assert live_val == batch_val, (
                f"Feature '{k}' mismatch: live={live_val} vs batch={batch_val}"
            )
