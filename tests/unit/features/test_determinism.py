"""Tests for Feature Determinism and Reproducibility (Phase 3 Section 36 & 42.I)."""

from features.config import FeatureEngineConfig
from features.engine import FeatureEngine
from features.registry import FeatureRegistry
from tests.unit.features.conftest import generate_candles


def test_determinism_identical_runs():
    """Verify that multiple independent executions on identical input yield bitwise identical output."""
    candles = generate_candles(count=150, seed=123)

    config = FeatureEngineConfig(version="v1.0.0")

    # Run 1
    engine1 = FeatureEngine(config=config, registry=FeatureRegistry())
    vectors1 = engine1.compute_historical_features(candles)

    # Run 2
    engine2 = FeatureEngine(config=config, registry=FeatureRegistry())
    vectors2 = engine2.compute_historical_features(candles)

    assert len(vectors1) == len(vectors2)

    for i in range(len(vectors1)):
        v1 = vectors1[i]
        v2 = vectors2[i]

        assert v1.timestamp == v2.timestamp
        assert v1.symbol == v2.symbol
        assert v1.feature_version == v2.feature_version
        assert v1.is_valid == v2.is_valid
        assert set(v1.values.keys()) == set(v2.values.keys())

        for k in v1.values:
            val1 = v1.values[k]
            val2 = v2.values[k]
            if val1 is None:
                assert val2 is None
            else:
                assert val1 == val2
            assert v1.validity[k] == v2.validity[k]
