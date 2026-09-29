"""Mandatory Anti-Leakage Tests (Phase 3 Section 29, 31, 42.J & 43).

Strictly verifies that:
1. Features calculated at timestamp T depend ONLY on market data available at or before T.
2. Adding subsequent/future candles after T has zero effect on the feature values at T.
3. Higher-timeframe incomplete candles do not leak completed data into prior timestamps.
"""

from datetime import UTC, datetime

from features.config import FeatureEngineConfig
from features.engine import FeatureEngine
from features.registry import FeatureRegistry
from tests.unit.features.conftest import generate_candles


def test_anti_leakage_base_features():
    """Explicit Anti-Leakage Test for all base timeframe features (Section 43).

    Steps:
    1. Calculate features up to timestamp T.
    2. Save all feature values at timestamp T.
    3. Add 50 candles occurring strictly after timestamp T.
    4. Recalculate features for the expanded dataset.
    5. Assert that all feature values at timestamp T remain identical.
    """
    start_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    # Generate initial 120 candles up to timestamp T
    initial_candles = generate_candles(count=120, start_dt=start_dt, seed=42)
    timestamp_t = initial_candles[-1].end_time

    config = FeatureEngineConfig(version="v1.0.0")
    config.multi_timeframe.enabled = False  # Isolate base features first

    engine = FeatureEngine(config=config, registry=FeatureRegistry())

    # 1. Calculate features at timestamp T
    initial_vectors = engine.compute_historical_features(initial_candles)
    vector_at_t_initial = initial_vectors[-1]
    assert vector_at_t_initial.timestamp == timestamp_t

    # 2. Save all feature values at T
    saved_features_at_t = dict(vector_at_t_initial.values)
    assert len(saved_features_at_t) > 0

    # 3. Add 50 future candles occurring strictly after T
    future_start = timestamp_t
    future_candles = generate_candles(
        count=50,
        start_dt=future_start,
        base_price=initial_candles[-1].close,
        seed=999,
    )
    expanded_candles = initial_candles + future_candles
    assert len(expanded_candles) == 170

    # 4. Recalculate features for expanded dataset
    expanded_engine = FeatureEngine(config=config, registry=FeatureRegistry())
    expanded_vectors = expanded_engine.compute_historical_features(expanded_candles)

    # 5. Retrieve feature vector at timestamp T
    vector_at_t_after = expanded_vectors[119]
    assert vector_at_t_after.timestamp == timestamp_t

    # Assert that ALL features at T remain completely unchanged
    for feat_name, original_val in saved_features_at_t.items():
        recalc_val = vector_at_t_after.values[feat_name]
        if original_val is None:
            assert recalc_val is None, (
                f"Feature '{feat_name}' leaked: was None, became {recalc_val}"
            )
        else:
            assert recalc_val is not None, (
                f"Feature '{feat_name}' leaked: was {original_val}, became None"
            )
            assert recalc_val == original_val, (
                f"Feature '{feat_name}' value changed after adding future data! Before: {original_val}, After: {recalc_val}"
            )


def test_anti_leakage_multi_timeframe_features():
    """Explicit Anti-Leakage Test for multi-timeframe features (Section 29 & 43).

    Verifies that a 1-hour candle completing at 11:00 does not leak into a 10:30 feature vector
    when future candles completing that 1-hour bar are added to the dataset.
    """
    start_dt = datetime(2026, 1, 1, 9, 0, 0, tzinfo=UTC)

    # 90 minutes of 1m data: 09:00 to 10:30
    initial_candles = generate_candles(count=90, start_dt=start_dt, seed=777)
    target_dt = initial_candles[-1].end_time  # 10:30:00

    config = FeatureEngineConfig(version="v1.0.0")
    config.multi_timeframe.enabled = True
    config.multi_timeframe.enabled_timeframes = ["1h"]

    engine = FeatureEngine(config=config, registry=FeatureRegistry())

    # Calculate initial vectors
    initial_vectors = engine.compute_historical_features(initial_candles)
    vector_at_t_initial = initial_vectors[-1]
    assert vector_at_t_initial.timestamp == target_dt
    saved_mtf_values = {k: v for k, v in vector_at_t_initial.values.items() if k.startswith("mtf_")}
    assert len(saved_mtf_values) > 0

    # Add 60 minutes of future candles up to 11:30 (closing the 10:00-11:00 1h candle!)
    future_start = target_dt
    future_candles = generate_candles(
        count=60,
        start_dt=future_start,
        base_price=initial_candles[-1].close,
        seed=888,
    )
    expanded_candles = initial_candles + future_candles
    assert len(expanded_candles) == 150

    # Recalculate on expanded dataset
    expanded_engine = FeatureEngine(config=config, registry=FeatureRegistry())
    expanded_vectors = expanded_engine.compute_historical_features(expanded_candles)

    vector_at_t_after = expanded_vectors[89]
    assert vector_at_t_after.timestamp == target_dt

    # Assert that NO MTF feature at 10:30 changed, even though the 10:00-11:00 candle is now known
    for feat_name, original_val in saved_mtf_values.items():
        recalc_val = vector_at_t_after.values[feat_name]
        if original_val is None:
            assert recalc_val is None, f"MTF feature '{feat_name}' leaked future completed bar!"
        else:
            assert recalc_val == original_val, (
                f"MTF feature '{feat_name}' changed at 10:30 after future bar completed! Was {original_val}, now {recalc_val}"
            )
