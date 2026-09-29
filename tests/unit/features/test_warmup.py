"""Tests for Feature Warm-Up Behavior and Missing Data (Phase 3 Section 32, 33, 41 & 42.H)."""

from features.config import FeatureEngineConfig
from features.engine import FeatureEngine
from features.models import FeatureValidity
from features.registry import FeatureRegistry
from tests.unit.features.conftest import generate_candles


def test_warmup_validity_states():
    """Verify warm-up features transition from WARMING_UP to VALID and never silent zeros."""
    # Generate 50 candles
    candles = generate_candles(count=50)

    # Configure an engine with a max lookback of 20
    config = FeatureEngineConfig()
    config.trend.ema_periods = [20]
    config.trend.sma_periods = [20]
    config.trend.structure_lookback = 10
    config.trend.slope_periods = [10]
    config.price.returns_periods = [5]
    config.price.log_returns_periods = [1]
    config.volatility.atr_periods = [7]
    config.volatility.historical_vol_periods = [10]
    config.volatility.rolling_vol_periods = [10]
    config.momentum.macd_slow = 12
    config.momentum.macd_signal = 5
    config.momentum.roc_periods = [5]
    config.momentum.rsi_periods = [7]
    config.momentum.stoch_k_period = 5
    config.volume.relative_vol_periods = [5]
    config.volume.volume_ma_periods = [10]
    config.volume.volume_trend_periods = [5]
    config.structure.support_resistance_lookback = 10
    config.multi_timeframe.enabled = False  # Isolate single timeframe for test

    reg = FeatureRegistry()
    engine = FeatureEngine(config=config, registry=reg)

    vectors = engine.compute_historical_features(candles)
    assert len(vectors) == 50

    # At bar 0: return_5, ema_20, sma_20 must all be in WARMING_UP
    v0 = vectors[0]
    assert v0.values["ema_20"] is None
    assert v0.values["sma_20"] is None
    assert v0.validity["ema_20"] == FeatureValidity.WARMING_UP
    assert v0.validity["sma_20"] == FeatureValidity.WARMING_UP
    assert v0.is_valid is False  # Vector overall is warming up

    # At bar 19 (the 20th bar, index 19):
    # sma_20 requires 20 periods, so at index 19 it should now be valid!
    v19 = vectors[19]
    assert v19.values["sma_20"] is not None
    assert v19.validity["sma_20"] == FeatureValidity.VALID

    # Check metrics
    metrics = engine.get_metrics()
    assert metrics["total_vectors_calculated"] == 50
    assert metrics["warming_up_vectors_count"] > 0
    assert metrics["valid_vectors_count"] > 0
    # No calculation errors
    assert metrics["calculation_errors_count"] == 0


def test_insufficient_history_never_produces_silent_zero():
    """Verify missing/insufficient values are None and NOT filled with 0.0."""
    candles = generate_candles(count=5)
    config = FeatureEngineConfig()
    config.price.returns_periods = [10]  # Requires 10 bars
    config.multi_timeframe.enabled = False

    reg = FeatureRegistry()
    engine = FeatureEngine(config=config, registry=reg)
    vectors = engine.compute_historical_features(candles)

    for v in vectors:
        assert v.values["return_10"] is None
        assert v.validity["return_10"] == FeatureValidity.WARMING_UP
        # Ensure it is not 0.0
        assert v.values["return_10"] != 0.0
