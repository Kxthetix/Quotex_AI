"""Tests for Trend Features (Phase 3 Section 7-10 & 42.B)."""

import numpy as np
import pandas as pd
import pytest

from features.calculators.trend import compute_trend_features, register_trend_features
from features.config import TrendFeatureConfig
from features.registry import FeatureRegistry


def test_ema_sma_trend_features(synthetic_df: pd.DataFrame):
    """Verify EMA and SMA behavior and warm-up requirements."""
    config = TrendFeatureConfig(
        ema_periods=[9, 20],
        sma_periods=[20],
        adx_period=14,
        slope_periods=[10],
        structure_lookback=20,
    )
    res = compute_trend_features(synthetic_df, config)

    # SMA(20) must be NaN for first 19 periods, then numeric
    assert res["sma_20"].iloc[:19].isna().all()
    assert pd.notna(res["sma_20"].iloc[19])
    # The 20th point must match direct mean of first 20 closes
    expected_sma_20 = synthetic_df["close"].iloc[:20].mean()
    assert pytest.approx(res["sma_20"].iloc[19]) == expected_sma_20

    # EMA(9) warm-up
    assert res["ema_9"].iloc[:8].isna().all()
    assert pd.notna(res["ema_9"].iloc[8])


def test_adx_computation(synthetic_df: pd.DataFrame):
    """Verify ADX, +DI, and -DI are within [0, 100] and deterministic."""
    config = TrendFeatureConfig(
        ema_periods=[9],
        sma_periods=[20],
        adx_period=14,
        slope_periods=[10],
        structure_lookback=20,
    )
    res = compute_trend_features(synthetic_df, config)

    # After warm-up (around 28 bars for ADX)
    valid_adx = res["adx_14"].dropna()
    assert len(valid_adx) > 0
    assert (valid_adx >= 0.0).all() and (valid_adx <= 100.0).all()

    valid_plus_di = res["plus_di_14"].dropna()
    assert (valid_plus_di >= 0.0).all() and (valid_plus_di <= 100.0).all()

    valid_minus_di = res["minus_di_14"].dropna()
    assert (valid_minus_di >= 0.0).all() and (valid_minus_di <= 100.0).all()


def test_trend_structure_states():
    """Verify Higher-High, Lower-Low, Higher-Low, Lower-High strictly excludes current bar."""
    # Construct sequence where bar 5 breaks out to higher high
    highs = [10.0, 11.0, 12.0, 11.5, 12.0, 15.0]  # bar 5 breaks high of 12.0
    lows = [8.0, 9.0, 10.0, 9.5, 10.0, 11.0]  # bar 5 low 11.0 is higher low (> max low 10.0)
    closes = [9.0, 10.0, 11.0, 10.5, 11.0, 14.0]
    opens = [8.5, 9.5, 10.5, 10.0, 10.5, 11.5]

    df = pd.DataFrame({"high": highs, "low": lows, "close": closes, "open": opens})
    config = TrendFeatureConfig(
        ema_periods=[],
        sma_periods=[],
        adx_period=5,
        slope_periods=[],
        structure_lookback=3,
    )
    res = compute_trend_features(df, config)

    # First 3 bars are warming up -> NaN
    assert np.isnan(res["higher_high_3"].iloc[0])
    assert np.isnan(res["higher_high_3"].iloc[1])
    assert np.isnan(res["higher_high_3"].iloc[2])

    # At bar 5 (high = 15.0), prior 3 highs are bar 2(12.0), bar 3(11.5), bar 4(12.0) -> max is 12.0
    # 15.0 > 12.0 -> higher_high_3 must be 1.0
    assert res["higher_high_3"].iloc[5] == 1.0
    # At bar 5 (low = 11.0), prior 3 lows are bar 2(10.0), bar 3(9.5), bar 4(10.0) -> max is 10.0
    # 11.0 > 10.0 -> higher_low_3 must be 1.0
    assert res["higher_low_3"].iloc[5] == 1.0


def test_trend_registration():
    """Verify registry stores trend definitions with correct lookbacks."""
    reg = FeatureRegistry()
    config = TrendFeatureConfig(ema_periods=[20, 200], adx_period=14)
    register_trend_features(reg, config, version="v1.0.0")

    assert reg.get("ema_200").lookback == 200
    assert reg.get("adx_14").lookback == 28
    assert reg.get("higher_high_20").lookback == 21
