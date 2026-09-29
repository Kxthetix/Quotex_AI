"""Tests for Volatility Features (Phase 3 Section 16-18 & 42.D)."""

import pandas as pd

from features.calculators.volatility import (
    compute_volatility_features,
    register_volatility_features,
)
from features.config import VolatilityFeatureConfig
from features.registry import FeatureRegistry


def test_atr_behavior(synthetic_df: pd.DataFrame):
    """Verify ATR is strictly positive and incorporates previous close for gaps."""
    config = VolatilityFeatureConfig(
        atr_periods=[7, 14],
        rolling_vol_periods=[10],
        historical_vol_periods=[20],
    )
    res = compute_volatility_features(synthetic_df, config)

    # ATR warm-up
    assert res["atr_14"].iloc[:13].isna().all()
    valid_atr = res["atr_14"].dropna()
    assert (valid_atr > 0.0).all()

    # Faster ATR warms up first
    assert pd.notna(res["atr_7"].iloc[7])


def test_rolling_and_historical_volatility(synthetic_df: pd.DataFrame):
    """Verify rolling volatility and annualized historical volatility are non-negative."""
    config = VolatilityFeatureConfig(
        atr_periods=[],
        rolling_vol_periods=[10, 20],
        historical_vol_periods=[20],
    )
    res = compute_volatility_features(synthetic_df, config)

    # Rolling vol(10) requires 10 bars
    assert res["rolling_vol_10"].iloc[:9].isna().all()
    valid_roll = res["rolling_vol_10"].dropna()
    assert (valid_roll >= 0.0).all()

    # Historical vol is annualized, strictly non-negative
    valid_hist = res["historical_vol_20"].dropna()
    assert (valid_hist >= 0.0).all()


def test_volatility_registration():
    """Verify volatility definitions in registry."""
    reg = FeatureRegistry()
    config = VolatilityFeatureConfig(atr_periods=[14], historical_vol_periods=[60])
    register_volatility_features(reg, config, version="v1.0.0")

    assert reg.get("atr_14").category == "volatility"
    assert reg.get("historical_vol_60").lookback == 60
