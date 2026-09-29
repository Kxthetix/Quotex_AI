"""Tests for Momentum Features (Phase 3 Section 11-15 & 42.C)."""

import pandas as pd
import pytest

from features.calculators.momentum import compute_momentum_features, register_momentum_features
from features.config import MomentumFeatureConfig
from features.registry import FeatureRegistry


def test_rsi_bounds_and_smoothing(synthetic_df: pd.DataFrame):
    """Verify RSI remains bounded in [0, 100] and exhibits correct warm-up."""
    config = MomentumFeatureConfig(
        rsi_periods=[7, 14],
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        roc_periods=[9],
        stoch_k_period=14,
        stoch_d_period=3,
    )
    res = compute_momentum_features(synthetic_df, config)

    # RSI 14 warm-up
    assert res["rsi_14"].iloc[:13].isna().all()

    valid_rsi_14 = res["rsi_14"].dropna()
    assert len(valid_rsi_14) > 0
    assert (valid_rsi_14 >= 0.0).all() and (valid_rsi_14 <= 100.0).all()

    # RSI 7 warms up earlier than RSI 14
    assert pd.notna(res["rsi_7"].iloc[7])


def test_macd_identity(synthetic_df: pd.DataFrame):
    """Verify MACD histogram is strictly macd_line - macd_signal."""
    config = MomentumFeatureConfig(
        rsi_periods=[14],
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        roc_periods=[],
        stoch_k_period=14,
    )
    res = compute_momentum_features(synthetic_df, config)

    # Once both line and signal are valid, hist == line - signal
    valid_mask = res["macd_line"].notna() & res["macd_signal"].notna()
    diff = res.loc[valid_mask, "macd_line"] - res.loc[valid_mask, "macd_signal"]
    hist = res.loc[valid_mask, "macd_hist"]

    # Difference must equal histogram to precision
    assert ((diff - hist).abs() < 1e-9).all()


def test_roc_calculation():
    """Verify Rate of Change calculation."""
    df = pd.DataFrame(
        {
            "close": [100.0, 110.0, 105.0, 120.0],
            "high": [105.0, 115.0, 110.0, 125.0],
            "low": [95.0, 105.0, 100.0, 115.0],
        }
    )
    config = MomentumFeatureConfig(
        rsi_periods=[],
        roc_periods=[1, 2],
        stoch_k_period=14,
    )
    res = compute_momentum_features(df, config)

    # 1-bar ROC at index 1: (110 - 100)/100 * 100 = 10.0%
    assert pytest.approx(res["roc_1"].iloc[1]) == 10.0
    # 2-bar ROC at index 2: (105 - 100)/100 * 100 = 5.0%
    assert pytest.approx(res["roc_2"].iloc[2]) == 5.0


def test_stochastic_oscillator(synthetic_df: pd.DataFrame):
    """Verify Stochastic %K and %D oscillators are bounded within [0, 100]."""
    config = MomentumFeatureConfig(
        rsi_periods=[],
        roc_periods=[],
        stoch_k_period=14,
        stoch_d_period=3,
        stoch_smooth_k=3,
    )
    res = compute_momentum_features(synthetic_df, config)

    k_vals = res["stoch_k_14"].dropna()
    d_vals = res["stoch_d_14"].dropna()

    assert len(k_vals) > 0
    assert len(d_vals) > 0
    assert (k_vals >= 0.0).all() and (k_vals <= 100.0).all()
    assert (d_vals >= 0.0).all() and (d_vals <= 100.0).all()


def test_momentum_registration():
    """Verify registry stores momentum definitions with correct metadata."""
    reg = FeatureRegistry()
    config = MomentumFeatureConfig()
    register_momentum_features(reg, config, version="v1.0.0")

    assert reg.get("rsi_14").category == "momentum"
    assert reg.get("macd_hist").lookback == 26 + 9
    assert reg.get("stoch_k_14") is not None
