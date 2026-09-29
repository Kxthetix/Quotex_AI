"""Tests for Price Features (Phase 3 Section 6 & 42.A)."""

import numpy as np
import pandas as pd
import pytest

from features.calculators.price import compute_price_features, register_price_features
from features.config import PriceFeatureConfig
from features.registry import FeatureRegistry


def test_price_features_basic_math():
    """Verify exact formula calculations for price features."""
    df = pd.DataFrame(
        {
            "open": [100.0, 105.0, 102.0],
            "high": [110.0, 108.0, 107.0],
            "low": [95.0, 101.0, 99.0],
            "close": [105.0, 103.0, 106.0],
            "volume": [10.0, 20.0, 15.0],
        }
    )
    config = PriceFeatureConfig(
        returns_periods=[1, 2],
        log_returns_periods=[1],
        include_candle_body=True,
        include_wicks=True,
        include_range=True,
        include_gap=True,
    )
    res = compute_price_features(df, config)

    # 1-bar simple return
    assert np.isnan(res["return_1"].iloc[0])
    assert pytest.approx(res["return_1"].iloc[1]) == (103.0 - 105.0) / 105.0
    assert pytest.approx(res["return_1"].iloc[2]) == (106.0 - 103.0) / 103.0

    # 2-bar return
    assert np.isnan(res["return_2"].iloc[0])
    assert np.isnan(res["return_2"].iloc[1])
    assert pytest.approx(res["return_2"].iloc[2]) == (106.0 - 105.0) / 105.0

    # 1-bar log return
    assert pytest.approx(res["log_return_1"].iloc[1]) == np.log(103.0 / 105.0)

    # Candle body: close - open
    assert pytest.approx(res["candle_body"].iloc[0]) == 105.0 - 100.0  # 5.0
    assert pytest.approx(res["candle_body"].iloc[1]) == 103.0 - 105.0  # -2.0

    # Upper wick: high - max(open, close)
    assert pytest.approx(res["upper_wick"].iloc[0]) == 110.0 - 105.0  # 5.0
    assert pytest.approx(res["upper_wick"].iloc[1]) == 108.0 - 105.0  # 3.0

    # Lower wick: min(open, close) - low
    assert pytest.approx(res["lower_wick"].iloc[0]) == 100.0 - 95.0  # 5.0
    assert pytest.approx(res["lower_wick"].iloc[1]) == 103.0 - 101.0  # 2.0

    # Range: high - low
    assert pytest.approx(res["candle_range"].iloc[0]) == 15.0
    assert pytest.approx(res["candle_range"].iloc[1]) == 7.0

    # Gap: open_t - close_{t-1}
    assert np.isnan(res["gap"].iloc[0])
    assert pytest.approx(res["gap"].iloc[1]) == 105.0 - 105.0  # 0.0
    assert pytest.approx(res["gap"].iloc[2]) == 102.0 - 103.0  # -1.0


def test_price_features_registration():
    """Verify feature registry correctly stores definitions and lookbacks."""
    reg = FeatureRegistry()
    config = PriceFeatureConfig(returns_periods=[1, 5], log_returns_periods=[1])
    register_price_features(reg, config, version="v1.0.0")

    defn = reg.get("return_5")
    assert defn is not None
    assert defn.category == "price"
    assert defn.lookback == 5
    assert defn.version == "v1.0.0"

    assert reg.get("candle_body") is not None
    assert reg.get("upper_wick") is not None
    assert reg.get("gap") is not None
    assert reg.get("gap").lookback == 2
