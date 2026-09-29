"""Tests for Volume Features (Phase 3 Section 19-22 & 42.E)."""

import numpy as np
import pandas as pd
import pytest

from features.calculators.volume import compute_volume_features, register_volume_features
from features.config import VolumeFeatureConfig
from features.registry import FeatureRegistry


def test_relative_volume_and_spike():
    """Verify relative volume calculation and volume spike detection."""
    # 25 candles with flat volume of 100, then spike of 500 at index 22
    volumes = [100.0] * 25
    volumes[22] = 500.0

    df = pd.DataFrame(
        {
            "open": [10.0] * 25,
            "high": [11.0] * 25,
            "low": [9.0] * 25,
            "close": [10.5] * 25,
            "volume": volumes,
        }
    )
    config = VolumeFeatureConfig(
        relative_vol_periods=[10, 20],
        volume_ma_periods=[20],
        volume_spike_threshold=2.0,
        volume_trend_periods=[10],
    )
    res = compute_volume_features(df, config)

    # Relative volume: at index 21, volume is 100, prior 10-period avg is 100 -> rel_vol_10 == 1.0
    assert pytest.approx(res["relative_vol_10"].iloc[21]) == 1.0

    # At index 22, volume is 500, prior 20-period avg was 100 -> rel_vol_20 == 5.0
    assert pytest.approx(res["relative_vol_20"].iloc[22]) == 5.0

    # Volume spike at index 22: 500 > 2.0 * 100 -> spike == 1.0
    assert res["volume_spike"].iloc[22] == 1.0
    # At index 21: not a spike -> 0.0
    assert res["volume_spike"].iloc[21] == 0.0


def test_volume_zero_division_safety():
    """Verify zero volume does not produce infinity or crashing exceptions."""
    df = pd.DataFrame(
        {
            "open": [10.0] * 30,
            "high": [11.0] * 30,
            "low": [9.0] * 30,
            "close": [10.5] * 30,
            "volume": [0.0] * 30,  # Zero volume provider scenario
        }
    )
    config = VolumeFeatureConfig(
        relative_vol_periods=[10],
        volume_ma_periods=[20],
        volume_spike_threshold=2.0,
        volume_trend_periods=[10],
    )
    res = compute_volume_features(df, config)

    # Relative volume when prior volume is 0 should safely be NaN, not inf
    valid_rel = res["relative_vol_10"].dropna()
    assert len(valid_rel) == 0 or not np.isinf(valid_rel).any()


def test_volume_registration():
    """Verify volume features in registry."""
    reg = FeatureRegistry()
    config = VolumeFeatureConfig()
    register_volume_features(reg, config, version="v1.0.0")

    assert reg.get("volume_ma_20").category == "volume"
    assert reg.get("volume_spike").lookback == 21
