"""Tests for Market Structure Features (Phase 3 Section 23-27 & 42.F)."""

import pandas as pd
import pytest

from features.calculators.structure import compute_structure_features, register_structure_features
from features.config import StructureFeatureConfig
from features.registry import FeatureRegistry


def test_support_resistance_and_breakout():
    """Verify support and resistance levels exclude the current bar and trigger breakouts."""
    # 5 bars with high=110, low=90, then bar 6 closes at 115 (breakout), bar 7 closes at 85 (breakdown)
    highs = [110.0] * 5 + [120.0, 100.0]
    lows = [90.0] * 5 + [105.0, 80.0]
    closes = [100.0] * 5 + [115.0, 85.0]
    opens = [95.0] * 5 + [105.0, 95.0]

    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
    config = StructureFeatureConfig(
        support_resistance_lookback=5,
        breakout_lookback=5,
        consolidation_lookback=5,
        consolidation_threshold=0.25,
    )
    res = compute_structure_features(df, config)

    # First 5 bars are warming up
    assert res["support_level_5"].iloc[:4].isna().all()

    # At bar index 5 (6th bar):
    # Prior 5 bars: lows are 90.0, highs are 110.0
    assert res["support_level_5"].iloc[5] == 90.0
    assert res["resistance_level_5"].iloc[5] == 110.0

    # Bar 5 close is 115.0 > resistance (110.0) -> breakout = 1.0, breakdown = 0.0
    assert res["breakout_5"].iloc[5] == 1.0
    assert res["breakdown_5"].iloc[5] == 0.0

    # Retest distance to resistance: |115 - 110| / 110 = 5 / 110
    assert pytest.approx(res["retest_dist_resistance_5"].iloc[5]) == 5.0 / 110.0


def test_consolidation_and_range_position():
    """Verify range compression ratio and range position bounds."""
    # Tight range channel between 99 and 101 for 5 bars
    highs = [101.0] * 6
    lows = [99.0] * 6
    closes = [100.0] * 6
    opens = [100.0] * 6

    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})
    config = StructureFeatureConfig(
        support_resistance_lookback=5,
        consolidation_threshold=0.05,
    )
    res = compute_structure_features(df, config)

    # At index 5: channel width = 101 - 99 = 2.0, mean_price = 100.0
    # Range compression = 2.0 / 100.0 = 0.02 < threshold (0.05) -> is_consolidating = 1.0
    assert pytest.approx(res["range_compression_5"].iloc[5]) == 0.02
    assert res["is_consolidating_5"].iloc[5] == 1.0

    # Range position: (100 - 99) / (101 - 99) = 1 / 2 = 0.5 (dead center)
    assert pytest.approx(res["range_position_5"].iloc[5]) == 0.5


def test_structure_registration():
    """Verify market structure definitions in registry."""
    reg = FeatureRegistry()
    config = StructureFeatureConfig(support_resistance_lookback=20)
    register_structure_features(reg, config, version="v1.0.0")

    assert reg.get("support_level_20").lookback == 21
    assert reg.get("breakout_20").category == "structure"
    assert reg.get("range_position_20") is not None
