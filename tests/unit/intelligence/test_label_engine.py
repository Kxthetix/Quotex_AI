"""Tests for LabelEngine (Phase 4 Section 15-20 & 46.D)."""

import numpy as np
import pandas as pd
import pytest

from learning.labels.engine import LabelEngine


def test_direction_and_return_labels():
    """Verify forward direction and return calculation."""
    df = pd.DataFrame(
        {
            "close": [100.0, 102.0, 101.0, 105.0, 104.0],
            "high": [101.0, 103.0, 102.0, 106.0, 105.0],
            "low": [99.0, 101.0, 100.0, 104.0, 103.0],
        }
    )
    engine = LabelEngine(horizons=[1, 2], label_version="v1.0.0")
    labels = engine.compute_labels(df)

    # Horizon 1:
    # At index 0: close is 100.0, next close is 102.0 -> return is +0.02 (2.0%), direction is 1.0
    assert pytest.approx(labels["return_1"].iloc[0]) == 0.02
    assert labels["direction_1"].iloc[0] == 1.0

    # At index 1: close is 102.0, next close is 101.0 -> return is -0.0098039, direction is 0.0
    assert labels["direction_1"].iloc[1] == 0.0

    # Horizon 2:
    # At index 0: close is 100.0, close at index 2 is 101.0 -> return is +0.01 (1.0%), direction is 1.0
    assert pytest.approx(labels["return_2"].iloc[0]) == 0.01
    assert labels["direction_2"].iloc[0] == 1.0

    # Tail rows where horizon extends past data must be NaN
    assert np.isnan(labels["return_1"].iloc[4])
    assert np.isnan(labels["direction_1"].iloc[4])
    assert np.isnan(labels["return_2"].iloc[3])
    assert np.isnan(labels["return_2"].iloc[4])


def test_trade_quality_label_profit_target_vs_stop():
    """Verify trade quality target is 1.0 when profit target hit before stop loss."""
    # Price starts at 100.0. Target +1.0% = 101.0. Stop -1.0% = 99.0.
    # Bar 1 high reaches 101.5 (hits target first!)
    df = pd.DataFrame(
        {
            "close": [100.0, 101.2, 98.5],
            "high": [100.5, 101.5, 99.0],
            "low": [99.5, 100.2, 98.0],
        }
    )
    engine = LabelEngine(
        horizons=[2],
        trade_quality_target_pct=0.01,  # 1% target
        trade_quality_stop_pct=0.01,  # 1% stop
    )
    labels = engine.compute_labels(df)

    # At index 0, high on next bar (index 1) hits 101.5 >= 101.0 -> success!
    assert labels["trade_quality_2"].iloc[0] == 1.0


def test_label_catalog_definitions():
    """Verify label definitions catalog is versioned and auditable."""
    engine = LabelEngine(horizons=[5, 15], label_version="v1.0.0")
    defs = engine.get_definitions()

    assert len(defs) == 8  # 4 labels * 2 horizons
    names = [d.name for d in defs]
    assert "direction_5" in names
    assert "return_15" in names
    assert "volatility_5" in names
    assert "trade_quality_15" in names
    assert all(d.version == "v1.0.0" for d in defs)
