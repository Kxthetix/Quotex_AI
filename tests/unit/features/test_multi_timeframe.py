"""Tests for Multi-Timeframe Alignment and Resampling (Phase 3 Section 28-30 & 42.G)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from features.config import MultiTimeframeConfig
from features.multi_timeframe.aligner import (
    MultiTimeframeAligner,
    register_multi_timeframe_features,
)
from features.multi_timeframe.resampler import (
    parse_timeframe_to_timedelta,
    resample_candles_to_timeframe,
)
from features.registry import FeatureRegistry


def test_timeframe_parsing():
    """Verify timeframe delta conversions."""
    assert parse_timeframe_to_timedelta("1m") == timedelta(minutes=1)
    assert parse_timeframe_to_timedelta("5m") == timedelta(minutes=5)
    assert parse_timeframe_to_timedelta("1h") == timedelta(hours=1)
    assert parse_timeframe_to_timedelta("4h") == timedelta(hours=4)
    assert parse_timeframe_to_timedelta("1d") == timedelta(days=1)


def test_resample_candles_incomplete_discarded():
    """Verify that incomplete candles at the end of the input stream are strictly discarded."""
    start = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    # Generate 25 minutes of 1m candles (10:00 to 10:25)
    rows = []
    for i in range(25):
        s = start + timedelta(minutes=i)
        e = s + timedelta(minutes=1)
        rows.append(
            {
                "start_time": s,
                "end_time": e,
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 102.0 + i,
                "volume": 10.0,
            }
        )
    df = pd.DataFrame(rows)

    # Resample to 5m: exactly 5 complete 5m candles should exist (10:00-10:05, 10:05-10:10, 10:10-10:15, 10:15-10:20, 10:20-10:25)
    res_5m = resample_candles_to_timeframe(df, target_timeframe="5m")
    assert len(res_5m) == 5
    assert res_5m["end_time"].iloc[-1] == datetime(2026, 1, 1, 10, 25, 0, tzinfo=UTC)

    # Resample to 1h: 25 minutes is INCOMPLETE for a 1h candle (needs 60 mins)
    # Incomplete candle must be discarded!
    res_1h = resample_candles_to_timeframe(df, target_timeframe="1h")
    assert len(res_1h) == 0


def test_timeframe_alignment_boundaries():
    """Verify exact boundary, before boundary, and after boundary behavior (Critical Anti-Leakage)."""
    # Create 120 minutes of 1m candles from 09:00 to 11:00
    start = datetime(2026, 1, 1, 9, 0, 0, tzinfo=UTC)
    rows = []
    for i in range(120):
        s = start + timedelta(minutes=i)
        e = s + timedelta(minutes=1)
        # Bar price increases with time
        rows.append(
            {
                "start_time": s,
                "end_time": e,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 50.0,
            }
        )
    base_df = pd.DataFrame(rows)

    config = MultiTimeframeConfig(
        enabled=True,
        enabled_timeframes=["1h"],
        higher_tf_ema_periods=[1],  # EMA(1) == close of the higher timeframe candle
        higher_tf_rsi_period=14,
        higher_tf_atr_period=14,
    )
    aligner = MultiTimeframeAligner(config=config, base_timeframe="1m")
    aligned = aligner.align_historical(base_df)

    # The 1st 1h candle runs 09:00 - 10:00 (end_time = 10:00).
    # Its close is bar 59's close (at 09:59-10:00), which is 100.5 + 59 = 159.5.

    # 1. Before boundary (at 09:59:00 end_time = index 58):
    # No 1h candle has completed yet! Values must be NaN.
    assert np.isnan(aligned["mtf_1h_ema_1"].iloc[58])

    # 2. Exact boundary (at 10:00:00 end_time = index 59):
    # The 09:00-10:00 1h candle completed at 10:00! Its EMA(1) = 159.5 is available.
    assert pytest.approx(aligned["mtf_1h_ema_1"].iloc[59]) == 159.5

    # 3. After boundary (at 10:30:00 end_time = index 89):
    # The 10:00-11:00 candle is still in progress and NOT completed.
    # Therefore, the available 1h candle is still the 09:00-10:00 one (159.5)!
    assert pytest.approx(aligned["mtf_1h_ema_1"].iloc[89]) == 159.5

    # 4. Next boundary (at 11:00:00 end_time = index 119):
    # The 10:00-11:00 candle has now completed! Its close is bar 119 close (100.5 + 119 = 219.5).
    assert pytest.approx(aligned["mtf_1h_ema_1"].iloc[119]) == 219.5


def test_multi_timeframe_live_alignment():
    """Verify align_live produces identical result at timestamp T without looking ahead."""
    start = datetime(2026, 1, 1, 9, 0, 0, tzinfo=UTC)
    rows = []
    for i in range(120):
        s = start + timedelta(minutes=i)
        e = s + timedelta(minutes=1)
        rows.append(
            {
                "start_time": s,
                "end_time": e,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 50.0,
            }
        )
    df = pd.DataFrame(rows)

    config = MultiTimeframeConfig(
        enabled=True,
        enabled_timeframes=["1h"],
        higher_tf_ema_periods=[1],
        higher_tf_rsi_period=14,
        higher_tf_atr_period=14,
    )
    aligner = MultiTimeframeAligner(config=config, base_timeframe="1m")

    # Pre-resample 1h candles
    htf_1h = resample_candles_to_timeframe(df, "1h")
    htf_map = {"1h": htf_1h}

    # At 10:30, live alignment must return 159.5
    live_res = aligner.align_live(datetime(2026, 1, 1, 10, 30, 0, tzinfo=UTC), htf_map)
    assert pytest.approx(live_res["mtf_1h_ema_1"]) == 159.5

    # At 09:30, before any completed 1h candle, must return None
    early_res = aligner.align_live(datetime(2026, 1, 1, 9, 30, 0, tzinfo=UTC), htf_map)
    assert early_res["mtf_1h_ema_1"] is None


def test_mtf_registration():
    """Verify registry stores multi-timeframe definitions."""
    reg = FeatureRegistry()
    config = MultiTimeframeConfig(enabled=True, enabled_timeframes=["5m", "1h"])
    register_multi_timeframe_features(reg, config, base_timeframe="1m", version="v1.0.0")

    assert reg.get("mtf_5m_ema_20") is not None
    assert reg.get("mtf_1h_rsi_14") is not None
    assert reg.get("mtf_1h_rsi_14").timeframe == "1h"
