"""Unit tests for Market Regime Presets and Dataset Builder."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from analysis.regime.catalog import (
    REGIME_PRESETS,
    RegimeCategory,
    RegimePreset,
    get_regime_preset,
    list_regime_presets,
)
from analysis.regime.dataset_builder import RegimeDatasetBuilder
from data.ingestion.models import Candle
from data.storage.parquet_store import load_candles_from_parquet


def test_regime_catalog_definitions():
    """Verify standard regime presets are present and well-formed."""
    presets = list_regime_presets()
    assert len(presets) >= 7

    bull = get_regime_preset("bull_parabolic_2020_2021")
    assert bull.category == RegimeCategory.BULL_EXPANSION
    assert bull.duration_days > 150
    assert bull.start < bull.end

    bear = get_regime_preset("bear_market_2022")
    assert bear.category == RegimeCategory.BEAR_CAPITULATION
    assert bear.duration_days > 300

    cycle = get_regime_preset("multi_year_cycle_2021_2024")
    assert cycle.category == RegimeCategory.MULTI_YEAR_CYCLE
    assert cycle.duration_days >= 1000

    with pytest.raises(KeyError):
        get_regime_preset("non_existent_preset_xyz")


def test_dataset_builder_profile_and_export(tmp_path: Path):
    """Verify RegimeDatasetBuilder validates, builds features, classifies regimes, and exports."""
    start_dt = datetime(2023, 1, 1, 0, 0, tzinfo=UTC)
    delta = timedelta(hours=1)
    candles: list[Candle] = []

    price = 20000.0
    for i in range(100):
        # Create an artificial trend up then chop
        change = 100.0 if i < 50 else (10.0 if i % 2 == 0 else -10.0)
        c_open = price
        c_close = price + change
        c_high = max(c_open, c_close) + 20.0
        c_low = min(c_open, c_close) - 20.0
        c = Candle(
            symbol="BTCUSDT",
            timeframe="1h",
            open=c_open,
            high=c_high,
            low=c_low,
            close=c_close,
            volume=500.0,
            trades_count=1000,
            start_time=start_dt + i * delta,
            end_time=start_dt + (i + 1) * delta,
            is_closed=True,
        )
        candles.append(c)
        price = c_close

    builder = RegimeDatasetBuilder()
    df, profile = builder.build_and_profile(
        candles=candles,
        dataset_name="test_synth",
        output_dir=tmp_path,
    )

    assert len(df) == 100
    assert "regime" in df.columns
    assert "regime_confidence" in df.columns
    assert profile.total_candles == 100
    assert profile.symbol == "BTCUSDT"
    assert profile.timeframe == "1h"
    assert profile.total_return_pct > 0
    assert len(profile.regime_distribution) > 0

    # Verify exported Parquet can be loaded back
    raw_loaded = load_candles_from_parquet(Path(profile.output_parquet_path))
    assert len(raw_loaded) == 100

    annotated_df = pd.read_parquet(profile.output_annotated_parquet_path)
    assert len(annotated_df) == 100
    assert "regime" in annotated_df.columns
