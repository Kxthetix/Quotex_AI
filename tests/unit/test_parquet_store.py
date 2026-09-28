"""Unit tests for Parquet offline storage."""

from datetime import UTC, datetime
from pathlib import Path

from data.ingestion.models import Candle
from data.storage.parquet_store import load_candles_from_parquet, save_candles_to_parquet


def test_parquet_save_and_load(tmp_path: Path):
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 12, 1, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 12, 2, 0, tzinfo=UTC)

    candles = [
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=65000.0,
            high=65100.0,
            low=64950.0,
            close=65050.0,
            volume=12.5,
            trades_count=100,
            start_time=t0,
            end_time=t1,
        ),
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=65050.0,
            high=65200.0,
            low=65000.0,
            close=65180.0,
            volume=8.3,
            trades_count=85,
            start_time=t1,
            end_time=t2,
        ),
    ]

    parquet_file = tmp_path / "test_candles.parquet"
    save_candles_to_parquet(candles, parquet_file)

    assert parquet_file.exists()

    loaded = load_candles_from_parquet(parquet_file)
    assert len(loaded) == 2
    assert loaded[0].symbol == "BTCUSDT"
    assert loaded[0].open == 65000.0
    assert loaded[0].close == 65050.0
    assert loaded[1].volume == 8.3
    assert loaded[0].start_time == t0
    assert loaded[0].is_closed is True
