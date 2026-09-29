"""Shared fixtures and mock candle generators for Feature Engine tests."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from data.ingestion.models import Candle
from features.config import FeatureEngineConfig, get_default_feature_config
from features.engine import FeatureEngine
from features.registry import FeatureRegistry


def generate_candles(
    count: int = 300,
    start_dt: datetime | None = None,
    base_price: float = 50000.0,
    seed: int = 42,
    timeframe: str = "1m",
    symbol: str = "BTC/USDT",
) -> list[Candle]:
    """Generate deterministic sequence of Candle objects for testing."""
    np.random.seed(seed)
    start = start_dt or datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    delta = timedelta(minutes=1)

    candles: list[Candle] = []
    curr_close = base_price

    for i in range(count):
        candle_start = start + i * delta
        candle_end = candle_start + delta

        # Deterministic random walk with trend & oscillation
        step = np.sin(i / 10.0) * 15.0 + (np.random.rand() - 0.48) * 20.0
        open_p = curr_close
        close_p = open_p + step
        high_p = max(open_p, close_p) + abs(np.random.rand() * 10.0)
        low_p = min(open_p, close_p) - abs(np.random.rand() * 10.0)
        volume = float(np.random.randint(100, 500) + abs(np.sin(i / 5.0)) * 200.0)

        candles.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                open=round(open_p, 4),
                high=round(high_p, 4),
                low=round(low_p, 4),
                close=round(close_p, 4),
                volume=round(volume, 2),
                start_time=candle_start,
                end_time=candle_end,
                trades_count=100,
                is_closed=True,
            )
        )
        curr_close = close_p

    return candles


@pytest.fixture
def synthetic_candles() -> list[Candle]:
    """Fixture providing 300 deterministic 1m candles."""
    return generate_candles(count=300)


@pytest.fixture
def synthetic_df(synthetic_candles: list[Candle]) -> pd.DataFrame:
    """Fixture providing OHLCV DataFrame of 300 1m candles."""
    records = [
        {
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "symbol": c.symbol,
        }
        for c in synthetic_candles
    ]
    df = pd.DataFrame(records)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    return df


@pytest.fixture
def test_config() -> FeatureEngineConfig:
    """Fixture providing default feature config."""
    return get_default_feature_config()


@pytest.fixture
def isolated_engine(test_config: FeatureEngineConfig) -> FeatureEngine:
    """Fixture providing FeatureEngine with an isolated fresh registry."""
    reg = FeatureRegistry()
    return FeatureEngine(config=test_config, registry=reg)
