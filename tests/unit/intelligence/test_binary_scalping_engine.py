"""Unit tests for the BinaryScalpingEngine."""

import numpy as np
import pandas as pd
import pytest

from signals.binary_scalping_engine import (
    BinaryAction,
    BinaryScalpingEngine,
    CandleGeometry,
)


def _generate_synthetic_ohlcv(
    count: int = 50,
    base_price: float = 1.0000,
    trend: float = 0.0002,
    volatility: float = 0.0001,
) -> pd.DataFrame:
    """Generate OHLCV series with controlled trend and volatility."""
    records = []
    curr = base_price
    for i in range(count):
        curr += trend
        noise = (i % 3 - 1) * volatility
        c_open = curr - 0.00005
        c_close = curr + noise
        c_high = max(c_open, c_close) + 0.0001
        c_low = min(c_open, c_close) - 0.0001
        records.append({
            "open": c_open,
            "high": c_high,
            "low": c_low,
            "close": c_close,
            "volume": 1000 + (i * 10),
        })
    return pd.DataFrame(records)


def test_insufficient_data():
    engine = BinaryScalpingEngine()
    df = _generate_synthetic_ohlcv(count=10)
    result = engine.evaluate(df, asset="EUR/CHF")
    assert result.action == BinaryAction.NO_TRADE
    assert "INSUFFICIENT_DATA" in result.rejection_flags


def test_rsi_overbought_blocks_call():
    engine = BinaryScalpingEngine(rsi_overbought=65.0)
    # Strongly rising price series pushing RSI above 70
    df = _generate_synthetic_ohlcv(count=40, trend=0.001)
    result = engine.evaluate(df, asset="EUR/CHF", crowd_sentiment_bullish_pct=95.0)
    assert result.action != BinaryAction.CALL
    assert "RSI_OVERBOUGHT" in result.rejection_flags
    assert "CROWD_TRAP_BULLISH" in result.rejection_flags


def test_rsi_oversold_blocks_put():
    engine = BinaryScalpingEngine(rsi_oversold=35.0)
    # Strongly dropping price series pushing RSI below 30
    df = _generate_synthetic_ohlcv(count=40, trend=-0.001)
    result = engine.evaluate(df, asset="EUR/CHF", crowd_sentiment_bullish_pct=5.0)
    assert result.action != BinaryAction.PUT
    assert "RSI_OVERSOLD" in result.rejection_flags
    assert "CROWD_TRAP_BEARISH" in result.rejection_flags


def test_candle_geometry_hammer():
    # Lower wick is large, upper wick is small, close > open
    c = BinaryScalpingEngine.analyze_candle_geometry(
        c_open=1.0000,
        c_high=1.0006,
        c_low=0.9980,
        c_close=1.0005,
    )
    assert c.is_hammer is True
    assert c.is_shooting_star is False


def test_candle_geometry_shooting_star():
    # Upper wick is large, lower wick is small, close < open
    c = BinaryScalpingEngine.analyze_candle_geometry(
        c_open=1.0005,
        c_high=1.0030,
        c_low=1.0000,
        c_close=1.0002,
    )
    assert c.is_shooting_star is True
    assert c.is_hammer is False


def test_doji_blocks_trade():
    engine = BinaryScalpingEngine()
    df = _generate_synthetic_ohlcv(count=40)
    # Force last bar to be an exact Doji (open == close)
    df.loc[len(df) - 1, "open"] = 1.0000
    df.loc[len(df) - 1, "close"] = 1.0000
    df.loc[len(df) - 1, "high"] = 1.0010
    df.loc[len(df) - 1, "low"] = 0.9990

    result = engine.evaluate(df, asset="EUR/CHF")
    assert result.action == BinaryAction.NO_TRADE
    assert "DOJI_INDECISION" in result.rejection_flags


def test_incomplete_candle_rejected():
    from datetime import UTC, datetime, timedelta
    from data.ingestion.models import Candle

    engine = BinaryScalpingEngine()
    start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    candles = [
        Candle(
            symbol="EUR/CHF",
            timeframe="1m",
            open=1.0,
            high=1.01,
            low=0.99,
            close=1.005,
            volume=100.0,
            start_time=start + timedelta(minutes=i),
            end_time=start + timedelta(minutes=i + 1),
            trades_count=10,
            is_closed=(i < 29),  # Last candle is unclosed / forming
        )
        for i in range(30)
    ]

    result = engine.evaluate(candles, asset="EUR/CHF")
    assert result.action == BinaryAction.NO_TRADE
    assert "INCOMPLETE_CANDLE" in result.rejection_flags


def test_incomplete_candle_in_dataframe_rejected():
    engine = BinaryScalpingEngine()
    df = _generate_synthetic_ohlcv(count=40)
    df["is_closed"] = True
    df.loc[len(df) - 1, "is_closed"] = False

    result = engine.evaluate(df, asset="EUR/CHF")
    assert result.action == BinaryAction.NO_TRADE
    assert "INCOMPLETE_CANDLE" in result.rejection_flags


def test_invalid_volatility_flag():
    engine = BinaryScalpingEngine()
    # Completely flat price series (zero range everywhere)
    df = pd.DataFrame([{
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0
    } for _ in range(35)])

    result = engine.evaluate(df, asset="EUR/CHF")
    assert result.action == BinaryAction.NO_TRADE
    assert "INVALID_VOLATILITY" in result.rejection_flags


def test_trend_consolidation_flag():
    engine = BinaryScalpingEngine(ema_separation_threshold=0.01)
    # Tightly oscillating series with near-zero EMA separation
    df = _generate_synthetic_ohlcv(count=40, trend=0.0, volatility=0.00001)
    result = engine.evaluate(df, asset="EUR/CHF")
    assert result.action == BinaryAction.NO_TRADE
    assert "TREND_CONSOLIDATION" in result.rejection_flags


def test_no_pullback_flag():
    engine = BinaryScalpingEngine()
    # Monotonically rocket price up with zero retracement
    df = _generate_synthetic_ohlcv(count=40, trend=0.005, volatility=0.00001)
    result = engine.evaluate(df, asset="EUR/CHF")
    assert result.action == BinaryAction.NO_TRADE
    assert "NO_PULLBACK_SETUP" in result.rejection_flags


def test_price_extended_from_mean_flag():
    engine = BinaryScalpingEngine()
    df = _generate_synthetic_ohlcv(count=40, trend=0.0001, volatility=0.0001)
    # Explode the last candle 20x ATR away from EMA-21
    df.loc[len(df) - 1, "open"] = 1.0050
    df.loc[len(df) - 1, "close"] = 1.0500
    df.loc[len(df) - 1, "high"] = 1.0510
    df.loc[len(df) - 1, "low"] = 1.0040

    result = engine.evaluate(df, asset="EUR/CHF")
    assert result.action == BinaryAction.NO_TRADE
    assert "PRICE_EXTENDED_FROM_MEAN" in result.rejection_flags
