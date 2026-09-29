"""Tests for MarketRegimeEngine (Phase 4 Section 6-7 & 46.B)."""

from datetime import UTC, datetime

from analysis.regime.engine import MarketRegimeEngine
from analysis.regime.models import MarketRegime
from features.models import FeatureVector


def test_regime_breakout_precedence():
    """Verify that a breakout state takes highest precedence."""
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    vec = FeatureVector(
        symbol="BTC/USDT",
        timestamp=ts,
        timeframe="1m",
        values={
            "breakout_20": 1.0,  # Active breakout
            "adx_14": 35.0,  # Also trending
            "plus_di_14": 30.0,
            "minus_di_14": 10.0,
        },
    )
    engine = MarketRegimeEngine(regime_version="v1.0.0")
    assessment = engine.classify(vec)

    assert assessment.regime == MarketRegime.BREAKOUT
    assert assessment.regime_version == "v1.0.0"
    assert assessment.confidence >= 0.80
    assert MarketRegime.TREND_UP in assessment.active_subregimes


def test_regime_reversal_detection():
    """Verify reversal regime detection on extreme RSI and rejection wick."""
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    vec = FeatureVector(
        symbol="BTC/USDT",
        timestamp=ts,
        timeframe="1m",
        values={
            "breakout_20": 0.0,
            "breakdown_20": 0.0,
            "rsi_14": 78.0,  # Overbought
            "candle_range": 50.0,
            "upper_wick": 35.0,  # Large upper rejection wick (35/50 = 70% of candle)
        },
    )
    engine = MarketRegimeEngine()
    assessment = engine.classify(vec)

    assert assessment.regime == MarketRegime.REVERSAL
    assert assessment.confidence >= 0.70


def test_regime_consolidation_detection():
    """Verify consolidation regime detection on volatility compression."""
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    vec = FeatureVector(
        symbol="BTC/USDT",
        timestamp=ts,
        timeframe="1m",
        values={
            "is_consolidating_20": 1.0,
            "range_compression_20": 0.008,
            "breakout_20": 0.0,
            "breakdown_20": 0.0,
        },
    )
    engine = MarketRegimeEngine()
    assessment = engine.classify(vec)

    assert assessment.regime == MarketRegime.CONSOLIDATION
    assert assessment.confidence >= 0.75


def test_regime_trend_up_and_trend_down():
    """Verify clean classification of trending regimes."""
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    engine = MarketRegimeEngine()

    # Trend Up
    vec_up = FeatureVector(
        symbol="BTC/USDT",
        timestamp=ts,
        values={
            "adx_14": 30.0,
            "plus_di_14": 28.0,
            "minus_di_14": 12.0,
            "ema_20": 51000.0,
            "ema_50": 50000.0,
        },
    )
    res_up = engine.classify(vec_up)
    assert res_up.regime == MarketRegime.TREND_UP
    assert res_up.confidence >= 0.70

    # Trend Down
    vec_down = FeatureVector(
        symbol="BTC/USDT",
        timestamp=ts,
        values={
            "adx_14": 32.0,
            "plus_di_14": 10.0,
            "minus_di_14": 30.0,
            "ema_20": 49000.0,
            "ema_50": 50000.0,
        },
    )
    res_down = engine.classify(vec_down)
    assert res_down.regime == MarketRegime.TREND_DOWN
    assert res_down.confidence >= 0.70
