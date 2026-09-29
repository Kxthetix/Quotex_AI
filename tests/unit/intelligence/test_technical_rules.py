"""Tests for TechnicalRuleEngine (Phase 4 Section 3-5 & 46.A)."""

from datetime import UTC, datetime

from analysis.technical.models import (
    MomentumState,
    MultiTimeframeState,
    StructureState,
    TrendState,
    VolatilityState,
    VolumeState,
)
from analysis.technical.rules import TechnicalRuleEngine
from features.models import FeatureVector


def test_technical_rules_bullish_and_breakout():
    """Verify technical rules accurately evaluate bullish trend and breakout states."""
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    vec = FeatureVector(
        symbol="BTC/USDT",
        timestamp=ts,
        timeframe="1m",
        feature_version="v1.0.0",
        values={
            "ema_20": 51000.0,
            "ema_50": 50000.0,
            "slope_10": 0.005,
            "rsi_14": 65.0,
            "macd_hist": 25.0,
            "range_compression_20": 0.03,
            "rolling_vol_20": 0.002,
            "historical_vol_20": 0.5,
            "volume_spike": 1.0,
            "relative_vol_20": 2.5,
            "breakout_20": 1.0,
            "breakdown_20": 0.0,
            "is_consolidating_20": 0.0,
            "mtf_1h_ema_20": 51500.0,
            "mtf_1h_ema_50": 50500.0,
            "mtf_5m_ema_20": 51200.0,
            "mtf_5m_ema_50": 50200.0,
        },
        is_valid=True,
    )

    engine = TechnicalRuleEngine(rule_version="v1.0.0")
    assessment = engine.evaluate(vec)

    assert assessment.rule_version == "v1.0.0"
    assert assessment.symbol == "BTC/USDT"
    assert assessment.timestamp == ts
    assert assessment.trend_state == TrendState.BULLISH
    assert assessment.momentum_state == MomentumState.ACCELERATING
    assert assessment.volume_state == VolumeState.SPIKE
    assert assessment.structure_state == StructureState.BREAKOUT_UP
    assert assessment.multi_timeframe_state == MultiTimeframeState.ALIGNED_BULLISH

    # Evidence auditability
    assert assessment.evidence["ema_20_gt_50"] is True
    assert assessment.evidence["breakout_flag"] is True


def test_technical_rules_bearish_and_oversold():
    """Verify technical rules evaluate bearish trend, oversold momentum, and breakdown."""
    ts = datetime(2026, 1, 1, 13, 0, 0, tzinfo=UTC)
    vec = FeatureVector(
        symbol="BTC/USDT",
        timestamp=ts,
        timeframe="1m",
        feature_version="v1.0.0",
        values={
            "ema_20": 49000.0,
            "ema_50": 50000.0,
            "slope_10": -0.008,
            "rsi_14": 25.0,  # Oversold
            "macd_hist": -15.0,
            "range_compression_20": 0.01,  # Compressing
            "volume_spike": 0.0,
            "relative_vol_20": 0.4,  # Low volume
            "breakout_20": 0.0,
            "breakdown_20": 1.0,  # Breakdown
            "is_consolidating_20": 1.0,
            "mtf_1h_ema_20": 49000.0,
            "mtf_1h_ema_50": 50000.0,
            "mtf_5m_ema_20": 49100.0,
            "mtf_5m_ema_50": 49900.0,
        },
        is_valid=True,
    )

    engine = TechnicalRuleEngine(rule_version="v1.0.0")
    assessment = engine.evaluate(vec)

    assert assessment.trend_state == TrendState.BEARISH
    assert assessment.momentum_state == MomentumState.OVERSOLD
    assert assessment.volatility_state == VolatilityState.COMPRESSING
    assert assessment.volume_state == VolumeState.LOW_VOLUME
    assert assessment.structure_state == StructureState.BREAKDOWN_DOWN
    assert assessment.multi_timeframe_state == MultiTimeframeState.ALIGNED_BEARISH


def test_technical_rules_warmup_handling():
    """Verify technical rules output WARMING_UP when higher-timeframe features are None."""
    ts = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    vec = FeatureVector(
        symbol="BTC/USDT",
        timestamp=ts,
        timeframe="1m",
        feature_version="v1.0.0",
        values={
            "ema_20": None,  # Warming up
            "ema_50": None,
            "mtf_1h_ema_20": None,
            "mtf_5m_ema_20": None,
        },
        is_valid=False,
    )

    engine = TechnicalRuleEngine()
    assessment = engine.evaluate(vec)

    assert assessment.trend_state == TrendState.NEUTRAL
    assert assessment.multi_timeframe_state == MultiTimeframeState.WARMING_UP
    assert assessment.is_valid is False
