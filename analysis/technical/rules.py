"""Deterministic Technical Rule Engine for Phase 4.

Transforms Phase 3 FeatureVectors into structured, versioned technical assessments.
Strict invariant: Answers 'What was measurable at time T?', never 'Should I buy or sell?'.
"""

from collections.abc import Sequence

from analysis.technical.models import (
    MomentumState,
    MultiTimeframeState,
    StructureState,
    TechnicalAssessment,
    TrendState,
    VolatilityState,
    VolumeState,
)
from features.models import FeatureVector


class TechnicalRuleEngine:
    """Evaluates deterministic, versioned technical rules across multiple market dimensions."""

    def __init__(self, rule_version: str = "v1.0.0") -> None:
        self.rule_version = rule_version

    def evaluate(self, vector: FeatureVector) -> TechnicalAssessment:
        """Evaluate a single FeatureVector and produce a structured TechnicalAssessment."""
        v = vector.values

        # 1. Trend State
        ema_20 = v.get("ema_20")
        ema_50 = v.get("ema_50")
        adx_14 = v.get("adx_14")
        slope_10 = v.get("slope_10")

        trend_state = TrendState.NEUTRAL
        if ema_20 is not None and ema_50 is not None:
            if ema_20 > ema_50 and (slope_10 is None or slope_10 > 0):
                trend_state = TrendState.BULLISH
            elif ema_20 < ema_50 and (slope_10 is None or slope_10 < 0):
                trend_state = TrendState.BEARISH

        # 2. Momentum State
        rsi_14 = v.get("rsi_14")
        macd_hist = v.get("macd_hist")

        momentum_state = MomentumState.NEUTRAL
        if rsi_14 is not None:
            if rsi_14 >= 70.0:
                momentum_state = MomentumState.OVERBOUGHT
            elif rsi_14 <= 30.0:
                momentum_state = MomentumState.OVERSOLD
            elif macd_hist is not None:
                if macd_hist > 0:
                    momentum_state = MomentumState.ACCELERATING
                elif macd_hist < 0:
                    momentum_state = MomentumState.DECELERATING

        # 3. Volatility State
        range_compression_20 = v.get("range_compression_20")
        rolling_vol_20 = v.get("rolling_vol_20")
        historical_vol_20 = v.get("historical_vol_20")

        volatility_state = VolatilityState.NORMAL
        if range_compression_20 is not None and range_compression_20 < 0.015:
            volatility_state = VolatilityState.COMPRESSING
        elif (
            rolling_vol_20 is not None
            and historical_vol_20 is not None
            and (rolling_vol_20 * 500.0 > historical_vol_20)
        ):
            # If rolling volatility is substantially elevated (> 1.8x annualized historical)
            volatility_state = VolatilityState.EXPANDING

        # 4. Volume State
        volume_spike = v.get("volume_spike")
        relative_vol_20 = v.get("relative_vol_20")

        volume_state = VolumeState.NORMAL
        if volume_spike == 1.0:
            volume_state = VolumeState.SPIKE
        elif relative_vol_20 is not None:
            if relative_vol_20 >= 1.5:
                volume_state = VolumeState.HIGH_VOLUME
            elif relative_vol_20 <= 0.6:
                volume_state = VolumeState.LOW_VOLUME

        # 5. Market Structure State
        breakout_20 = v.get("breakout_20")
        breakdown_20 = v.get("breakdown_20")
        is_consolidating_20 = v.get("is_consolidating_20")

        structure_state = StructureState.CHANNEL_RANGE
        if breakout_20 == 1.0:
            structure_state = StructureState.BREAKOUT_UP
        elif breakdown_20 == 1.0:
            structure_state = StructureState.BREAKDOWN_DOWN
        elif is_consolidating_20 == 1.0:
            structure_state = StructureState.CONSOLIDATION

        # 6. Multi-Timeframe Alignment
        mtf_1h_ema_20 = v.get("mtf_1h_ema_20")
        mtf_1h_ema_50 = v.get("mtf_1h_ema_50")
        mtf_5m_ema_20 = v.get("mtf_5m_ema_20")
        mtf_5m_ema_50 = v.get("mtf_5m_ema_50")

        if mtf_1h_ema_20 is None or mtf_5m_ema_20 is None:
            mtf_state = MultiTimeframeState.WARMING_UP
        else:
            htf_1h_bullish = mtf_1h_ema_50 is not None and mtf_1h_ema_20 > mtf_1h_ema_50
            htf_5m_bullish = mtf_5m_ema_50 is not None and mtf_5m_ema_20 > mtf_5m_ema_50
            base_bullish = trend_state == TrendState.BULLISH

            htf_1h_bearish = mtf_1h_ema_50 is not None and mtf_1h_ema_20 < mtf_1h_ema_50
            htf_5m_bearish = mtf_5m_ema_50 is not None and mtf_5m_ema_20 < mtf_5m_ema_50
            base_bearish = trend_state == TrendState.BEARISH

            if htf_1h_bullish and htf_5m_bullish and base_bullish:
                mtf_state = MultiTimeframeState.ALIGNED_BULLISH
            elif htf_1h_bearish and htf_5m_bearish and base_bearish:
                mtf_state = MultiTimeframeState.ALIGNED_BEARISH
            else:
                mtf_state = MultiTimeframeState.MIXED

        # 7. Quantitative Evidence Ledger
        evidence = {
            "ema_20_gt_50": bool(ema_20 > ema_50) if (ema_20 and ema_50) else None,
            "adx_strength": float(adx_14) if adx_14 is not None else None,
            "rsi_level": float(rsi_14) if rsi_14 is not None else None,
            "relative_volume": float(relative_vol_20) if relative_vol_20 is not None else None,
            "breakout_flag": bool(breakout_20 == 1.0) if breakout_20 is not None else False,
            "breakdown_flag": bool(breakdown_20 == 1.0) if breakdown_20 is not None else False,
            "consolidation_flag": bool(is_consolidating_20 == 1.0)
            if is_consolidating_20 is not None
            else False,
        }

        return TechnicalAssessment(
            symbol=vector.symbol,
            timestamp=vector.timestamp,
            timeframe=vector.timeframe,
            feature_version=vector.feature_version,
            rule_version=self.rule_version,
            trend_state=trend_state,
            momentum_state=momentum_state,
            volatility_state=volatility_state,
            volume_state=volume_state,
            structure_state=structure_state,
            multi_timeframe_state=mtf_state,
            evidence=evidence,
            is_valid=vector.is_valid,
        )

    def evaluate_batch(self, vectors: Sequence[FeatureVector]) -> list[TechnicalAssessment]:
        """Evaluate a sequence of FeatureVectors."""
        return [self.evaluate(v) for v in vectors]
