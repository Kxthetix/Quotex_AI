"""Baseline Market Regime Engine for Phase 4.

Implements robust, rule-and-statistics-based classification across the 8 required regimes:
TREND_UP, TREND_DOWN, SIDEWAYS, HIGH_VOLATILITY, LOW_VOLATILITY, BREAKOUT, CONSOLIDATION, REVERSAL.
Applies explicit, deterministic precedence when multiple regime criteria overlap.
"""

from collections.abc import Sequence

from analysis.regime.models import MarketRegime, RegimeAssessment
from features.models import FeatureVector


class MarketRegimeEngine:
    """Classifies market regime deterministically from versioned FeatureVectors."""

    def __init__(
        self,
        regime_version: str = "v1.0.0",
        adx_trend_threshold: float = 22.0,
        consolidation_threshold: float = 0.015,
        high_vol_percentile_factor: float = 1.6,
        low_vol_percentile_factor: float = 0.6,
    ) -> None:
        self.regime_version = regime_version
        self.adx_trend_threshold = adx_trend_threshold
        self.consolidation_threshold = consolidation_threshold
        self.high_vol_percentile_factor = high_vol_percentile_factor
        self.low_vol_percentile_factor = low_vol_percentile_factor

    def classify(self, vector: FeatureVector) -> RegimeAssessment:
        """Classify prevailing market regime from a single FeatureVector."""
        v = vector.values

        breakout_20 = v.get("breakout_20")
        breakdown_20 = v.get("breakdown_20")
        is_consolidating_20 = v.get("is_consolidating_20")
        range_compression_20 = v.get("range_compression_20")
        rsi_14 = v.get("rsi_14")
        candle_range = v.get("candle_range")
        upper_wick = v.get("upper_wick")
        lower_wick = v.get("lower_wick")
        adx_14 = v.get("adx_14")
        plus_di_14 = v.get("plus_di_14")
        minus_di_14 = v.get("minus_di_14")
        ema_20 = v.get("ema_20")
        ema_50 = v.get("ema_50")
        rolling_vol_20 = v.get("rolling_vol_20")
        historical_vol_20 = v.get("historical_vol_20")

        active_subregimes: list[MarketRegime] = []
        scores: dict[MarketRegime, float] = {}

        # 1. Breakout Check
        if breakout_20 == 1.0 or breakdown_20 == 1.0:
            active_subregimes.append(MarketRegime.BREAKOUT)
            scores[MarketRegime.BREAKOUT] = 0.85

        # 2. Consolidation Check
        if is_consolidating_20 == 1.0 or (
            range_compression_20 is not None and range_compression_20 < self.consolidation_threshold
        ):
            active_subregimes.append(MarketRegime.CONSOLIDATION)
            scores[MarketRegime.CONSOLIDATION] = 0.80

        # 3. Reversal Check (exhaustion wicks at oscillator extremes)
        if (
            rsi_14 is not None
            and candle_range is not None
            and candle_range > 0
            and (
                (rsi_14 >= 72.0 and upper_wick is not None and (upper_wick / candle_range) >= 0.5)
                or (
                    rsi_14 <= 28.0 and lower_wick is not None and (lower_wick / candle_range) >= 0.5
                )
            )
        ):
            active_subregimes.append(MarketRegime.REVERSAL)
            scores[MarketRegime.REVERSAL] = 0.75

        # 4. Volatility Regimes
        if rolling_vol_20 is not None and historical_vol_20 is not None and historical_vol_20 > 0:
            vol_ratio = (rolling_vol_20 * 725.0) / historical_vol_20
            if vol_ratio >= self.high_vol_percentile_factor:
                active_subregimes.append(MarketRegime.HIGH_VOLATILITY)
                scores[MarketRegime.HIGH_VOLATILITY] = min(0.90, 0.6 + vol_ratio * 0.1)
            elif vol_ratio <= self.low_vol_percentile_factor:
                active_subregimes.append(MarketRegime.LOW_VOLATILITY)
                scores[MarketRegime.LOW_VOLATILITY] = 0.70

        # 5. Trend Regimes
        is_trending = adx_14 is not None and adx_14 >= self.adx_trend_threshold
        ema_aligned_up = ema_20 is not None and ema_50 is not None and ema_20 > ema_50
        ema_aligned_down = ema_20 is not None and ema_50 is not None and ema_20 < ema_50
        di_up = plus_di_14 is not None and minus_di_14 is not None and plus_di_14 > minus_di_14
        di_down = plus_di_14 is not None and minus_di_14 is not None and minus_di_14 > plus_di_14

        if (ema_aligned_up and di_up) or (is_trending and di_up):
            active_subregimes.append(MarketRegime.TREND_UP)
            adx_val = adx_14 if adx_14 is not None else 25.0
            scores[MarketRegime.TREND_UP] = min(0.95, 0.60 + (adx_val / 100.0) * 0.35)
        elif (ema_aligned_down and di_down) or (is_trending and di_down):
            active_subregimes.append(MarketRegime.TREND_DOWN)
            adx_val = adx_14 if adx_14 is not None else 25.0
            scores[MarketRegime.TREND_DOWN] = min(0.95, 0.60 + (adx_val / 100.0) * 0.35)

        # 6. Precedence Decision Logic:
        # Precedence order: BREAKOUT > REVERSAL > CONSOLIDATION > HIGH_VOLATILITY > TREND_UP / TREND_DOWN > LOW_VOLATILITY > SIDEWAYS
        primary_regime = MarketRegime.SIDEWAYS
        confidence = 0.50

        if MarketRegime.BREAKOUT in active_subregimes:
            primary_regime = MarketRegime.BREAKOUT
            confidence = scores[MarketRegime.BREAKOUT]
        elif MarketRegime.REVERSAL in active_subregimes:
            primary_regime = MarketRegime.REVERSAL
            confidence = scores[MarketRegime.REVERSAL]
        elif MarketRegime.CONSOLIDATION in active_subregimes:
            primary_regime = MarketRegime.CONSOLIDATION
            confidence = scores[MarketRegime.CONSOLIDATION]
        elif MarketRegime.HIGH_VOLATILITY in active_subregimes and not (
            MarketRegime.TREND_UP in active_subregimes
            or MarketRegime.TREND_DOWN in active_subregimes
        ):
            primary_regime = MarketRegime.HIGH_VOLATILITY
            confidence = scores[MarketRegime.HIGH_VOLATILITY]
        elif MarketRegime.TREND_UP in active_subregimes:
            primary_regime = MarketRegime.TREND_UP
            confidence = scores[MarketRegime.TREND_UP]
        elif MarketRegime.TREND_DOWN in active_subregimes:
            primary_regime = MarketRegime.TREND_DOWN
            confidence = scores[MarketRegime.TREND_DOWN]
        elif MarketRegime.LOW_VOLATILITY in active_subregimes:
            primary_regime = MarketRegime.LOW_VOLATILITY
            confidence = scores[MarketRegime.LOW_VOLATILITY]
        else:
            primary_regime = MarketRegime.SIDEWAYS
            confidence = 0.55
            active_subregimes.append(MarketRegime.SIDEWAYS)

        supporting = {
            "adx_14": float(adx_14) if adx_14 is not None else None,
            "rsi_14": float(rsi_14) if rsi_14 is not None else None,
            "breakout_flag": bool(breakout_20 == 1.0 or breakdown_20 == 1.0),
            "range_compression": float(range_compression_20)
            if range_compression_20 is not None
            else None,
            "ema_aligned_up": ema_aligned_up,
            "ema_aligned_down": ema_aligned_down,
        }

        return RegimeAssessment(
            symbol=vector.symbol,
            timestamp=vector.timestamp,
            timeframe=vector.timeframe,
            regime=primary_regime,
            confidence=round(confidence, 4),
            regime_version=self.regime_version,
            supporting_features=supporting,
            active_subregimes=active_subregimes,
        )

    def classify_batch(self, vectors: Sequence[FeatureVector]) -> list[RegimeAssessment]:
        """Classify a sequence of FeatureVectors."""
        return [self.classify(v) for v in vectors]
