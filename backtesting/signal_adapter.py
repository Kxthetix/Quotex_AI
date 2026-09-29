"""Signal input adapter translating Phase 4 intelligence outputs into backtest signal events (Phase 5).

Serves as an adapter around Phase 4 models, technical rules, patterns, and regimes.
Does NOT perform final multi-source decision fusion (reserved for Phase 6).
"""

from analysis.pattern.models import PatternSearchResult
from analysis.regime.models import MarketRegime, RegimeAssessment
from analysis.technical.models import TechnicalAssessment, TrendState
from backtesting.events import SignalEvent
from features.models import FeatureVector


class SignalAdapter:
    """Adapts Phase 4 intelligence outputs into standardized backtesting signals."""

    def __init__(
        self,
        long_prob_threshold: float = 0.55,
        exit_prob_threshold: float = 0.45,
        require_trend_alignment: bool = False,
    ) -> None:
        self.long_prob_threshold = long_prob_threshold
        self.exit_prob_threshold = exit_prob_threshold
        self.require_trend_alignment = require_trend_alignment

    def generate_signal(
        self,
        feature_vector: FeatureVector,
        direction_prob: float = 0.5,
        expected_return: float = 0.0,
        expected_volatility: float = 0.0,
        trade_quality_prob: float = 0.5,
        technical_assessment: TechnicalAssessment | None = None,
        regime_assessment: RegimeAssessment | None = None,
        pattern_result: PatternSearchResult | None = None,
        model_version: str = "v1.0.0",
        feature_version: str = "v1.0.0",
    ) -> SignalEvent:
        """Derive candidate signal action from available Phase 4 intelligence at time T."""
        regime_str = (
            regime_assessment.regime.value if regime_assessment else MarketRegime.SIDEWAYS.value
        )

        suggested_action = "HOLD"
        strength = 1.0

        # Evaluate directional conviction
        is_bullish_trend = (
            technical_assessment.trend_state == TrendState.BULLISH
            if technical_assessment
            else False
        )
        is_bearish_trend = (
            technical_assessment.trend_state == TrendState.BEARISH
            if technical_assessment
            else False
        )

        meets_long = direction_prob >= self.long_prob_threshold
        meets_exit = direction_prob <= self.exit_prob_threshold

        if self.require_trend_alignment:
            meets_long = meets_long and is_bullish_trend
            meets_exit = meets_exit or is_bearish_trend

        if meets_long:
            suggested_action = "ENTER_LONG"
            strength = float(direction_prob)
        elif meets_exit:
            suggested_action = "EXIT"
            strength = float(1.0 - direction_prob)

        return SignalEvent(
            timestamp=feature_vector.timestamp,
            symbol=feature_vector.symbol,
            direction_prob=direction_prob,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            trade_quality_prob=trade_quality_prob,
            regime=regime_str,
            suggested_action=suggested_action,
            strength=strength,
            model_version=model_version,
            feature_version=feature_version,
            metadata={
                "rsi_14": feature_vector.features.get("rsi_14"),
                "adx_14": feature_vector.features.get("adx_14"),
            },
        )
