"""Evidence extraction adapters for Phase 6 Signal Engine.

Each adapter converts a specific Phase 4 intelligence output into a standardized
Evidence record. Adapters are stateless, deterministic functions.

Evidence Types:
1. TechnicalEvidenceAdapter — From TechnicalAssessment
2. PatternEvidenceAdapter — From PatternSearchResult
3. RegimeEvidenceAdapter — From RegimeAssessment
4. MLDirectionEvidenceAdapter — From DirectionModel predictions
5. MLReturnEvidenceAdapter — From ReturnModel predictions
6. MLTradeQualityEvidenceAdapter — From TradeQualityModel predictions
7. MLVolatilityEvidenceAdapter — From VolatilityModel predictions
8. VolumeEvidenceAdapter — From TechnicalAssessment volume state
9. MultiTimeframeEvidenceAdapter — From TechnicalAssessment MTF state
10. RiskRewardEvidenceAdapter — Computed from return/volatility estimates
"""

from analysis.pattern.models import PatternSearchResult
from analysis.regime.models import MarketRegime, RegimeAssessment
from analysis.technical.models import (
    MomentumState,
    MultiTimeframeState,
    StructureState,
    TechnicalAssessment,
    TrendState,
    VolatilityState,
    VolumeState,
)
from signals.models import Evidence, EvidenceType


class TechnicalEvidenceAdapter:
    """Convert TechnicalAssessment into a directional evidence record.

    Scoring heuristic:
    - Trend, momentum, and structure states are mapped to directional votes.
    - Confidence is derived from ADX strength and RSI distance from neutral.
    """

    DEFAULT_WEIGHT = 0.15

    def extract(self, assessment: TechnicalAssessment, weight: float | None = None) -> Evidence:
        w = weight if weight is not None else self.DEFAULT_WEIGHT

        vote = 0.0
        conf_signals: list[float] = []

        # Trend contribution
        if assessment.trend_state == TrendState.BULLISH:
            vote += 0.4
            conf_signals.append(0.7)
        elif assessment.trend_state == TrendState.BEARISH:
            vote -= 0.4
            conf_signals.append(0.7)

        # Momentum contribution
        if assessment.momentum_state == MomentumState.ACCELERATING:
            vote += 0.2
            conf_signals.append(0.6)
        elif assessment.momentum_state == MomentumState.DECELERATING:
            vote -= 0.2
            conf_signals.append(0.6)
        elif assessment.momentum_state == MomentumState.OVERBOUGHT:
            vote -= 0.15  # Contrarian: overbought suggests pullback risk
            conf_signals.append(0.5)
        elif assessment.momentum_state == MomentumState.OVERSOLD:
            vote += 0.15  # Contrarian: oversold suggests bounce potential
            conf_signals.append(0.5)

        # Structure contribution
        if assessment.structure_state == StructureState.BREAKOUT_UP:
            vote += 0.3
            conf_signals.append(0.8)
        elif assessment.structure_state == StructureState.BREAKDOWN_DOWN:
            vote -= 0.3
            conf_signals.append(0.8)

        # ADX-based confidence boost
        adx = assessment.evidence.get("adx_strength")
        if adx is not None and isinstance(adx, (int, float)):
            conf_signals.append(min(1.0, adx / 50.0))

        # Clamp vote to [-1, 1]
        vote = max(-1.0, min(1.0, vote))

        # Aggregate confidence
        confidence = sum(conf_signals) / len(conf_signals) if conf_signals else 0.5

        return Evidence(
            evidence_type=EvidenceType.TECHNICAL,
            vote=round(vote, 4),
            confidence=round(min(1.0, confidence), 4),
            weight=w,
            details={
                "trend": assessment.trend_state.value,
                "momentum": assessment.momentum_state.value,
                "structure": assessment.structure_state.value,
                "volatility": assessment.volatility_state.value,
            },
        )


class PatternEvidenceAdapter:
    """Convert PatternSearchResult into directional evidence.

    Uses the conditional distribution of historical forward returns from matched patterns.
    """

    DEFAULT_WEIGHT = 0.10
    MIN_MATCHES = 5

    def extract(self, result: PatternSearchResult, weight: float | None = None) -> Evidence:
        w = weight if weight is not None else self.DEFAULT_WEIGHT

        dist = result.distribution

        if dist.match_count < self.MIN_MATCHES:
            return Evidence(
                evidence_type=EvidenceType.PATTERN,
                vote=0.0,
                confidence=0.1,
                weight=w,
                details={"insufficient_matches": True, "match_count": dist.match_count},
            )

        # Vote direction from positive outcome ratio
        # 0.5 = neutral, >0.5 = bullish, <0.5 = bearish
        directional_bias = (dist.positive_outcome_ratio - 0.5) * 2.0
        vote = max(-1.0, min(1.0, directional_bias))

        # Confidence from match count and consistency
        count_conf = min(1.0, dist.match_count / 30.0)
        consistency = 1.0 - min(1.0, dist.std_forward_return / (abs(dist.mean_forward_return) + 1e-8))
        consistency = max(0.0, consistency)
        confidence = (count_conf * 0.4 + consistency * 0.6)

        return Evidence(
            evidence_type=EvidenceType.PATTERN,
            vote=round(vote, 4),
            confidence=round(min(1.0, confidence), 4),
            weight=w,
            details={
                "match_count": dist.match_count,
                "mean_return": dist.mean_forward_return,
                "positive_ratio": dist.positive_outcome_ratio,
                "mean_mfe": dist.mean_mfe,
                "mean_mae": dist.mean_mae,
            },
        )


class RegimeEvidenceAdapter:
    """Convert RegimeAssessment into directional evidence.

    Regime acts as a contextual filter that modulates signal conviction
    rather than providing strong directional votes.
    """

    DEFAULT_WEIGHT = 0.10

    # Regime-specific directional biases
    _REGIME_VOTES: dict[MarketRegime, float] = {
        MarketRegime.TREND_UP: 0.4,
        MarketRegime.TREND_DOWN: -0.4,
        MarketRegime.BREAKOUT: 0.2,
        MarketRegime.REVERSAL: 0.0,  # Ambiguous direction
        MarketRegime.HIGH_VOLATILITY: 0.0,
        MarketRegime.LOW_VOLATILITY: 0.0,
        MarketRegime.CONSOLIDATION: 0.0,
        MarketRegime.SIDEWAYS: 0.0,
    }

    def extract(self, assessment: RegimeAssessment, weight: float | None = None) -> Evidence:
        w = weight if weight is not None else self.DEFAULT_WEIGHT

        vote = self._REGIME_VOTES.get(assessment.regime, 0.0)
        confidence = assessment.confidence

        return Evidence(
            evidence_type=EvidenceType.REGIME,
            vote=round(vote, 4),
            confidence=round(confidence, 4),
            weight=w,
            details={
                "regime": assessment.regime.value,
                "active_subregimes": [r.value for r in assessment.active_subregimes],
            },
        )


class MLDirectionEvidenceAdapter:
    """Convert DirectionModel probability output into directional evidence.

    Expects predict_proba output shape [N, 2] where column 1 is P(UP).
    """

    DEFAULT_WEIGHT = 0.20

    def extract(
        self,
        prob_up: float,
        model_version: str = "v1.0.0",
        weight: float | None = None,
    ) -> Evidence:
        w = weight if weight is not None else self.DEFAULT_WEIGHT

        # Vote: map P(UP) from [0,1] to [-1, +1]
        vote = (prob_up - 0.5) * 2.0
        vote = max(-1.0, min(1.0, vote))

        # Confidence: distance from 0.5 decision boundary
        confidence = abs(prob_up - 0.5) * 2.0

        return Evidence(
            evidence_type=EvidenceType.ML_DIRECTION,
            vote=round(vote, 4),
            confidence=round(confidence, 4),
            weight=w,
            details={"prob_up": prob_up, "model_version": model_version},
        )


class MLReturnEvidenceAdapter:
    """Convert ReturnModel predicted return into directional evidence."""

    DEFAULT_WEIGHT = 0.15

    def extract(
        self,
        expected_return: float,
        model_version: str = "v1.0.0",
        weight: float | None = None,
    ) -> Evidence:
        w = weight if weight is not None else self.DEFAULT_WEIGHT

        # Sign of return determines vote direction
        if abs(expected_return) < 1e-6:
            vote = 0.0
        else:
            vote = 1.0 if expected_return > 0 else -1.0

        # Confidence: scaled by magnitude (capped at 2% return = full confidence)
        confidence = min(1.0, abs(expected_return) / 0.02)

        return Evidence(
            evidence_type=EvidenceType.ML_RETURN,
            vote=round(vote, 4),
            confidence=round(confidence, 4),
            weight=w,
            details={"expected_return": expected_return, "model_version": model_version},
        )


class MLTradeQualityEvidenceAdapter:
    """Convert TradeQualityModel probability into directional evidence."""

    DEFAULT_WEIGHT = 0.10

    def extract(
        self,
        prob_quality: float,
        model_version: str = "v1.0.0",
        weight: float | None = None,
    ) -> Evidence:
        w = weight if weight is not None else self.DEFAULT_WEIGHT

        # Trade quality > 0.5 means the model thinks target is hit before stop
        # This is a confidence amplifier, not a strong directional signal
        vote = (prob_quality - 0.5) * 2.0
        vote = max(-1.0, min(1.0, vote))

        confidence = prob_quality  # Direct probability as confidence

        return Evidence(
            evidence_type=EvidenceType.ML_TRADE_QUALITY,
            vote=round(vote, 4),
            confidence=round(confidence, 4),
            weight=w,
            details={"prob_quality": prob_quality, "model_version": model_version},
        )


class MLVolatilityEvidenceAdapter:
    """Convert VolatilityModel prediction into volatility evidence.

    Volatility evidence doesn't carry directional vote.
    It modulates overall conviction via the confidence field.
    Low volatility → higher conviction for directional signals.
    High volatility → lower conviction (wider uncertainty band).
    """

    DEFAULT_WEIGHT = 0.05
    BASELINE_VOLATILITY = 0.20  # ~20% annualized as baseline

    def extract(
        self,
        expected_volatility: float,
        model_version: str = "v1.0.0",
        weight: float | None = None,
    ) -> Evidence:
        w = weight if weight is not None else self.DEFAULT_WEIGHT

        # No directional vote from volatility
        vote = 0.0

        # Confidence inversely proportional to volatility
        # Lower volatility = higher confidence in other signals
        if expected_volatility <= 0:
            confidence = 0.8
        else:
            ratio = self.BASELINE_VOLATILITY / max(expected_volatility, 0.01)
            confidence = min(1.0, max(0.1, ratio * 0.5))

        return Evidence(
            evidence_type=EvidenceType.VOLATILITY,
            vote=round(vote, 4),
            confidence=round(confidence, 4),
            weight=w,
            details={
                "expected_volatility": expected_volatility,
                "baseline": self.BASELINE_VOLATILITY,
                "model_version": model_version,
            },
        )


class VolumeEvidenceAdapter:
    """Convert volume state from TechnicalAssessment into evidence.

    Volume confirms or questions the strength of directional moves.
    """

    DEFAULT_WEIGHT = 0.05

    def extract(self, assessment: TechnicalAssessment, weight: float | None = None) -> Evidence:
        w = weight if weight is not None else self.DEFAULT_WEIGHT

        # Volume doesn't have its own directional vote
        # but high volume increases confidence in existing trend
        vote = 0.0
        confidence = 0.5

        vol_state = assessment.volume_state
        if vol_state == VolumeState.SPIKE:
            confidence = 0.9
        elif vol_state == VolumeState.HIGH_VOLUME:
            confidence = 0.7
        elif vol_state == VolumeState.LOW_VOLUME:
            confidence = 0.3
        elif vol_state == VolumeState.NORMAL:
            confidence = 0.5

        return Evidence(
            evidence_type=EvidenceType.VOLUME,
            vote=round(vote, 4),
            confidence=round(confidence, 4),
            weight=w,
            details={
                "volume_state": vol_state.value,
                "relative_volume": assessment.evidence.get("relative_volume"),
            },
        )


class MultiTimeframeEvidenceAdapter:
    """Convert multi-timeframe alignment from TechnicalAssessment into evidence."""

    DEFAULT_WEIGHT = 0.05

    def extract(self, assessment: TechnicalAssessment, weight: float | None = None) -> Evidence:
        w = weight if weight is not None else self.DEFAULT_WEIGHT

        mtf = assessment.multi_timeframe_state

        if mtf == MultiTimeframeState.ALIGNED_BULLISH:
            vote = 0.6
            confidence = 0.8
        elif mtf == MultiTimeframeState.ALIGNED_BEARISH:
            vote = -0.6
            confidence = 0.8
        elif mtf == MultiTimeframeState.MIXED:
            vote = 0.0
            confidence = 0.4
        else:  # WARMING_UP
            vote = 0.0
            confidence = 0.2

        return Evidence(
            evidence_type=EvidenceType.MULTI_TIMEFRAME,
            vote=round(vote, 4),
            confidence=round(confidence, 4),
            weight=w,
            details={"multi_timeframe_state": mtf.value},
        )


class RiskRewardEvidenceAdapter:
    """Compute risk/reward evidence from return and volatility estimates.

    A favorable risk/reward ratio amplifies signal conviction.
    An unfavorable ratio suppresses it.
    """

    DEFAULT_WEIGHT = 0.05
    MIN_RISK_REWARD = 1.5  # Minimum acceptable risk/reward ratio

    def extract(
        self,
        expected_return: float,
        expected_volatility: float,
        weight: float | None = None,
    ) -> Evidence:
        w = weight if weight is not None else self.DEFAULT_WEIGHT

        if expected_volatility <= 0 or abs(expected_return) < 1e-8:
            return Evidence(
                evidence_type=EvidenceType.RISK_REWARD,
                vote=0.0,
                confidence=0.3,
                weight=w,
                details={"risk_reward_ratio": 0.0, "insufficient_data": True},
            )

        risk_reward = abs(expected_return) / expected_volatility

        # Direction aligned with return sign
        vote = 0.0
        if risk_reward >= self.MIN_RISK_REWARD:
            vote = 0.3 if expected_return > 0 else -0.3
        elif risk_reward < 0.5:
            vote = 0.0  # Poor risk/reward, no directional support

        confidence = min(1.0, risk_reward / (self.MIN_RISK_REWARD * 2.0))

        return Evidence(
            evidence_type=EvidenceType.RISK_REWARD,
            vote=round(vote, 4),
            confidence=round(confidence, 4),
            weight=w,
            details={"risk_reward_ratio": round(risk_reward, 4)},
        )
