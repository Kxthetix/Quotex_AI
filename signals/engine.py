"""Signal Engine — the orchestrating pipeline for Phase 6.

Assembles evidence from all intelligence sources, fuses them into a single
directional score, applies regime-adaptive thresholds, confidence adjustments,
and produces the final versioned TradingSignal.

Architecture:
    FeatureVector + Assessments + Model Predictions
                         |
                    EVIDENCE ADAPTERS (10 streams)
                         |
                    FUSION ENGINE (weighted score + threshold)
                         |
                    CONFIDENCE SCORER (regime suitability + diversity)
                         |
                    TradingSignal (BUY / SELL / NO_TRADE)

The SignalEngine is the only entry point for signal generation.
It is stateless: all state is passed in via method arguments and config objects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analysis.pattern.models import PatternSearchResult
from analysis.regime.models import RegimeAssessment
from analysis.technical.models import TechnicalAssessment
from signals.confidence.scoring import ConfidenceConfig, ConfidenceScorer
from signals.fusion.engine import FusionConfig, FusionEngine
from signals.generation.adapters import (
    MLDirectionEvidenceAdapter,
    MLReturnEvidenceAdapter,
    MLTradeQualityEvidenceAdapter,
    MLVolatilityEvidenceAdapter,
    MultiTimeframeEvidenceAdapter,
    PatternEvidenceAdapter,
    RegimeEvidenceAdapter,
    RiskRewardEvidenceAdapter,
    TechnicalEvidenceAdapter,
    VolumeEvidenceAdapter,
)
from signals.models import Evidence, SignalAction, TradingSignal


@dataclass
class IntelligenceSnapshot:
    """Collects all available intelligence outputs at timestamp T.

    Not all fields are required. The SignalEngine will extract evidence
    from whichever sources are available.
    """

    symbol: str
    timestamp: datetime

    # Phase 4 rule-based assessments
    technical_assessment: TechnicalAssessment | None = None
    regime_assessment: RegimeAssessment | None = None
    pattern_result: PatternSearchResult | None = None

    # Phase 4 ML model predictions
    direction_prob_up: float | None = None
    expected_return: float | None = None
    expected_volatility: float | None = None
    trade_quality_prob: float | None = None

    # Versioning
    model_version: str = "v1.0.0"
    feature_version: str = "v1.0.0"

    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalEngineConfig:
    """Master configuration for the Signal Engine."""

    version: str = "v1.0.0"
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    confidence_config: ConfidenceConfig = field(default_factory=ConfidenceConfig)

    # Evidence weight overrides (keyed by EvidenceType.value)
    weight_overrides: dict[str, float] = field(default_factory=dict)


class SignalEngine:
    """The unified Signal Engine for Phase 6.

    Orchestrates evidence extraction, fusion, and confidence scoring
    to produce the final BUY / SELL / NO_TRADE trading signal.

    Invariant: Stateless. Does NOT track positions, manage risk, or place orders.
    """

    def __init__(self, config: SignalEngineConfig | None = None) -> None:
        self.config = config or SignalEngineConfig()

        # Initialize sub-components
        self._fusion = FusionEngine(self.config.fusion_config)
        self._confidence = ConfidenceScorer(self.config.confidence_config)

        # Initialize evidence adapters
        self._technical = TechnicalEvidenceAdapter()
        self._pattern = PatternEvidenceAdapter()
        self._regime = RegimeEvidenceAdapter()
        self._ml_direction = MLDirectionEvidenceAdapter()
        self._ml_return = MLReturnEvidenceAdapter()
        self._ml_trade_quality = MLTradeQualityEvidenceAdapter()
        self._ml_volatility = MLVolatilityEvidenceAdapter()
        self._volume = VolumeEvidenceAdapter()
        self._mtf = MultiTimeframeEvidenceAdapter()
        self._risk_reward = RiskRewardEvidenceAdapter()

    def generate(self, snapshot: IntelligenceSnapshot) -> TradingSignal:
        """Generate a TradingSignal from a complete IntelligenceSnapshot.

        This is the primary entry point for signal generation.
        """
        # 1. Extract evidence from all available sources
        evidence_list = self._extract_evidence(snapshot)

        # 2. Determine current regime
        regime = "SIDEWAYS"
        if snapshot.regime_assessment is not None:
            regime = snapshot.regime_assessment.regime.value

        # 3. Fuse evidence into action + raw conviction
        action, fusion_score, raw_conviction = self._fusion.fuse(evidence_list, regime)

        # 4. Adjust conviction via confidence scorer
        conviction = self._confidence.adjust(raw_conviction, evidence_list, regime)

        # 5. Compute risk/reward ratio
        rr_ratio = 0.0
        if snapshot.expected_return is not None and snapshot.expected_volatility is not None:
            if snapshot.expected_volatility > 0:
                rr_ratio = abs(snapshot.expected_return) / snapshot.expected_volatility

        # 6. Assemble final signal
        return TradingSignal(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            action=action,
            conviction=conviction,
            expected_return=snapshot.expected_return or 0.0,
            expected_volatility=snapshot.expected_volatility or 0.0,
            risk_reward_ratio=round(rr_ratio, 4),
            regime=regime,
            evidence=evidence_list,
            fusion_score=fusion_score,
            signal_version=self.config.version,
            feature_version=snapshot.feature_version,
            model_version=snapshot.model_version,
            metadata=snapshot.metadata,
        )

    def _extract_evidence(self, snapshot: IntelligenceSnapshot) -> list[Evidence]:
        """Extract evidence records from all available intelligence sources."""
        evidence: list[Evidence] = []
        overrides = self.config.weight_overrides

        # 1. Technical assessment evidence
        if snapshot.technical_assessment is not None:
            evidence.append(
                self._technical.extract(
                    snapshot.technical_assessment,
                    weight=overrides.get("TECHNICAL"),
                )
            )

            # 1a. Volume evidence (derived from technical assessment)
            evidence.append(
                self._volume.extract(
                    snapshot.technical_assessment,
                    weight=overrides.get("VOLUME"),
                )
            )

            # 1b. Multi-timeframe evidence (derived from technical assessment)
            evidence.append(
                self._mtf.extract(
                    snapshot.technical_assessment,
                    weight=overrides.get("MULTI_TIMEFRAME"),
                )
            )

        # 2. Pattern evidence
        if snapshot.pattern_result is not None:
            evidence.append(
                self._pattern.extract(
                    snapshot.pattern_result,
                    weight=overrides.get("PATTERN"),
                )
            )

        # 3. Regime evidence
        if snapshot.regime_assessment is not None:
            evidence.append(
                self._regime.extract(
                    snapshot.regime_assessment,
                    weight=overrides.get("REGIME"),
                )
            )

        # 4. ML Direction evidence
        if snapshot.direction_prob_up is not None:
            evidence.append(
                self._ml_direction.extract(
                    snapshot.direction_prob_up,
                    model_version=snapshot.model_version,
                    weight=overrides.get("ML_DIRECTION"),
                )
            )

        # 5. ML Return evidence
        if snapshot.expected_return is not None:
            evidence.append(
                self._ml_return.extract(
                    snapshot.expected_return,
                    model_version=snapshot.model_version,
                    weight=overrides.get("ML_RETURN"),
                )
            )

        # 6. ML Trade Quality evidence
        if snapshot.trade_quality_prob is not None:
            evidence.append(
                self._ml_trade_quality.extract(
                    snapshot.trade_quality_prob,
                    model_version=snapshot.model_version,
                    weight=overrides.get("ML_TRADE_QUALITY"),
                )
            )

        # 7. ML Volatility evidence
        if snapshot.expected_volatility is not None:
            evidence.append(
                self._ml_volatility.extract(
                    snapshot.expected_volatility,
                    model_version=snapshot.model_version,
                    weight=overrides.get("VOLATILITY"),
                )
            )

        # 8. Risk/Reward evidence
        if snapshot.expected_return is not None and snapshot.expected_volatility is not None:
            evidence.append(
                self._risk_reward.extract(
                    snapshot.expected_return,
                    snapshot.expected_volatility,
                    weight=overrides.get("RISK_REWARD"),
                )
            )

        return evidence
