"""Signal confidence scoring and calibration for Phase 6.

Provides post-fusion confidence adjustment based on:
1. Regime suitability — some regimes are inherently more predictable
2. Evidence diversity — signals from diverse sources are more reliable
3. Historical calibration — optional Platt scaling from validation data
"""

from dataclasses import dataclass, field

import numpy as np

from analysis.regime.models import MarketRegime
from signals.models import Evidence, EvidenceType


@dataclass(frozen=True)
class ConfidenceConfig:
    """Configuration for post-fusion confidence adjustments."""

    version: str = "v1.0.0"

    # Regime suitability multipliers: how suitable is each regime for trading?
    regime_suitability: dict[str, float] = field(default_factory=lambda: {
        MarketRegime.TREND_UP.value: 1.0,
        MarketRegime.TREND_DOWN.value: 1.0,
        MarketRegime.BREAKOUT.value: 0.85,
        MarketRegime.REVERSAL.value: 0.6,
        MarketRegime.HIGH_VOLATILITY.value: 0.7,
        MarketRegime.LOW_VOLATILITY.value: 0.9,
        MarketRegime.CONSOLIDATION.value: 0.5,
        MarketRegime.SIDEWAYS.value: 0.6,
    })

    # Minimum distinct evidence types for full diversity bonus
    min_diversity_types: int = 4

    # Diversity bonus range [0, max_diversity_bonus]
    max_diversity_bonus: float = 0.15


class ConfidenceScorer:
    """Post-fusion confidence adjustments.

    Applied after the FusionEngine produces a raw conviction score.
    """

    def __init__(self, config: ConfidenceConfig | None = None) -> None:
        self.config = config or ConfidenceConfig()

    def adjust(
        self,
        raw_conviction: float,
        evidence_list: list[Evidence],
        regime: str = "SIDEWAYS",
    ) -> float:
        """Apply regime suitability and evidence diversity adjustments.

        Args:
            raw_conviction: Raw conviction from FusionEngine [0, 1].
            evidence_list: All evidence records used in fusion.
            regime: Current market regime.

        Returns:
            Adjusted conviction in [0.0, 1.0].
        """
        # 1. Regime suitability scaling
        suitability = self.config.regime_suitability.get(regime, 0.7)
        adjusted = raw_conviction * suitability

        # 2. Evidence diversity bonus
        diversity = self._compute_diversity(evidence_list)
        adjusted += diversity

        return round(min(1.0, max(0.0, adjusted)), 4)

    def _compute_diversity(self, evidence_list: list[Evidence]) -> float:
        """Compute diversity bonus from the number of distinct evidence types.

        More diverse evidence sources → higher confidence in the fused signal.
        """
        if not evidence_list:
            return 0.0

        distinct_types: set[EvidenceType] = set()
        for e in evidence_list:
            if abs(e.vote) > 1e-6 or e.confidence > 0.3:
                distinct_types.add(e.evidence_type)

        ratio = min(1.0, len(distinct_types) / self.config.min_diversity_types)
        return ratio * self.config.max_diversity_bonus


class HistoricalConfidenceCalibrator:
    """Optional Platt-scaling calibrator trained on historical signal outcomes.

    Maps raw conviction scores to calibrated probabilities using logistic regression
    fitted on a validation dataset of (conviction, actual_outcome) pairs.

    NOT fitted automatically. Must be explicitly trained and applied.
    """

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self._a: float = 1.0  # slope
        self._b: float = 0.0  # intercept

    def fit(self, convictions: np.ndarray, outcomes: np.ndarray) -> "HistoricalConfidenceCalibrator":
        """Fit logistic calibration: P(success) = 1 / (1 + exp(-(a*x + b))).

        Args:
            convictions: Raw conviction scores from past signals [N].
            outcomes: Binary outcomes (1 = signal was correct, 0 = incorrect) [N].
        """
        if len(convictions) < 10:
            raise ValueError("Need at least 10 samples for calibration fitting.")

        from sklearn.linear_model import LogisticRegression

        X = convictions.reshape(-1, 1)
        y = outcomes.astype(int)

        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X, y)

        self._a = float(lr.coef_[0, 0])
        self._b = float(lr.intercept_[0])
        self.is_fitted = True
        return self

    def calibrate(self, conviction: float) -> float:
        """Map raw conviction to calibrated probability."""
        if not self.is_fitted:
            return conviction

        logit = self._a * conviction + self._b
        # Sigmoid
        prob = 1.0 / (1.0 + np.exp(-logit))
        return float(round(min(1.0, max(0.0, prob)), 4))
