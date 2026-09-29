"""Evidence Fusion Engine for Phase 6.

Implements weighted evidence fusion with regime-adaptive thresholds to produce
the final BUY / SELL / NO_TRADE signal decision.

Fusion Algorithm:
1. Collect all Evidence records from intelligence adapters.
2. Compute the weighted directional score: Σ(vote_i × confidence_i × weight_i).
3. Normalize by total weight: score / Σ(weight_i).
4. Apply regime-adaptive signal thresholds to classify the action.
5. Compute conviction from score magnitude and evidence agreement.

The fusion engine is stateless and versioned. All parameters are externalized
to a FusionConfig dataclass.
"""

from dataclasses import dataclass, field

from analysis.regime.models import MarketRegime
from signals.models import Evidence, SignalAction


@dataclass(frozen=True)
class FusionConfig:
    """Externalized, versioned fusion configuration.

    Thresholds determine the minimum absolute fusion score required to emit
    a BUY or SELL signal. Scores below threshold result in NO_TRADE.
    """

    version: str = "v1.0.0"

    # Default threshold for directional signal emission
    base_buy_threshold: float = 0.15
    base_sell_threshold: float = -0.15

    # Regime-specific threshold adjustments (additive)
    # Positive adjustment = harder to trigger signal (more conservative)
    # Negative adjustment = easier to trigger signal (more aggressive)
    regime_adjustments: dict[str, float] = field(default_factory=lambda: {
        MarketRegime.TREND_UP.value: -0.03,        # Easier to BUY in uptrend
        MarketRegime.TREND_DOWN.value: -0.03,       # Easier to SELL in downtrend
        MarketRegime.BREAKOUT.value: -0.02,         # Slightly easier on breakout
        MarketRegime.REVERSAL.value: 0.05,          # Harder during reversal (risky)
        MarketRegime.HIGH_VOLATILITY.value: 0.05,   # Harder during high vol
        MarketRegime.LOW_VOLATILITY.value: 0.0,     # Neutral
        MarketRegime.CONSOLIDATION.value: 0.05,     # Harder during consolidation
        MarketRegime.SIDEWAYS.value: 0.03,          # Slightly harder in sideways
    })

    # Minimum required evidence streams for signal emission
    min_evidence_count: int = 3

    # Conviction scaling parameters
    conviction_scale: float = 2.0  # Multiplier to map fusion score to [0, 1] conviction


class FusionEngine:
    """Weighted evidence fusion with regime-adaptive thresholds.

    Stateless: all state comes from the evidence list and config.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse(
        self,
        evidence_list: list[Evidence],
        regime: str = "SIDEWAYS",
    ) -> tuple[SignalAction, float, float]:
        """Fuse evidence into a signal action, fusion score, and conviction.

        Args:
            evidence_list: Scored evidence records from all intelligence adapters.
            regime: Current market regime string for threshold adaptation.

        Returns:
            Tuple of (SignalAction, fusion_score, conviction).
        """
        if len(evidence_list) < self.config.min_evidence_count:
            return SignalAction.NO_TRADE, 0.0, 0.0

        # 1. Compute weighted fusion score
        total_weighted = 0.0
        total_weight = 0.0

        for e in evidence_list:
            total_weighted += e.weighted_score
            total_weight += e.weight

        if total_weight < 1e-8:
            return SignalAction.NO_TRADE, 0.0, 0.0

        fusion_score = total_weighted / total_weight

        # 2. Regime-adaptive thresholds
        adjustment = self.config.regime_adjustments.get(regime, 0.0)
        buy_threshold = self.config.base_buy_threshold + adjustment
        sell_threshold = self.config.base_sell_threshold - adjustment  # More negative for sell

        # 3. Threshold decision
        if fusion_score >= buy_threshold:
            action = SignalAction.BUY
        elif fusion_score <= sell_threshold:
            action = SignalAction.SELL
        else:
            action = SignalAction.NO_TRADE

        # 4. Conviction: scaled magnitude of fusion score, clamped to [0, 1]
        raw_conviction = abs(fusion_score) * self.config.conviction_scale
        conviction = min(1.0, max(0.0, raw_conviction))

        # 5. Evidence agreement bonus: if most evidence agrees on direction, boost conviction
        if action != SignalAction.NO_TRADE:
            agreement = self._compute_agreement(evidence_list, action)
            conviction = min(1.0, conviction * (0.7 + 0.3 * agreement))

        return action, round(fusion_score, 6), round(conviction, 4)

    def _compute_agreement(self, evidence_list: list[Evidence], action: SignalAction) -> float:
        """Compute fraction of evidence streams that agree with the chosen action direction.

        Returns a float in [0.0, 1.0]. Higher = more evidence agreement.
        """
        if not evidence_list:
            return 0.0

        target_sign = 1.0 if action == SignalAction.BUY else -1.0
        agreeing = 0
        total_voting = 0

        for e in evidence_list:
            if abs(e.vote) < 1e-6:
                continue  # Non-directional evidence doesn't count
            total_voting += 1
            if (e.vote > 0 and target_sign > 0) or (e.vote < 0 and target_sign < 0):
                agreeing += 1

        return agreeing / total_voting if total_voting > 0 else 0.5
