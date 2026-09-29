"""Regime Drift Monitoring Engine (Phase 8 Section 15).

Compares market regime distributions between historical baselines and recent periods:
- TREND_UP, TREND_DOWN, SIDEWAYS, HIGH_VOLATILITY, LOW_VOLATILITY
- Evaluates whether perceived model weakness is attributable to market regime shifts.
"""

from __future__ import annotations

from typing import Any

from learning.model_governance.models import DriftStatus
from monitoring.logging import get_logger

logger = get_logger(__name__)


class RegimeDriftDetector:
    """Evaluates macro regime composition shifts over observation windows."""

    def __init__(self, max_proportional_shift_threshold: float = 0.25) -> None:
        self.max_proportional_shift_threshold = max_proportional_shift_threshold

    def evaluate_regimes(
        self,
        reference_regimes: list[str],
        current_regimes: list[str],
    ) -> dict[str, Any]:
        """Compare regime frequencies between reference and current windows."""
        if not reference_regimes or not current_regimes:
            return {"status": str(DriftStatus.INSUFFICIENT_DATA)}

        all_regimes = sorted(list(set(reference_regimes + current_regimes)))

        ref_total = len(reference_regimes)
        cur_total = len(current_regimes)

        ref_dist = {r: reference_regimes.count(r) / ref_total for r in all_regimes}
        cur_dist = {r: current_regimes.count(r) / cur_total for r in all_regimes}

        max_shift = max(abs(cur_dist[r] - ref_dist[r]) for r in all_regimes)

        if max_shift >= (self.max_proportional_shift_threshold * 1.5):
            status = DriftStatus.SEVERE_DRIFT
        elif max_shift >= self.max_proportional_shift_threshold:
            status = DriftStatus.MODERATE_DRIFT
        else:
            status = DriftStatus.NO_DRIFT

        return {
            "status": str(status),
            "max_regime_shift": round(max_shift, 4),
            "reference_distribution": {r: round(v, 4) for r, v in ref_dist.items()},
            "current_distribution": {r: round(v, 4) for r, v in cur_dist.items()},
        }
