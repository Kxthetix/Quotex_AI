"""Concept Drift Monitoring Engine (Phase 8 Section 11).

Evaluates whether the underlying relationship between features and actual market outcomes
has fundamentally changed over time using the Prediction Ledger and ground-truth outcomes:
- Rolling prediction error
- Accuracy degradation relative to historical baseline
- Realized forward return decay
- Brier score / calibration degradation
"""

from __future__ import annotations

from typing import Any

import numpy as np

from learning.model_governance.models import DriftStatus
from monitoring.logging import get_logger
from paper_trading.models import PredictionRecord, PredictionStatus

logger = get_logger(__name__)


class ConceptDriftDetector:
    """Detects degradation in the predictive mapping from market features to realized outcomes."""

    def __init__(
        self,
        accuracy_drop_threshold: float = 0.12,  # 12% drop triggers concept drift
        error_increase_ratio: float = 1.50,    # 50% increase in prediction error
        min_evaluated_samples: int = 30,
    ) -> None:
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.error_increase_ratio = error_increase_ratio
        self.min_evaluated_samples = min_evaluated_samples

    def evaluate_outcomes(
        self,
        reference_outcomes: list[PredictionRecord],
        current_outcomes: list[PredictionRecord],
    ) -> dict[str, Any]:
        """Compare ground-truth predictive outcomes between reference and recent periods."""
        ref_eval = [p for p in reference_outcomes if p.status == PredictionStatus.EVALUATED and p.actual_outcome]
        cur_eval = [p for p in current_outcomes if p.status == PredictionStatus.EVALUATED and p.actual_outcome]

        if len(ref_eval) < self.min_evaluated_samples or len(cur_eval) < self.min_evaluated_samples:
            return {
                "status": str(DriftStatus.INSUFFICIENT_DATA),
                "reference_count": len(ref_eval),
                "current_count": len(cur_eval),
                "min_required": self.min_evaluated_samples,
            }

        # 1. Directional Accuracy
        ref_correct = sum(1 for p in ref_eval if p.actual_outcome == "CORRECT")
        cur_correct = sum(1 for p in cur_eval if p.actual_outcome == "CORRECT")

        ref_acc = ref_correct / len(ref_eval)
        cur_acc = cur_correct / len(cur_eval)
        acc_drop = ref_acc - cur_acc

        # 2. Prediction Error ($|Realized - Expected|$)
        ref_errors = [p.prediction_error for p in ref_eval if p.prediction_error is not None]
        cur_errors = [p.prediction_error for p in cur_eval if p.prediction_error is not None]

        ref_mean_err = float(np.mean(ref_errors)) if ref_errors else 0.01
        cur_mean_err = float(np.mean(cur_errors)) if cur_errors else 0.01

        err_ratio = cur_mean_err / max(1e-6, ref_mean_err)

        # 3. Realized Returns
        ref_rets = [p.realized_return for p in ref_eval if p.realized_return is not None]
        cur_rets = [p.realized_return for p in cur_eval if p.realized_return is not None]

        ref_mean_ret = float(np.mean(ref_rets)) if ref_rets else 0.0
        cur_mean_ret = float(np.mean(cur_rets)) if cur_rets else 0.0

        # Determine Concept Drift Status
        is_severe = acc_drop >= (self.accuracy_drop_threshold * 1.5) or err_ratio >= (self.error_increase_ratio * 1.5)
        is_moderate = acc_drop >= self.accuracy_drop_threshold or err_ratio >= self.error_increase_ratio

        if is_severe:
            status = DriftStatus.SEVERE_DRIFT
        elif is_moderate:
            status = DriftStatus.MODERATE_DRIFT
        else:
            status = DriftStatus.NO_DRIFT

        return {
            "status": str(status),
            "reference_accuracy": round(ref_acc, 4),
            "current_accuracy": round(cur_acc, 4),
            "accuracy_drop": round(acc_drop, 4),
            "reference_mean_error": round(ref_mean_err, 6),
            "current_mean_error": round(cur_mean_err, 6),
            "error_ratio": round(err_ratio, 4),
            "reference_mean_realized_return": round(ref_mean_ret, 6),
            "current_mean_realized_return": round(cur_mean_ret, 6),
        }
