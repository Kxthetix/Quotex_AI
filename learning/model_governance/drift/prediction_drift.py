"""Prediction Drift Detection Engine (Phase 8 Section 13).

Monitors model output distributions:
- Confidence score drift
- Class probability distribution shift
- Expected return output shift
- Signal frequency distribution changes (BUY / SELL / NO_TRADE)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from learning.model_governance.drift.feature_drift import calculate_psi
from learning.model_governance.models import DriftStatus
from monitoring.logging import get_logger
from paper_trading.models import PredictionRecord

logger = get_logger(__name__)


class PredictionDriftDetector:
    """Monitors shifts in model output distributions between baseline and recent windows."""

    def __init__(
        self,
        psi_threshold: float = 0.15,
        min_samples: int = 30,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.min_samples = min_samples

    def evaluate_predictions(
        self,
        reference_predictions: list[PredictionRecord],
        current_predictions: list[PredictionRecord],
    ) -> dict[str, Any]:
        """Evaluate prediction output drift across confidence, return, and action distributions."""
        if len(reference_predictions) < self.min_samples or len(current_predictions) < self.min_samples:
            return {
                "status": str(DriftStatus.INSUFFICIENT_DATA),
                "sample_count_reference": len(reference_predictions),
                "sample_count_current": len(current_predictions),
                "min_required": self.min_samples,
            }

        # 1. Confidence Distribution Drift
        ref_conf = np.array([p.confidence for p in reference_predictions])
        cur_conf = np.array([p.confidence for p in current_predictions])
        psi_confidence = calculate_psi(ref_conf, cur_conf)

        # 2. Expected Return Output Drift
        ref_ret = np.array([p.expected_return for p in reference_predictions])
        cur_ret = np.array([p.expected_return for p in current_predictions])
        psi_return = calculate_psi(ref_ret, cur_ret)

        # 3. Action Frequency Drift (Categorical Chi-Square / Proportions)
        actions = ["BUY", "SELL", "NO_TRADE"]
        ref_action_counts = {a: sum(1 for p in reference_predictions if p.signal_action == a) for a in actions}
        cur_action_counts = {a: sum(1 for p in current_predictions if p.signal_action == a) for a in actions}

        ref_total = len(reference_predictions)
        cur_total = len(current_predictions)

        ref_pcts = {a: ref_action_counts[a] / ref_total for a in actions}
        cur_pcts = {a: cur_action_counts[a] / cur_total for a in actions}

        max_action_shift = max(abs(cur_pcts[a] - ref_pcts[a]) for a in actions)

        # Determine overall status
        is_drifted = (
            psi_confidence >= self.psi_threshold
            or psi_return >= self.psi_threshold
            or max_action_shift >= 0.20
        )

        overall_status = DriftStatus.MODERATE_DRIFT if is_drifted else DriftStatus.NO_DRIFT
        if psi_confidence >= 0.30 or psi_return >= 0.30 or max_action_shift >= 0.35:
            overall_status = DriftStatus.SEVERE_DRIFT

        return {
            "status": str(overall_status),
            "psi_confidence": round(psi_confidence, 4),
            "psi_expected_return": round(psi_return, 4),
            "max_action_frequency_shift": round(max_action_shift, 4),
            "reference_action_proportions": {a: round(v, 4) for a, v in ref_pcts.items()},
            "current_action_proportions": {a: round(v, 4) for a, v in cur_pcts.items()},
            "reference_mean_confidence": round(float(np.mean(ref_conf)), 4),
            "current_mean_confidence": round(float(np.mean(cur_conf)), 4),
            "reference_mean_expected_return": round(float(np.mean(ref_ret)), 6),
            "current_mean_expected_return": round(float(np.mean(cur_ret)), 6),
        }
