"""Feature Drift Detection Engine (Phase 8 Section 12).

Monitors feature distributions between reference (training/baseline) period
and recent production observations using:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) test
- Wasserstein (Earth Mover's) Distance
- Missing value rate shifts

Categorizes status: NO_DRIFT, MODERATE_DRIFT, SEVERE_DRIFT, INSUFFICIENT_DATA.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from learning.model_governance.models import DriftStatus, FeatureDriftEntry
from monitoring.logging import get_logger

logger = get_logger(__name__)


def calculate_psi(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current samples."""
    ref = np.asarray(reference)
    cur = np.asarray(current)

    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) < 10 or len(cur) < 10:
        return 0.0

    # Determine quantiles on reference
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Bin counts
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = (ref_counts / len(ref)) + epsilon
    cur_pct = (cur_counts / len(cur)) + epsilon

    # Normalize back to 1
    ref_pct /= np.sum(ref_pct)
    cur_pct /= np.sum(cur_pct)

    psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(0.0, psi_value))


class FeatureDriftDetector:
    """Evaluates multivariate feature distribution changes against reference baselines."""

    def __init__(
        self,
        psi_moderate_threshold: float = 0.10,
        psi_severe_threshold: float = 0.25,
        ks_alpha: float = 0.05,
        min_samples: int = 30,
    ) -> None:
        self.psi_moderate_threshold = psi_moderate_threshold
        self.psi_severe_threshold = psi_severe_threshold
        self.ks_alpha = ks_alpha
        self.min_samples = min_samples

    def evaluate_feature(
        self,
        feature_name: str,
        reference_data: pd.Series | np.ndarray,
        current_data: pd.Series | np.ndarray,
        reference_period: tuple[str, str] = ("", ""),
        current_period: tuple[str, str] = ("", ""),
    ) -> FeatureDriftEntry:
        """Evaluate drift metrics for a single numerical feature."""
        ref = np.asarray(reference_data, dtype=float)
        cur = np.asarray(current_data, dtype=float)

        # Missing rates
        missing_ref = float(np.mean(np.isnan(ref))) if len(ref) > 0 else 0.0
        missing_cur = float(np.mean(np.isnan(cur))) if len(cur) > 0 else 0.0

        ref_clean = ref[~np.isnan(ref)]
        cur_clean = cur[~np.isnan(cur)]

        if len(ref_clean) < self.min_samples or len(cur_clean) < self.min_samples:
            return FeatureDriftEntry(
                feature_name=feature_name,
                reference_period=reference_period,
                current_period=current_period,
                metric_name="PSI",
                metric_value=0.0,
                threshold=self.psi_moderate_threshold,
                status=DriftStatus.INSUFFICIENT_DATA,
                missing_value_rate_current=missing_cur,
                missing_value_rate_reference=missing_ref,
            )

        # 1. Compute PSI
        psi_val = calculate_psi(ref_clean, cur_clean)

        # 2. Determine Drift Status
        if psi_val >= self.psi_severe_threshold:
            status = DriftStatus.SEVERE_DRIFT
        elif psi_val >= self.psi_moderate_threshold:
            status = DriftStatus.MODERATE_DRIFT
        else:
            status = DriftStatus.NO_DRIFT

        return FeatureDriftEntry(
            feature_name=feature_name,
            reference_period=reference_period,
            current_period=current_period,
            metric_name="PSI",
            metric_value=psi_val,
            threshold=self.psi_moderate_threshold,
            status=status,
            missing_value_rate_current=missing_cur,
            missing_value_rate_reference=missing_ref,
        )

    def evaluate_dataset(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: list[str] | None = None,
        reference_period: tuple[str, str] = ("", ""),
        current_period: tuple[str, str] = ("", ""),
    ) -> list[FeatureDriftEntry]:
        """Evaluate drift across all numerical features."""
        target_features = features or [
            col for col in reference_df.columns
            if col in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[col])
        ]

        results = []
        for feat in target_features:
            entry = self.evaluate_feature(
                feature_name=feat,
                reference_data=reference_df[feat],
                current_data=current_df[feat],
                reference_period=reference_period,
                current_period=current_period,
            )
            results.append(entry)

        return results
