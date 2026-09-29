"""Confidence calibration and Expected Calibration Error (ECE) for Phase 4.

Calibrates raw model probabilities using historical validation performance
via Platt scaling (logistic/sigmoid) or Isotonic regression without lookahead leakage.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across probability bins."""
    p1 = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p1, bins) - 1

    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            continue
        bin_size = np.sum(mask)
        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(p1[mask]))
        ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return float(ece)


class ProbabilityCalibrator:
    """Post-hoc probability calibration fitted exclusively on validation observations."""

    def __init__(self, method: str = "sigmoid") -> None:
        self.method = method.lower()
        self.is_fitted: bool = False
        if self.method in ("sigmoid", "platt"):
            self._calibrator = LogisticRegression(C=1.0)
        elif self.method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
        else:
            raise ValueError(
                f"Unsupported calibration method: '{method}' (choose 'sigmoid'/'platt' or 'isotonic')"
            )

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibration curve on validation predictions."""
        p1 = (
            val_probs[:, 1]
            if val_probs.ndim == 2 and val_probs.shape[1] >= 2
            else val_probs.ravel()
        )
        if self.method in ("sigmoid", "platt"):
            # Reshape for LogisticRegression
            self._calibrator.fit(p1.reshape(-1, 1), val_labels.astype(int))
        else:
            self._calibrator.fit(p1, val_labels.astype(float))
        self.is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities into calibrated probability estimates."""
        if not self.is_fitted:
            raise RuntimeError("ProbabilityCalibrator must be fitted before calibrate.")

        p1 = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.ravel()
        if self.method in ("sigmoid", "platt"):
            cal_p1 = self._calibrator.predict_proba(p1.reshape(-1, 1))[:, 1]
        else:
            cal_p1 = self._calibrator.predict(p1)

        cal_p1 = np.clip(cal_p1, 0.0, 1.0)
        cal_p0 = 1.0 - cal_p1
        return np.column_stack([cal_p0, cal_p1])
