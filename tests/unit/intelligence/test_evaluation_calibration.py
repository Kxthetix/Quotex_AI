"""Tests for Model Evaluation Metrics and Confidence Calibration.

Covers:
- Section 33: Confidence calibration (Platt scaling, Isotonic regression, ECE)
- Section 34: Standardized evaluation metrics for classification and regression
"""

import numpy as np
import pytest

from learning.evaluation.calibration import (
    ProbabilityCalibrator,
    calculate_expected_calibration_error,
)
from learning.evaluation.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    evaluate_classifier,
    evaluate_regressor,
)


class TestEvaluationMetrics:
    """Tests for classification and regression metrics."""

    def test_evaluate_classifier(self) -> None:
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 0, 1, 1])
        y_prob = np.array(
            [
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.9, 0.1],
                [0.7, 0.3],
                [0.3, 0.7],
                [0.4, 0.6],
            ]
        )

        metrics = evaluate_classifier(y_true, y_pred, y_prob)

        assert isinstance(metrics, ClassificationMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
        assert metrics.roc_auc is not None and 0.0 <= metrics.roc_auc <= 1.0
        assert metrics.brier_score is not None and metrics.brier_score >= 0.0
        assert metrics.log_loss is not None and metrics.log_loss >= 0.0

        d = metrics.to_dict()
        assert "accuracy" in d
        assert "precision" in d
        assert "roc_auc" in d

    def test_evaluate_regressor(self) -> None:
        y_true = np.array([0.01, -0.02, 0.03, 0.00, -0.01])
        y_pred = np.array([0.012, -0.018, 0.028, 0.001, -0.009])

        metrics = evaluate_regressor(y_true, y_pred)

        assert isinstance(metrics, RegressionMetrics)
        assert metrics.mae >= 0.0
        assert metrics.rmse >= 0.0
        assert metrics.median_ae >= 0.0
        # Check strong correlation between true and predicted
        assert metrics.correlation > 0.90

        d = metrics.to_dict()
        assert "mae" in d
        assert "rmse" in d
        assert "correlation" in d


class TestConfidenceCalibration:
    """Tests for Platt and Isotonic probability calibration and ECE."""

    def test_expected_calibration_error(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        # Perfectly overconfident predictions
        y_prob = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )

        ece = calculate_expected_calibration_error(y_true, y_prob, n_bins=5)
        # When predictions are 100% accurate, ECE is 0.0
        assert pytest.approx(ece, abs=1e-5) == 0.0

    @pytest.mark.parametrize("method", ["sigmoid", "platt", "isotonic"])
    def test_calibrator_fit_and_calibrate(self, method: str) -> None:
        np.random.seed(42)
        val_probs = np.random.uniform(0.1, 0.9, size=50)
        val_labels = (val_probs > 0.5).astype(int)

        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(val_probs, val_labels)

        test_probs = np.array([0.2, 0.5, 0.8])
        calibrated = calibrator.calibrate(test_probs)

        assert calibrated.shape == (3, 2)
        # Sum of P(0) and P(1) equals 1.0
        np.testing.assert_allclose(np.sum(calibrated, axis=1), 1.0, atol=1e-5)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)
