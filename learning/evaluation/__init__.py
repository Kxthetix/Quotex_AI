"""Model evaluation and probability calibration package (Phase 4)."""

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

__all__ = [
    "ClassificationMetrics",
    "ProbabilityCalibrator",
    "RegressionMetrics",
    "calculate_expected_calibration_error",
    "evaluate_classifier",
    "evaluate_regressor",
]
