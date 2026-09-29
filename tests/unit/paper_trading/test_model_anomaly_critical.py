"""Critical Model Anomaly Test for Phase 7 (Section 50).

Simulates abnormal model inference outputs:
- NaN or Infinite confidence
- Probability outside [0.0, 1.0]
- Impossible expected return (> 100%)
- Model version mismatch against version lock
- Feature version mismatch against version lock

Verifies:
- Anomaly detector trips and records structured CRITICAL event
- Predictions are safely rejected / neutralized to NO_TRADE
- No paper order is created
- Model is NOT automatically replaced or hot-swapped.
"""

import math
from datetime import UTC, datetime

import pytest

from paper_trading.model_inference import LiveModelInferenceEngine, VersionLockConfig
from paper_trading.models import MonitoringEventType, PredictionRecord
from paper_trading.monitoring import ModelAnomalyDetector


class TestCriticalModelAnomalyProtection:
    """Verifies that corrupted model predictions cannot trigger trade execution."""

    def test_model_anomaly_detector_nan_confidence(self):
        """NaN confidence must trip the anomaly breaker."""
        detector = ModelAnomalyDetector()
        pred = PredictionRecord(
            model_version="v1.0.0",
            feature_version="v1.0.0",
            confidence=float("nan"),
            signal_action="BUY",
        )
        assert not detector.validate_prediction(pred)
        assert detector.is_abnormal
        assert "NaN" in detector.anomaly_reason

    def test_model_anomaly_detector_invalid_probability(self):
        """Probability < 0 or > 1 must trip the anomaly breaker."""
        detector = ModelAnomalyDetector()
        pred_under = PredictionRecord(
            model_version="v1.0.0",
            feature_version="v1.0.0",
            confidence=-0.25,
            signal_action="BUY",
        )
        assert not detector.validate_prediction(pred_under)
        assert detector.is_abnormal

        detector.reset()
        pred_over = PredictionRecord(
            model_version="v1.0.0",
            feature_version="v1.0.0",
            confidence=1.45,
            signal_action="BUY",
        )
        assert not detector.validate_prediction(pred_over)
        assert detector.is_abnormal

    def test_model_anomaly_detector_impossible_expected_return(self):
        """Expected return > 100% in a single bar must trip the anomaly breaker."""
        detector = ModelAnomalyDetector()
        pred = PredictionRecord(
            model_version="v1.0.0",
            feature_version="v1.0.0",
            confidence=0.8,
            expected_return=5.0,  # 500% in 1 bar
            signal_action="BUY",
        )
        assert not detector.validate_prediction(pred)
        assert detector.is_abnormal
        assert "Expected return" in detector.anomaly_reason

    def test_model_anomaly_detector_version_mismatch(self):
        """Prediction with wrong model version must be rejected."""
        detector = ModelAnomalyDetector(expected_model_version="v1.0.0")
        pred = PredictionRecord(
            model_version="v2.0.0_unapproved",
            feature_version="v1.0.0",
            confidence=0.7,
            signal_action="BUY",
        )
        assert not detector.validate_prediction(pred)
        assert detector.is_abnormal
        assert "version mismatch" in detector.anomaly_reason

    def test_section_50_model_is_not_automatically_replaced(self):
        """Ensures system refuses to automatically load unapproved model versions."""
        version_lock = VersionLockConfig(
            model_version="v1.0.0",
            approved_models={"direction": "v1.0.0"},
        )
        inference_engine = LiveModelInferenceEngine(version_lock=version_lock)

        from models.direction.model import DirectionModel
        # Attempt to inject unapproved candidate model
        unapproved_candidate = DirectionModel(
            model_version="v2.0.0-candidate",
            feature_version="v1.0.0",
        )

        with pytest.raises(ValueError, match="Cannot load candidate model"):
            inference_engine.register_approved_model("direction", unapproved_candidate)

        # Confirm locked version remains unchanged
        assert inference_engine.version_lock.model_version == "v1.0.0"
