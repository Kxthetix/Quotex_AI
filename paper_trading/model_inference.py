"""Live Model Inference and Version Lock Engine for Phase 7.

Enforces:
- Strict Model Version Lock at startup (Section 13): explicitly approved model versions
  are locked and never automatically replaced or promoted by live paper trading.
- Feature version compatibility verification.
- Live predictions generation with all required metadata (Section 12).
- Model anomaly monitoring and rejection of corrupted predictions (Section 28, 50).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from models.base import BaselineModel, IncompatibleFeatureError
from monitoring.logging import get_logger
from paper_trading.models import PredictionRecord, PredictionStatus
from paper_trading.monitoring import ModelAnomalyDetector

logger = get_logger(__name__)


@dataclass
class VersionLockConfig:
    """Explicit version manifest locked at startup."""
    model_version: str = "v1.0.0"
    feature_version: str = "v1.0.0"
    signal_version: str = "v1.0.0"
    configuration_version: str = "v1.0.0"
    approved_models: dict[str, str] = field(default_factory=lambda: {
        "direction": "v1.0.0",
        "volatility": "v1.0.0",
        "trade_quality": "v1.0.0",
    })


class LiveModelInferenceEngine:
    """Orchestrates live model predictions under strict version locks and safety checks."""

    def __init__(
        self,
        version_lock: VersionLockConfig | None = None,
        anomaly_detector: ModelAnomalyDetector | None = None,
    ) -> None:
        self.version_lock = version_lock or VersionLockConfig()
        self.anomaly_detector = anomaly_detector or ModelAnomalyDetector(
            expected_model_version=self.version_lock.model_version,
            expected_feature_version=self.version_lock.feature_version,
        )
        self._loaded_models: dict[str, BaselineModel] = {}
        self._is_locked: bool = True

        logger.info(
            "model_version_lock_engaged",
            model_version=self.version_lock.model_version,
            feature_version=self.version_lock.feature_version,
            signal_version=self.version_lock.signal_version,
        )

    def register_approved_model(self, name: str, model: BaselineModel) -> None:
        """Register an explicitly approved model artifact matching the version lock."""
        expected_ver = self.version_lock.approved_models.get(name, self.version_lock.model_version)
        if model.model_version != expected_ver:
            raise ValueError(
                f"Cannot load candidate model '{name}' version {model.model_version}. "
                f"Version lock strictly enforces approved version {expected_ver}."
            )
        if model.feature_version != self.version_lock.feature_version:
            raise IncompatibleFeatureError(
                f"Model feature version {model.feature_version} does not match locked feature version "
                f"{self.version_lock.feature_version}."
            )

        self._loaded_models[name] = model
        logger.info("approved_model_registered", model_name=name, version=model.model_version)

    def generate_prediction(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        current_price: float,
        regime: str = "SIDEWAYS",
        timeframe: str = "1m",
        horizon_seconds: float = 900.0,
        timestamp: datetime | None = None,
    ) -> PredictionRecord:
        """Generate a complete, immutable PredictionRecord from current features.

        Validates against anomalies before returning.
        """
        now = timestamp or datetime.now(UTC)
        if not now.tzinfo:
            now = now.replace(tzinfo=UTC)

        # Baseline directional probabilities
        prob_up = 0.50
        expected_ret = 0.001
        expected_vol = 0.01

        # Use loaded direction model if available
        dir_model = self._loaded_models.get("direction")
        if dir_model is not None and not features_df.empty:
            try:
                probs = dir_model.predict_proba(features_df)
                if len(probs) > 0:
                    prob_up = float(probs[-1][1]) if len(probs[-1]) > 1 else float(probs[-1][0])
            except Exception as e:
                logger.error("direction_model_inference_failed", error=str(e))
                # Fallback to neutral probability
                prob_up = 0.50

        # Calculate signal and confidence
        if prob_up > 0.55:
            action = "BUY"
            confidence = (prob_up - 0.50) * 2.0  # Scale [0.55, 1.0] -> [0.1, 1.0]
            target_price = current_price * (1.0 + max(0.005, expected_ret * 2))
            stop_price = current_price * (1.0 - max(0.003, expected_vol))
        elif prob_up < 0.45:
            action = "SELL"
            confidence = (0.50 - prob_up) * 2.0
            target_price = current_price * (1.0 - max(0.005, expected_ret * 2))
            stop_price = current_price * (1.0 + max(0.003, expected_vol))
        else:
            action = "NO_TRADE"
            confidence = 0.0
            target_price = current_price
            stop_price = current_price

        # Synthesize expected duration in bars (e.g. 15 bars for 1m = 15m)
        bars = max(1, int(horizon_seconds / 60))

        pred = PredictionRecord(
            prediction_id=f"pred_{uuid.uuid4().hex[:10]}",
            symbol=symbol.upper(),
            timestamp=now,
            timeframe=timeframe,
            model_version=self.version_lock.model_version,
            feature_version=self.version_lock.feature_version,
            signal_version=self.version_lock.signal_version,
            signal_action=action,
            confidence=round(confidence, 4),
            regime=regime,
            setup="STANDARD",
            expected_return=round(expected_ret, 6),
            expected_volatility=round(expected_vol, 6),
            expected_duration_bars=bars,
            entry_price=current_price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            evaluation_horizon_seconds=horizon_seconds,
            status=PredictionStatus.PENDING,
        )

        # Anomaly check (Section 28, 50)
        is_valid = self.anomaly_detector.validate_prediction(pred)
        if not is_valid:
            logger.error("prediction_rejected_due_to_anomaly", prediction_id=pred.prediction_id)
            pred.signal_action = "NO_TRADE"
            pred.confidence = 0.0
            pred.metadata["anomaly_detected"] = self.anomaly_detector.anomaly_reason

        return pred
