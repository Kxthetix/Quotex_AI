"""Model Auditor Engine for Phase 8 (Sections 8, 9, 10, 16, 17, 39).

Evaluates the active production model across:
- Rolling prediction accuracy and error metrics
- Feature distribution stability (PSI)
- Model output and prediction drift
- Concept drift indicators
- Confidence calibration (ECE, Brier score)
- Macro market regime breakdown
- Strict minimum sample size enforcement (prevents false conclusions on small samples)

Generates auditable ModelAuditResult with recommendation states:
HEALTHY, MONITOR, REVIEW, RETRAIN_CANDIDATE, INSUFFICIENT_DATA.

CRITICAL INVARIANT:
The ModelAuditor only produces audit evidence.
It NEVER automatically promotes, retrains, or replaces production models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from learning.evaluation.calibration import calculate_expected_calibration_error
from learning.model_governance.drift import (
    ConceptDriftDetector,
    FeatureDriftDetector,
    PredictionDriftDetector,
    RegimeDriftDetector,
)
from learning.model_governance.models import (
    AuditRecommendation,
    DriftStatus,
    ModelAuditResult,
)
from monitoring.logging import get_logger
from paper_trading.models import PredictionRecord, PredictionStatus

logger = get_logger(__name__)


@dataclass
class AuditorConfig:
    """Configurable audit parameters and sample-size thresholds (Section 9, 10)."""
    min_audit_sample_size: int = 30
    accuracy_warning_threshold: float = 0.52   # Below 52% triggers warning
    accuracy_critical_threshold: float = 0.48  # Below 48% triggers retrain review
    max_acceptable_prediction_error: float = 0.05
    max_acceptable_ece: float = 0.28           # Expected Calibration Error
    evaluation_window_days: int = 30


class ModelAuditor:
    """Independent auditor evaluating active production models."""

    def __init__(
        self,
        config: AuditorConfig | None = None,
        feature_drift_detector: FeatureDriftDetector | None = None,
        prediction_drift_detector: PredictionDriftDetector | None = None,
        concept_drift_detector: ConceptDriftDetector | None = None,
        regime_drift_detector: RegimeDriftDetector | None = None,
    ) -> None:
        self.config = config or AuditorConfig()
        self.feature_drift = feature_drift_detector or FeatureDriftDetector()
        self.pred_drift = prediction_drift_detector or PredictionDriftDetector()
        self.concept_drift = concept_drift_detector or ConceptDriftDetector()
        self.regime_drift = regime_drift_detector or RegimeDriftDetector()

    def audit_model(
        self,
        model_version: str,
        predictions: list[PredictionRecord],
        reference_predictions: list[PredictionRecord] | None = None,
        reference_features_df: pd.DataFrame | None = None,
        current_features_df: pd.DataFrame | None = None,
        audit_window_start: datetime | None = None,
        audit_window_end: datetime | None = None,
    ) -> ModelAuditResult:
        """Run complete scientific audit of model behavior."""
        now = datetime.now(UTC)
        start_time = audit_window_start or (now - timedelta(days=self.config.evaluation_window_days))
        end_time = audit_window_end or now

        # Filter predictions within window
        window_preds = [
            p for p in predictions
            if (p.timestamp if p.timestamp.tzinfo else p.timestamp.replace(tzinfo=UTC)) >= start_time
            and (p.timestamp if p.timestamp.tzinfo else p.timestamp.replace(tzinfo=UTC)) <= end_time
            and p.model_version == model_version
        ]

        evaluated_preds = [p for p in window_preds if p.status == PredictionStatus.EVALUATED and p.actual_outcome]
        warnings: list[str] = []

        # 1. Check Sample Size Requirements (Section 10)
        if len(evaluated_preds) < self.config.min_audit_sample_size:
            return ModelAuditResult(
                model_version=model_version,
                audit_period_start=start_time,
                audit_period_end=end_time,
                prediction_count=len(window_preds),
                performance_metrics={"evaluated_count": len(evaluated_preds)},
                recommendation_state=AuditRecommendation.INSUFFICIENT_DATA,
                warnings=[
                    f"Sample size {len(evaluated_preds)} is below minimum requirement "
                    f"{self.config.min_audit_sample_size} for conclusive audit."
                ],
            )

        # 2. Performance Metrics
        correct_count = sum(1 for p in evaluated_preds if p.actual_outcome == "CORRECT")
        accuracy = correct_count / len(evaluated_preds)

        errors = [p.prediction_error for p in evaluated_preds if p.prediction_error is not None]
        avg_error = float(np.mean(errors)) if errors else 0.0

        mfes = [p.mfe for p in evaluated_preds if p.mfe is not None]
        maes = [p.mae for p in evaluated_preds if p.mae is not None]
        avg_mfe = float(np.mean(mfes)) if mfes else 0.0
        avg_mae = float(np.mean(maes)) if maes else 0.0

        perf_metrics = {
            "accuracy": round(accuracy, 4),
            "average_error": round(avg_error, 6),
            "average_mfe": round(avg_mfe, 6),
            "average_mae": round(avg_mae, 6),
            "evaluated_count": len(evaluated_preds),
            "total_predictions": len(window_preds),
        }

        # 3. Calibration Metrics (Section 16)
        probs = np.array([p.confidence for p in evaluated_preds])
        labels = np.array([1 if p.actual_outcome == "CORRECT" else 0 for p in evaluated_preds])
        ece = calculate_expected_calibration_error(labels, probs, n_bins=5)
        brier_score = float(np.mean((probs - labels) ** 2)) if len(probs) > 0 else 0.0

        calib_metrics = {
            "ece": round(ece, 4),
            "brier_score": round(brier_score, 4),
        }

        # 4. Drift Assessments (Sections 11-15)
        drift_metrics: dict[str, Any] = {}

        # Feature drift
        if reference_features_df is not None and current_features_df is not None:
            feat_drift_entries = self.feature_drift.evaluate_dataset(
                reference_features_df, current_features_df
            )
            severe_feats = [f.feature_name for f in feat_drift_entries if f.status == DriftStatus.SEVERE_DRIFT]
            moderate_feats = [f.feature_name for f in feat_drift_entries if f.status == DriftStatus.MODERATE_DRIFT]
            drift_metrics["feature_drift"] = {
                "total_features_evaluated": len(feat_drift_entries),
                "severe_drift_features": severe_feats,
                "moderate_drift_features": moderate_feats,
            }
            if severe_feats:
                warnings.append(f"Severe feature drift detected in: {', '.join(severe_feats)}")

        # Prediction drift
        if reference_predictions:
            p_drift = self.pred_drift.evaluate_predictions(reference_predictions, window_preds)
            drift_metrics["prediction_drift"] = p_drift
            if p_drift.get("status") in (str(DriftStatus.MODERATE_DRIFT), str(DriftStatus.SEVERE_DRIFT)):
                warnings.append(f"Prediction output drift observed: {p_drift.get('status')}")

        # Concept drift
        if reference_predictions:
            c_drift = self.concept_drift.evaluate_outcomes(reference_predictions, evaluated_preds)
            drift_metrics["concept_drift"] = c_drift
            if c_drift.get("status") == str(DriftStatus.SEVERE_DRIFT):
                warnings.append("Severe concept drift detected between reference and recent outcomes.")

        # 5. Regime Performance Breakdown (Section 15, 27)
        regimes = sorted(list({p.regime for p in evaluated_preds if p.regime}))
        regime_metrics: dict[str, Any] = {}
        for r in regimes:
            r_preds = [p for p in evaluated_preds if p.regime == r]
            if r_preds:
                r_correct = sum(1 for p in r_preds if p.actual_outcome == "CORRECT")
                regime_metrics[r] = {
                    "count": len(r_preds),
                    "accuracy": round(r_correct / len(r_preds), 4),
                    "mean_error": round(float(np.mean([p.prediction_error or 0.0 for p in r_preds])), 6),
                }

        # 6. Recommendation Synthesis (Section 8, 18)
        recommendation = AuditRecommendation.HEALTHY

        if accuracy < self.config.accuracy_critical_threshold:
            recommendation = AuditRecommendation.RETRAIN_CANDIDATE
            warnings.append(
                f"Accuracy {accuracy:.2%} is below critical threshold {self.config.accuracy_critical_threshold:.2%}."
            )
        elif accuracy < self.config.accuracy_warning_threshold or ece > self.config.max_acceptable_ece:
            recommendation = AuditRecommendation.REVIEW
            warnings.append(
                f"Performance degradation or calibration decay: accuracy {accuracy:.2%}, ECE {ece:.4f}."
            )
        elif drift_metrics.get("concept_drift", {}).get("status") == str(DriftStatus.SEVERE_DRIFT):
            recommendation = AuditRecommendation.RETRAIN_CANDIDATE
            warnings.append("Concept drift justifies training a candidate model.")
        elif warnings:
            recommendation = AuditRecommendation.MONITOR

        result = ModelAuditResult(
            model_version=model_version,
            audit_period_start=start_time,
            audit_period_end=end_time,
            prediction_count=len(window_preds),
            performance_metrics=perf_metrics,
            drift_metrics=drift_metrics,
            calibration_metrics=calib_metrics,
            regime_metrics=regime_metrics,
            warnings=warnings,
            recommendation_state=recommendation,
        )

        logger.info(
            "model_audit_completed",
            model_version=model_version,
            recommendation=str(recommendation),
            accuracy=accuracy,
            warnings_count=len(warnings),
        )
        return result
