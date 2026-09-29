"""Monitoring, Telemetry, Latency, and Anomaly Detection for Phase 7.

Implements:
- Latency percentiles (p50, p95, p99) across all 8 pipeline stages (Section 31)
- End-to-end event timestamps audit trail (Section 32)
- Continuous model performance & regime breakdown (Sections 25, 26, 27)
- Model anomaly detection & safety circuit (Section 28)
- System health aggregation: HEALTHY / DEGRADED / BLOCKED / FAILED (Section 30)
- Structured monitoring events & tiered alerting (Sections 33, 35)
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from monitoring.logging import get_logger
from paper_trading.models import (
    ComponentStatus,
    LatencyMeasurement,
    MonitoringEvent,
    MonitoringEventType,
    MonitoringSeverity,
    PredictionRecord,
    PredictionStatus,
    SystemState,
)

logger = get_logger(__name__)


# ==============================================================================
# Latency Monitoring (Section 31, 32)
# ==============================================================================

class LatencyTracker:
    """Tracks latency across individual stages and computes p50, p95, p99 percentiles."""

    def __init__(self, max_samples: int = 10_000) -> None:
        self.max_samples = max_samples
        self._latencies: dict[str, deque[float]] = {
            "receive_latency": deque(maxlen=max_samples),
            "normalization_latency": deque(maxlen=max_samples),
            "candle_latency": deque(maxlen=max_samples),
            "feature_latency": deque(maxlen=max_samples),
            "inference_latency": deque(maxlen=max_samples),
            "signal_latency": deque(maxlen=max_samples),
            "risk_latency": deque(maxlen=max_samples),
            "execution_latency": deque(maxlen=max_samples),
            "total_pipeline_latency": deque(maxlen=max_samples),
        }

    def record_stage(self, stage: str, latency_ms: float) -> None:
        """Record a single stage duration in milliseconds."""
        if stage in self._latencies:
            self._latencies[stage].append(latency_ms)

    def record_measurement(self, m: LatencyMeasurement) -> None:
        """Record all applicable stage deltas from a LatencyMeasurement."""
        if m.event_time and m.receive_time:
            rec_ms = (m.receive_time - m.event_time).total_seconds() * 1000.0
            self.record_stage("receive_latency", max(0.0, rec_ms))

        if m.receive_time and m.normalize_time:
            norm_ms = (m.normalize_time - m.receive_time).total_seconds() * 1000.0
            self.record_stage("normalization_latency", max(0.0, norm_ms))

        if m.normalize_time and m.candle_time:
            c_ms = (m.candle_time - m.normalize_time).total_seconds() * 1000.0
            self.record_stage("candle_latency", max(0.0, c_ms))

        if m.candle_time and m.feature_time:
            f_ms = (m.feature_time - m.candle_time).total_seconds() * 1000.0
            self.record_stage("feature_latency", max(0.0, f_ms))

        if m.feature_time and m.prediction_time:
            inf_ms = (m.prediction_time - m.feature_time).total_seconds() * 1000.0
            self.record_stage("inference_latency", max(0.0, inf_ms))

        if m.prediction_time and m.signal_time:
            sig_ms = (m.signal_time - m.prediction_time).total_seconds() * 1000.0
            self.record_stage("signal_latency", max(0.0, sig_ms))

        if m.signal_time and m.risk_time:
            risk_ms = (m.risk_time - m.signal_time).total_seconds() * 1000.0
            self.record_stage("risk_latency", max(0.0, risk_ms))

        if m.risk_time and m.execution_time:
            exec_ms = (m.execution_time - m.risk_time).total_seconds() * 1000.0
            self.record_stage("execution_latency", max(0.0, exec_ms))

        if m.receive_time and m.execution_time:
            total_ms = (m.execution_time - m.receive_time).total_seconds() * 1000.0
            self.record_stage("total_pipeline_latency", max(0.0, total_ms))

    def get_percentiles(self, stage: str = "total_pipeline_latency") -> dict[str, float]:
        """Compute p50, p95, p99 percentiles for a stage."""
        values = list(self._latencies.get(stage, []))
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}

        return {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "count": len(values),
        }

    def get_all_summary(self) -> dict[str, dict[str, float]]:
        """Return percentiles for every tracked stage."""
        return {stage: self.get_percentiles(stage) for stage in self._latencies}


# ==============================================================================
# Model Anomaly Detection (Section 28, 50)
# ==============================================================================

class ModelAnomalyDetector:
    """Monitors model inference outputs for corruption, anomalies, or distribution collapse."""

    def __init__(
        self,
        expected_model_version: str = "v1.0.0",
        expected_feature_version: str = "v1.0.0",
        min_probability: float = 0.0,
        max_probability: float = 1.0,
        max_expected_return: float = 1.0,  # 100% in a single bar is impossible
    ) -> None:
        self.expected_model_version = expected_model_version
        self.expected_feature_version = expected_feature_version
        self.min_probability = min_probability
        self.max_probability = max_probability
        self.max_expected_return = max_expected_return

        self.is_abnormal: bool = False
        self.anomaly_reason: str = ""
        self._anomaly_events: list[MonitoringEvent] = []

    def validate_prediction(self, pred: PredictionRecord) -> bool:
        """Validate a prediction record. Returns True if valid, False if anomalous."""
        # 1. Version Mismatch
        if pred.model_version != self.expected_model_version:
            return self._trip(
                f"Model version mismatch: expected {self.expected_model_version}, got {pred.model_version}"
            )
        if pred.feature_version != self.expected_feature_version:
            return self._trip(
                f"Feature version mismatch: expected {self.expected_feature_version}, got {pred.feature_version}"
            )

        # 2. Probability bounds
        if math.isnan(pred.confidence) or math.isinf(pred.confidence):
            return self._trip("Confidence is NaN or Infinite")
        if pred.confidence < self.min_probability or pred.confidence > self.max_probability:
            return self._trip(f"Confidence {pred.confidence} outside [{self.min_probability}, {self.max_probability}]")

        # 3. Expected return sanity
        if math.isnan(pred.expected_return) or math.isinf(pred.expected_return):
            return self._trip("Expected return is NaN or Infinite")
        if abs(pred.expected_return) > self.max_expected_return:
            return self._trip(f"Expected return {pred.expected_return} exceeds maximum allowable threshold")

        return True

    def _trip(self, reason: str) -> bool:
        self.is_abnormal = True
        self.anomaly_reason = reason
        event = MonitoringEvent(
            severity=MonitoringSeverity.CRITICAL,
            component="model_anomaly_detector",
            event_type=MonitoringEventType.MODEL_ANOMALY,
            message=f"Model Anomaly Detected: {reason}",
            metadata={"reason": reason},
        )
        self._anomaly_events.append(event)
        logger.error("model_anomaly_detected", reason=reason)
        return False

    def reset(self) -> None:
        self.is_abnormal = False
        self.anomaly_reason = ""


# ==============================================================================
# Model Performance Tracker & Regime Breakdown (Sections 26, 27)
# ==============================================================================

class ModelPerformanceTracker:
    """Tracks model performance continuously, broken down by regime and model version."""

    def __init__(self) -> None:
        self._regime_metrics: dict[str, list[PredictionRecord]] = defaultdict(list)
        self._version_metrics: dict[str, list[PredictionRecord]] = defaultdict(list)

    def record_evaluated_prediction(self, pred: PredictionRecord) -> None:
        """Add an evaluated prediction into performance tracking."""
        if pred.status != PredictionStatus.EVALUATED:
            return

        regime = pred.regime or "UNKNOWN"
        self._regime_metrics[regime].append(pred)
        version = pred.model_version or "v1.0.0"
        self._version_metrics[version].append(pred)

    def get_regime_performance(self) -> dict[str, dict[str, Any]]:
        """Calculate performance broken down by market regime."""
        results: dict[str, dict[str, Any]] = {}
        for regime, preds in self._regime_metrics.items():
            count = len(preds)
            if count == 0:
                continue
            correct = sum(1 for p in preds if p.actual_outcome == "CORRECT")
            avg_return = sum(p.realized_return or 0.0 for p in preds) / count
            avg_error = sum(p.prediction_error or 0.0 for p in preds) / count
            avg_mfe = sum(p.mfe or 0.0 for p in preds) / count
            avg_mae = sum(p.mae or 0.0 for p in preds) / count

            results[regime] = {
                "prediction_count": count,
                "accuracy": round(correct / count, 4),
                "avg_realized_return": round(avg_return, 6),
                "avg_prediction_error": round(avg_error, 6),
                "avg_mfe": round(avg_mfe, 6),
                "avg_mae": round(avg_mae, 6),
            }
        return results

    def get_overall_summary(self) -> dict[str, Any]:
        all_preds = [p for sub in self._regime_metrics.values() for p in sub]
        total = len(all_preds)
        if total == 0:
            return {"total_predictions": 0, "accuracy": 0.0}

        correct = sum(1 for p in all_preds if p.actual_outcome == "CORRECT")
        avg_err = sum(p.prediction_error or 0.0 for p in all_preds) / total
        return {
            "total_evaluated_predictions": total,
            "accuracy": round(correct / total, 4),
            "avg_prediction_error": round(avg_err, 6),
            "regimes_tracked": list(self._regime_metrics.keys()),
        }


# ==============================================================================
# System Health & Alert Manager (Sections 30, 33, 35)
# ==============================================================================

class SystemHealthMonitor:
    """Aggregates all component health states and computes master system health."""

    def __init__(self) -> None:
        self.latency_tracker = LatencyTracker()
        self.anomaly_detector = ModelAnomalyDetector()
        self.performance_tracker = ModelPerformanceTracker()
        self._events: deque[MonitoringEvent] = deque(maxlen=2000)
        self.error_count: int = 0
        self.total_processed_events: int = 0

    def record_event(self, event: MonitoringEvent) -> None:
        self._events.append(event)
        if event.severity == MonitoringSeverity.CRITICAL:
            self.error_count += 1

    def determine_system_state(
        self,
        connection_status: str,
        is_stale: bool,
        is_quality_breaker_tripped: bool,
        is_risk_breaker_tripped: bool,
    ) -> SystemState:
        """Compute the master SystemState based on all component checks."""
        # 1. FAILED: Connection completely failed or critical unrecoverable state
        if connection_status == "FAILED":
            return SystemState.FAILED

        # 2. BLOCKED: Circuit breakers active or serious model anomaly
        if is_quality_breaker_tripped or self.anomaly_detector.is_abnormal:
            return SystemState.BLOCKED

        # 3. DEGRADED: Stale data, reconnecting, or risk circuit breaker
        if is_stale or connection_status in ("RECONNECTING", "DEGRADED") or is_risk_breaker_tripped:
            return SystemState.DEGRADED

        # 4. HEALTHY: All components active and nominal
        return SystemState.HEALTHY

    def get_health_report(
        self,
        connection_status: str = "CONNECTED",
        is_stale: bool = False,
        is_quality_breaker_tripped: bool = False,
        is_risk_breaker_tripped: bool = False,
    ) -> dict[str, Any]:
        """Produce full system telemetry and operational diagnostic report."""
        state = self.determine_system_state(
            connection_status, is_stale, is_quality_breaker_tripped, is_risk_breaker_tripped
        )
        return {
            "system_state": str(state),
            "connection_status": connection_status,
            "stale_data": is_stale,
            "data_quality_breaker_tripped": is_quality_breaker_tripped,
            "risk_circuit_breaker_tripped": is_risk_breaker_tripped,
            "model_anomaly": self.anomaly_detector.is_abnormal,
            "model_anomaly_reason": self.anomaly_detector.anomaly_reason,
            "error_count": self.error_count,
            "total_events_processed": self.total_processed_events,
            "latency_percentiles": self.latency_tracker.get_percentiles(),
            "model_performance": self.performance_tracker.get_overall_summary(),
        }

    def get_recent_events(self, limit: int = 50) -> list[MonitoringEvent]:
        return list(self._events)[-limit:]
