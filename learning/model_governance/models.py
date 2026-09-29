"""Governance models and schemas for Phase 8 Learning Loop.

Defines:
- Explicit model lifecycle states (Section 35)
- Lineage provenance records (Section 7)
- Validation result structures and gate statuses (Sections 30, 31)
- Promotion and Rollback audit records (Sections 33, 34)
- Audit recommendation states (Section 8)
- Drift metrics and report schemas (Sections 11-15, 40)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from models.base import ModelStatus


# ==============================================================================
# Model Lifecycle States (Section 35)
# ==============================================================================

class ModelLifecycleState(StrEnum):
    """Explicit lifecycle states for models in the registry."""
    DEVELOPMENT = "DEVELOPMENT"
    CANDIDATE = "CANDIDATE"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    APPROVED = "APPROVED"
    PRODUCTION = "PRODUCTION"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"
    ARCHIVED = "ARCHIVED"


# ==============================================================================
# Validation Gate Statuses (Section 30)
# ==============================================================================

class GateStatus(StrEnum):
    """Result of an individual validation gate check."""
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class GateResult:
    """Result of an individual validation gate."""
    gate_name: str
    status: GateStatus
    metric_name: str
    observed_value: float
    threshold_value: float
    details: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "status": str(self.status),
            "metric_name": self.metric_name,
            "observed_value": self.observed_value,
            "threshold_value": self.threshold_value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationResult:
    """Consolidated outcome of all validation gates for a candidate model (Section 31)."""
    validation_id: str = field(default_factory=lambda: f"val_{uuid.uuid4().hex[:10]}")
    candidate_model_version: str = ""
    production_model_version: str = ""
    dataset_version: str = ""
    feature_version: str = ""
    configuration_version: str = "v1.0.0"
    gate_results: list[GateResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    final_status: GateStatus = GateStatus.FAIL
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "candidate_model_version": self.candidate_model_version,
            "production_model_version": self.production_model_version,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "configuration_version": self.configuration_version,
            "gate_results": [g.to_dict() for g in self.gate_results],
            "metrics": self.metrics,
            "warnings": self.warnings,
            "final_status": str(self.final_status),
            "timestamp": self.timestamp.isoformat(),
        }


# ==============================================================================
# Lineage, Promotion & Rollback Records (Sections 7, 33, 34)
# ==============================================================================

@dataclass
class ModelLineageRecord:
    """Full auditable lineage for a trained model (Section 7)."""
    model_version: str
    parent_production_model: str | None
    dataset_version: str
    feature_version: str
    label_version: str
    training_period: tuple[str, str]
    validation_period: tuple[str, str]
    test_period: tuple[str, str]
    code_version: str = "v1.0.0"
    configuration_version: str = "v1.0.0"
    random_seed: int = 42
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: str = "system"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "parent_production_model": self.parent_production_model,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "training_period": list(self.training_period),
            "validation_period": list(self.validation_period),
            "test_period": list(self.test_period),
            "code_version": self.code_version,
            "configuration_version": self.configuration_version,
            "random_seed": self.random_seed,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }


@dataclass
class PromotionRecord:
    """Auditable record of a controlled model promotion (Section 33)."""
    promotion_id: str = field(default_factory=lambda: f"prom_{uuid.uuid4().hex[:10]}")
    old_production_model: str = ""
    new_production_model: str = ""
    validation_id: str = ""
    approved_by: str = ""
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    configuration_version: str = "v1.0.0"
    code_version: str = "v1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotion_id": self.promotion_id,
            "old_production_model": self.old_production_model,
            "new_production_model": self.new_production_model,
            "validation_id": self.validation_id,
            "approved_by": self.approved_by,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "configuration_version": self.configuration_version,
            "code_version": self.code_version,
        }


@dataclass
class RollbackRecord:
    """Auditable record of a model rollback to previous known-good model (Section 34)."""
    rollback_id: str = field(default_factory=lambda: f"rbk_{uuid.uuid4().hex[:10]}")
    problematic_model: str = ""
    restored_model: str = ""
    reason: str = ""
    initiated_by: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "problematic_model": self.problematic_model,
            "restored_model": self.restored_model,
            "reason": self.reason,
            "initiated_by": self.initiated_by,
            "timestamp": self.timestamp.isoformat(),
        }


# ==============================================================================
# Model Auditor Recommendation States (Sections 8, 18)
# ==============================================================================

class AuditRecommendation(StrEnum):
    """Recommendation state produced by ModelAuditor."""
    HEALTHY = "HEALTHY"
    MONITOR = "MONITOR"
    REVIEW = "REVIEW"
    RETRAIN_CANDIDATE = "RETRAIN_CANDIDATE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class ModelAuditResult:
    """Structured report produced by ModelAuditor (Section 8, 39)."""
    audit_id: str = field(default_factory=lambda: f"adt_{uuid.uuid4().hex[:10]}")
    model_version: str = ""
    audit_period_start: datetime | None = None
    audit_period_end: datetime | None = None
    prediction_count: int = 0
    performance_metrics: dict[str, float] = field(default_factory=dict)
    drift_metrics: dict[str, Any] = field(default_factory=dict)
    calibration_metrics: dict[str, float] = field(default_factory=dict)
    regime_metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    recommendation_state: AuditRecommendation = AuditRecommendation.HEALTHY
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "model_version": self.model_version,
            "audit_period_start": self.audit_period_start.isoformat() if self.audit_period_start else None,
            "audit_period_end": self.audit_period_end.isoformat() if self.audit_period_end else None,
            "prediction_count": self.prediction_count,
            "performance_metrics": self.performance_metrics,
            "drift_metrics": self.drift_metrics,
            "calibration_metrics": self.calibration_metrics,
            "regime_metrics": self.regime_metrics,
            "warnings": self.warnings,
            "recommendation_state": str(self.recommendation_state),
            "timestamp": self.timestamp.isoformat(),
        }


# ==============================================================================
# Drift Result Schemas (Sections 11-15, 40)
# ==============================================================================

class DriftStatus(StrEnum):
    NO_DRIFT = "NO_DRIFT"
    MODERATE_DRIFT = "MODERATE_DRIFT"
    SEVERE_DRIFT = "SEVERE_DRIFT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class FeatureDriftEntry:
    """Record of drift assessment for a single feature (Section 12, 40)."""
    feature_name: str
    reference_period: tuple[str, str]
    current_period: tuple[str, str]
    metric_name: str            # e.g., "PSI", "KS", "Wasserstein"
    metric_value: float
    threshold: float
    status: DriftStatus
    missing_value_rate_current: float = 0.0
    missing_value_rate_reference: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "reference_period": list(self.reference_period),
            "current_period": list(self.current_period),
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 6),
            "threshold": self.threshold,
            "status": str(self.status),
            "missing_value_rate_current": round(self.missing_value_rate_current, 4),
            "missing_value_rate_reference": round(self.missing_value_rate_reference, 4),
        }
