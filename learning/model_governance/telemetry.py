"""Governance Telemetry and Monitoring Integration for Phase 8 (Section 48).

Integrates the Learning Loop with the monitoring system created in Phase 7.
Exposes:
- current production model
- candidate models
- model status
- last audit
- drift status
- validation status
- last retraining
- promotion history
- rollback history
"""

from __future__ import annotations

from typing import Any

from learning.model_governance.models import (
    ModelAuditResult,
    ModelLifecycleState,
    ValidationResult,
)
from learning.model_governance.promotion import ControlledPromotionManager
from learning.model_governance.registry import EnterpriseModelRegistry


class GovernanceTelemetryProvider:
    """Collects and provides real-time model governance telemetry for operational dashboards."""

    def __init__(
        self,
        registry: EnterpriseModelRegistry,
        promotion_manager: ControlledPromotionManager | None = None,
    ) -> None:
        self.registry = registry
        self.promotion_manager = promotion_manager or ControlledPromotionManager(registry)
        self.last_audit: ModelAuditResult | None = None
        self.last_validation: ValidationResult | None = None
        self.last_retraining_timestamp: str | None = None

    def record_audit(self, audit: ModelAuditResult) -> None:
        self.last_audit = audit

    def record_validation(self, validation: ValidationResult) -> None:
        self.last_validation = validation

    def record_retraining_event(self, timestamp: str) -> None:
        self.last_retraining_timestamp = timestamp

    def get_governance_telemetry(self) -> dict[str, Any]:
        """Compile complete governance telemetry matching Phase 8 Section 48."""
        active_prod = self.registry.get_active_production_version() or "none"

        # Discover candidate models
        all_candidates = self.registry.list_all_models(status=ModelLifecycleState.CANDIDATE)
        validated_candidates = self.registry.list_all_models(status=ModelLifecycleState.VALIDATED)
        candidate_versions = [c["model_version"] for c in (all_candidates + validated_candidates)]

        # Active production model status
        prod_status = "UNKNOWN"
        if active_prod != "none":
            for m in self.registry.list_all_models():
                if m.get("model_version") == active_prod:
                    prod_status = m.get("status", "PRODUCTION")
                    break

        promotions = self.promotion_manager.get_promotion_history()
        rollbacks = self.promotion_manager.get_rollback_history()

        drift_summary = "NO_DRIFT"
        if self.last_audit and self.last_audit.drift_metrics:
            if "concept_drift" in self.last_audit.drift_metrics:
                drift_summary = self.last_audit.drift_metrics["concept_drift"].get("status", "NO_DRIFT")

        return {
            "current_production_model": active_prod,
            "candidate_models": candidate_versions,
            "model_status": prod_status,
            "last_audit": {
                "audit_id": self.last_audit.audit_id if self.last_audit else None,
                "recommendation_state": str(self.last_audit.recommendation_state) if self.last_audit else "HEALTHY",
                "accuracy": self.last_audit.performance_metrics.get("accuracy") if self.last_audit else None,
                "evaluated_samples": self.last_audit.performance_metrics.get("evaluated_count") if self.last_audit else 0,
            } if self.last_audit else None,
            "drift_status": drift_summary,
            "validation_status": {
                "validation_id": self.last_validation.validation_id if self.last_validation else None,
                "final_status": str(self.last_validation.final_status) if self.last_validation else None,
                "candidate_version": self.last_validation.candidate_model_version if self.last_validation else None,
            } if self.last_validation else None,
            "last_retraining": self.last_retraining_timestamp,
            "promotion_history_count": len(promotions),
            "promotion_history": [p.to_dict() for p in promotions[-5:]],
            "rollback_history_count": len(rollbacks),
            "rollback_history": [r.to_dict() for r in rollbacks[-5:]],
        }
