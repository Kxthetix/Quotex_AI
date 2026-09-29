"""Retraining Decision Engine for Phase 8 (Section 18).

Evaluates audit results, drift metrics, and performance indicators to determine
whether retraining a candidate model should be considered:
- NO_ACTION: Current production model is performing satisfactorily.
- MONITOR: Mild drift or transient dip observed; continue observation.
- REVIEW: Sustained performance drop or notable drift requires human review.
- RETRAIN_CANDIDATE: Clear justification exists to train a candidate model.

CRITICAL INVARIANT:
The decision to retrain only initiates candidate model training.
It NEVER automatically deploys or promotes any model.
"""

from __future__ import annotations

from typing import Any

from learning.model_governance.models import (
    AuditRecommendation,
    ModelAuditResult,
)
from monitoring.logging import get_logger

logger = get_logger(__name__)


class RetrainingDecisionEngine:
    """Evaluates whether retraining a candidate model is justified."""

    def __init__(
        self,
        min_consecutive_degraded_audits: int = 2,
    ) -> None:
        self.min_consecutive_degraded_audits = min_consecutive_degraded_audits
        self._audit_history: list[ModelAuditResult] = []

    def evaluate(self, latest_audit: ModelAuditResult) -> dict[str, Any]:
        """Evaluate latest audit result alongside historical audit trends."""
        self._audit_history.append(latest_audit)

        # Insufficient data
        if latest_audit.recommendation_state == AuditRecommendation.INSUFFICIENT_DATA:
            return {
                "decision": "NO_ACTION",
                "reason": "Insufficient sample size to evaluate retraining eligibility.",
                "justification": latest_audit.warnings,
            }

        # Check consecutive degradation
        recent_audits = self._audit_history[-self.min_consecutive_degraded_audits:]
        all_degraded = all(
            a.recommendation_state in (AuditRecommendation.REVIEW, AuditRecommendation.RETRAIN_CANDIDATE)
            for a in recent_audits
        ) and len(recent_audits) >= self.min_consecutive_degraded_audits

        if latest_audit.recommendation_state == AuditRecommendation.RETRAIN_CANDIDATE or all_degraded:
            decision = "RETRAIN_CANDIDATE"
            reason = "Sustained performance degradation and/or concept drift detected."
        elif latest_audit.recommendation_state == AuditRecommendation.REVIEW:
            decision = "REVIEW"
            reason = "Performance dip or moderate drift observed; requires review."
        elif latest_audit.recommendation_state == AuditRecommendation.MONITOR:
            decision = "MONITOR"
            reason = "Minor drift indicators; continue monitoring."
        else:
            decision = "NO_ACTION"
            reason = "Production model performance is nominal."

        logger.info("retraining_decision_evaluated", decision=decision, reason=reason)
        return {
            "decision": decision,
            "reason": reason,
            "model_version": latest_audit.model_version,
            "audit_id": latest_audit.audit_id,
            "accuracy": latest_audit.performance_metrics.get("accuracy"),
            "warnings": latest_audit.warnings,
        }
