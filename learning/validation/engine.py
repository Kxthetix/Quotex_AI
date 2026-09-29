"""Validation Engine coordinating the 10 Validation Gates for Phase 8.

Executes all 10 gates against candidate model metadata, evaluation metrics,
walk-forward results, and artifact checks. Produces an auditable ValidationResult.
Updates registry status to VALIDATED or REJECTED accordingly.
"""

from __future__ import annotations

from typing import Any

from learning.model_governance.models import (
    GateResult,
    GateStatus,
    ModelLifecycleState,
    ValidationResult,
)
from learning.model_governance.registry import EnterpriseModelRegistry
from learning.validation.gates import (
    ArtifactIntegrityGate,
    CalibrationGate,
    DataQualityGate,
    DriftRobustnessGate,
    FeatureCompatibilityGate,
    RiskConstraintsGate,
    TestPerformanceGate,
    TrainingReproducibilityGate,
    ValidationGate,
    ValidationPerformanceGate,
    WalkForwardStabilityGate,
)
from monitoring.logging import get_logger

logger = get_logger(__name__)


class ValidationEngine:
    """Orchestrates comprehensive validation gate execution for candidate models."""

    def __init__(self, registry: EnterpriseModelRegistry | None = None) -> None:
        self.registry = registry
        self.gates: list[ValidationGate] = [
            DataQualityGate(),
            FeatureCompatibilityGate(),
            TrainingReproducibilityGate(),
            ValidationPerformanceGate(),
            TestPerformanceGate(),
            WalkForwardStabilityGate(),
            CalibrationGate(),
            DriftRobustnessGate(),
            RiskConstraintsGate(),
            ArtifactIntegrityGate(),
        ]

    def validate_candidate(
        self,
        candidate_model_version: str,
        candidate_data: dict[str, Any],
        production_model_version: str = "",
    ) -> ValidationResult:
        """Run all 10 validation gates and compile comprehensive ValidationResult."""
        logger.info("initiating_validation_gates", candidate=candidate_model_version)

        gate_results: list[GateResult] = []
        warnings: list[str] = []

        for gate in self.gates:
            try:
                res = gate.evaluate(candidate_data)
                gate_results.append(res)
                if res.status == GateStatus.FAIL:
                    warnings.append(f"{gate.name} FAILED: {res.details}")
                elif res.status == GateStatus.INSUFFICIENT_DATA:
                    warnings.append(f"{gate.name} INSUFFICIENT DATA: {res.details}")
            except Exception as e:
                logger.error("validation_gate_exception", gate=gate.name, error=str(e))
                res = GateResult(
                    gate_name=gate.name,
                    status=GateStatus.FAIL,
                    metric_name="exception",
                    observed_value=0.0,
                    threshold_value=1.0,
                    details=f"Gate encountered exception: {e}",
                )
                gate_results.append(res)
                warnings.append(f"{gate.name} crashed with exception: {e}")

        # Overall Status
        has_fail = any(g.status == GateStatus.FAIL for g in gate_results)
        has_insufficient = any(g.status == GateStatus.INSUFFICIENT_DATA for g in gate_results)

        if has_fail:
            final_status = GateStatus.FAIL
        elif has_insufficient:
            final_status = GateStatus.INSUFFICIENT_DATA
        else:
            final_status = GateStatus.PASS

        prod_ver = production_model_version or (self.registry.get_active_production_version() if self.registry else "")

        val_result = ValidationResult(
            candidate_model_version=candidate_model_version,
            production_model_version=prod_ver or "none",
            dataset_version=candidate_data.get("dataset_version", "v1.0.0"),
            feature_version=candidate_data.get("feature_version", "v1.0.0"),
            gate_results=gate_results,
            metrics={
                "val_accuracy": candidate_data.get("val_accuracy", 0.0),
                "test_accuracy": candidate_data.get("test_accuracy", 0.0),
                "ece": candidate_data.get("ece", 0.0),
            },
            warnings=warnings,
            final_status=final_status,
        )

        # Update candidate state in registry if available
        if self.registry and candidate_data.get("model_name"):
            m_name = candidate_data["model_name"]
            new_state = ModelLifecycleState.VALIDATED if final_status == GateStatus.PASS else ModelLifecycleState.REJECTED
            self.registry.update_lifecycle_state(
                name=m_name,
                version=candidate_model_version,
                new_state=new_state,
                operator="ValidationEngine",
                reason=f"Validation outcome: {final_status}. Warnings: {len(warnings)}",
            )

        logger.info(
            "validation_gates_completed",
            candidate=candidate_model_version,
            final_status=str(final_status),
            passed_count=sum(1 for g in gate_results if g.status == GateStatus.PASS),
        )
        return val_result
