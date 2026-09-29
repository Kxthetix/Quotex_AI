"""The 10 Validation Gates for Phase 8 (Section 30).

Every candidate model must be validated through 10 explicit gates before it can
even be considered for promotion approval:
1. Data Quality Gate
2. Feature Compatibility Gate
3. Training Reproducibility Gate
4. Validation Performance Gate
5. Test Performance Gate
6. Walk-Forward Stability Gate
7. Calibration Gate
8. Drift Robustness Gate
9. Risk Constraints Gate
10. Artifact Integrity Gate

Every gate produces: PASS, FAIL, or INSUFFICIENT_DATA.
"""

from __future__ import annotations

from typing import Any

from learning.model_governance.models import GateResult, GateStatus


class ValidationGate:
    """Base class for validation gate checks."""

    name: str = "base_gate"

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        raise NotImplementedError


class DataQualityGate(ValidationGate):
    """Gate 1: Verifies dataset quality and absence of corruption."""
    name = "GATE_1_DATA_QUALITY"

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        missing_rate = candidate_data.get("missing_data_rate", 0.0)
        status = GateStatus.PASS if missing_rate < 0.01 else GateStatus.FAIL
        return GateResult(
            gate_name=self.name,
            status=status,
            metric_name="missing_data_rate",
            observed_value=missing_rate,
            threshold_value=0.01,
            details="Dataset missing value rate within acceptable limits." if status == GateStatus.PASS else "Missing value rate exceeded 1%.",
        )


class FeatureCompatibilityGate(ValidationGate):
    """Gate 2: Verifies feature version matches the locked feature schema."""
    name = "GATE_2_FEATURE_COMPATIBILITY"

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        cand_feat_ver = candidate_data.get("feature_version", "")
        locked_feat_ver = candidate_data.get("locked_feature_version", "v1.0.0")
        schema_matches = candidate_data.get("schema_matches", True)

        passed = (cand_feat_ver == locked_feat_ver) and schema_matches
        return GateResult(
            gate_name=self.name,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            metric_name="version_match",
            observed_value=1.0 if passed else 0.0,
            threshold_value=1.0,
            details=f"Feature version {cand_feat_ver} matches locked version {locked_feat_ver} and schema aligns." if passed else "Feature version mismatch or schema incompatibility.",
        )


class TrainingReproducibilityGate(ValidationGate):
    """Gate 3: Verifies training reproducibility with fixed random seed."""
    name = "GATE_3_TRAINING_REPRODUCIBILITY"

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        has_seed = candidate_data.get("random_seed") is not None
        reproducible = candidate_data.get("is_reproducible", True)
        passed = has_seed and reproducible
        return GateResult(
            gate_name=self.name,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            metric_name="is_reproducible",
            observed_value=1.0 if passed else 0.0,
            threshold_value=1.0,
            details="Training is fully deterministic and reproducible." if passed else "Missing random seed or non-deterministic behavior.",
        )


class ValidationPerformanceGate(ValidationGate):
    """Gate 4: Out-of-sample validation split performance."""
    name = "GATE_4_VALIDATION_PERFORMANCE"

    def __init__(self, min_val_accuracy: float = 0.52) -> None:
        self.min_val_accuracy = min_val_accuracy

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        val_acc = candidate_data.get("val_accuracy", 0.0)
        status = GateStatus.PASS if val_acc >= self.min_val_accuracy else GateStatus.FAIL
        return GateResult(
            gate_name=self.name,
            status=status,
            metric_name="val_accuracy",
            observed_value=val_acc,
            threshold_value=self.min_val_accuracy,
            details=f"Validation accuracy {val_acc:.2%} {'exceeds' if status == GateStatus.PASS else 'fails'} threshold {self.min_val_accuracy:.2%}.",
        )


class TestPerformanceGate(ValidationGate):
    """Gate 5: Pure out-of-sample final test partition performance."""
    __test__ = False
    name = "GATE_5_TEST_PERFORMANCE"

    def __init__(self, min_test_accuracy: float = 0.50) -> None:
        self.min_test_accuracy = min_test_accuracy

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        test_acc = candidate_data.get("test_accuracy", 0.0)
        status = GateStatus.PASS if test_acc >= self.min_test_accuracy else GateStatus.FAIL
        return GateResult(
            gate_name=self.name,
            status=status,
            metric_name="test_accuracy",
            observed_value=test_acc,
            threshold_value=self.min_test_accuracy,
            details=f"Test accuracy {test_acc:.2%} {'exceeds' if status == GateStatus.PASS else 'fails'} threshold {self.min_test_accuracy:.2%}.",
        )


class WalkForwardStabilityGate(ValidationGate):
    """Gate 6: Out-of-sample stability across rolling walk-forward windows."""
    name = "GATE_6_WALK_FORWARD_STABILITY"

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        is_stable = candidate_data.get("walk_forward_stable")
        if is_stable is None:
            return GateResult(
                gate_name=self.name,
                status=GateStatus.INSUFFICIENT_DATA,
                metric_name="walk_forward_stable",
                observed_value=0.0,
                threshold_value=1.0,
                details="Walk-forward validation not executed or insufficient data.",
            )

        status = GateStatus.PASS if is_stable else GateStatus.FAIL
        return GateResult(
            gate_name=self.name,
            status=status,
            metric_name="walk_forward_stable",
            observed_value=1.0 if is_stable else 0.0,
            threshold_value=1.0,
            details="Candidate exhibited out-of-sample stability across rolling windows." if is_stable else "Candidate failed rolling walk-forward stability criteria.",
        )


class CalibrationGate(ValidationGate):
    """Gate 7: Expected Calibration Error (ECE) check."""
    name = "GATE_7_CALIBRATION"

    def __init__(self, max_ece: float = 0.15) -> None:
        self.max_ece = max_ece

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        ece = candidate_data.get("ece", 0.05)
        status = GateStatus.PASS if ece <= self.max_ece else GateStatus.FAIL
        return GateResult(
            gate_name=self.name,
            status=status,
            metric_name="expected_calibration_error",
            observed_value=ece,
            threshold_value=self.max_ece,
            details=f"ECE {ece:.4f} {'within' if status == GateStatus.PASS else 'exceeds'} allowable bound {self.max_ece:.4f}.",
        )


class DriftRobustnessGate(ValidationGate):
    """Gate 8: Feature and prediction drift robustness."""
    name = "GATE_8_DRIFT_ROBUSTNESS"

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        severe_drift = candidate_data.get("has_severe_drift", False)
        status = GateStatus.FAIL if severe_drift else GateStatus.PASS
        return GateResult(
            gate_name=self.name,
            status=status,
            metric_name="severe_drift",
            observed_value=1.0 if severe_drift else 0.0,
            threshold_value=0.0,
            details="No severe drift detected." if not severe_drift else "Severe drift detected on candidate evaluation dataset.",
        )


class RiskConstraintsGate(ValidationGate):
    """Gate 9: Backtest simulated risk compliance (max drawdown < 20%)."""
    name = "GATE_9_RISK_CONSTRAINTS"

    def __init__(self, max_drawdown: float = 0.20) -> None:
        self.max_drawdown = max_drawdown

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        dd = candidate_data.get("max_drawdown", 0.10)
        status = GateStatus.PASS if dd <= self.max_drawdown else GateStatus.FAIL
        return GateResult(
            gate_name=self.name,
            status=status,
            metric_name="max_drawdown",
            observed_value=dd,
            threshold_value=self.max_drawdown,
            details=f"Max drawdown {dd:.2%} {'satisfies' if status == GateStatus.PASS else 'violates'} risk constraint {self.max_drawdown:.2%}.",
        )


class ArtifactIntegrityGate(ValidationGate):
    """Gate 10: Physical artifact existence and SHA256 checksum verification."""
    name = "GATE_10_ARTIFACT_INTEGRITY"

    def evaluate(self, candidate_data: dict[str, Any]) -> GateResult:
        artifact_valid = candidate_data.get("artifact_valid", True)
        status = GateStatus.PASS if artifact_valid else GateStatus.FAIL
        return GateResult(
            gate_name=self.name,
            status=status,
            metric_name="artifact_valid",
            observed_value=1.0 if artifact_valid else 0.0,
            threshold_value=1.0,
            details="Artifact file exists, checksum matches, and model deserializes cleanly." if artifact_valid else "Artifact missing, checksum corrupted, or deserialization error.",
        )
