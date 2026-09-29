"""Tests for Validation Gates and ValidationEngine (Phase 8 Sections 30, 31)."""

import pytest

from learning.model_governance.models import GateStatus
from learning.validation.engine import ValidationEngine
from learning.validation.gates import (
    ArtifactIntegrityGate,
    CalibrationGate,
    DataQualityGate,
    DriftRobustnessGate,
    FeatureCompatibilityGate,
    RiskConstraintsGate,
    TestPerformanceGate,
    TrainingReproducibilityGate,
    ValidationPerformanceGate,
    WalkForwardStabilityGate,
)


def test_individual_gates() -> None:
    # 1. Data Quality
    dq = DataQualityGate()
    assert dq.evaluate({"missing_data_rate": 0.005}).status == GateStatus.PASS
    assert dq.evaluate({"missing_data_rate": 0.05}).status == GateStatus.FAIL

    # 2. Feature Compatibility
    fc = FeatureCompatibilityGate()
    assert fc.evaluate({"feature_version": "v1.0.0", "locked_feature_version": "v1.0.0", "schema_matches": True}).status == GateStatus.PASS
    assert fc.evaluate({"feature_version": "v2.0.0", "locked_feature_version": "v1.0.0", "schema_matches": True}).status == GateStatus.FAIL

    # 3. Training Reproducibility
    tr = TrainingReproducibilityGate()
    assert tr.evaluate({"random_seed": 42, "is_reproducible": True}).status == GateStatus.PASS
    assert tr.evaluate({"random_seed": None}).status == GateStatus.FAIL

    # 4. Validation Performance
    vp = ValidationPerformanceGate(min_val_accuracy=0.52)
    assert vp.evaluate({"val_accuracy": 0.55}).status == GateStatus.PASS
    assert vp.evaluate({"val_accuracy": 0.50}).status == GateStatus.FAIL

    # 5. Test Performance
    tp = TestPerformanceGate(min_test_accuracy=0.50)
    assert tp.evaluate({"test_accuracy": 0.53}).status == GateStatus.PASS
    assert tp.evaluate({"test_accuracy": 0.47}).status == GateStatus.FAIL

    # 6. Walk-Forward Stability
    wf = WalkForwardStabilityGate()
    assert wf.evaluate({"walk_forward_stable": True}).status == GateStatus.PASS
    assert wf.evaluate({"walk_forward_stable": False}).status == GateStatus.FAIL
    assert wf.evaluate({}).status == GateStatus.INSUFFICIENT_DATA

    # 7. Calibration
    cb = CalibrationGate(max_ece=0.15)
    assert cb.evaluate({"ece": 0.08}).status == GateStatus.PASS
    assert cb.evaluate({"ece": 0.22}).status == GateStatus.FAIL

    # 8. Drift Robustness
    dr = DriftRobustnessGate()
    assert dr.evaluate({"has_severe_drift": False}).status == GateStatus.PASS
    assert dr.evaluate({"has_severe_drift": True}).status == GateStatus.FAIL

    # 9. Risk Constraints
    rc = RiskConstraintsGate(max_drawdown=0.20)
    assert rc.evaluate({"max_drawdown": 0.12}).status == GateStatus.PASS
    assert rc.evaluate({"max_drawdown": 0.25}).status == GateStatus.FAIL

    # 10. Artifact Integrity
    ai = ArtifactIntegrityGate()
    assert ai.evaluate({"artifact_valid": True}).status == GateStatus.PASS
    assert ai.evaluate({"artifact_valid": False}).status == GateStatus.FAIL


def test_validation_engine_all_pass() -> None:
    engine = ValidationEngine()
    candidate_data = {
        "missing_data_rate": 0.0,
        "feature_version": "v1.0.0",
        "locked_feature_version": "v1.0.0",
        "schema_matches": True,
        "random_seed": 42,
        "is_reproducible": True,
        "val_accuracy": 0.56,
        "test_accuracy": 0.54,
        "walk_forward_stable": True,
        "ece": 0.06,
        "has_severe_drift": False,
        "max_drawdown": 0.11,
        "artifact_valid": True,
    }

    result = engine.validate_candidate("cand_v001", candidate_data)
    assert result.final_status == GateStatus.PASS
    assert len(result.gate_results) == 10
    assert len(result.warnings) == 0


def test_validation_engine_one_gate_fails() -> None:
    engine = ValidationEngine()
    candidate_data = {
        "missing_data_rate": 0.0,
        "feature_version": "v1.0.0",
        "locked_feature_version": "v1.0.0",
        "schema_matches": True,
        "random_seed": 42,
        "is_reproducible": True,
        "val_accuracy": 0.56,
        "test_accuracy": 0.54,
        "walk_forward_stable": True,
        "ece": 0.06,
        "has_severe_drift": True,  # FAILS Gate 8
        "max_drawdown": 0.11,
        "artifact_valid": True,
    }

    result = engine.validate_candidate("cand_v001", candidate_data)
    assert result.final_status == GateStatus.FAIL
    assert any("GATE_8_DRIFT_ROBUSTNESS FAILED" in w for w in result.warnings)
