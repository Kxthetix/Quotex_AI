"""Tests for Controlled Promotion and Rollback (Phase 8 Sections 32, 33, 34, 54, 55)."""

from pathlib import Path
import pytest

from learning.model_governance.models import (
    GateStatus,
    ModelLifecycleState,
    ModelLineageRecord,
    ValidationResult,
)
from learning.model_governance.promotion import ControlledPromotionManager, PromotionError
from learning.model_governance.registry import EnterpriseModelRegistry
from models.direction.model import DirectionModel


@pytest.fixture
def registry_with_prod(tmp_path: Path) -> EnterpriseModelRegistry:
    reg = EnterpriseModelRegistry(tmp_path / "models")
    prod_model = DirectionModel(
        model_name="direction_classifier",
        model_version="prod_v001",
    )
    reg.register_initial_production(prod_model)
    return reg


def test_promotion_blocks_unvalidated_candidate(registry_with_prod: EnterpriseModelRegistry) -> None:
    promoter = ControlledPromotionManager(registry_with_prod)

    # Candidate model registered
    cand_model = DirectionModel(
        model_name="direction_classifier",
        model_version="cand_v002",
    )
    lineage = ModelLineageRecord(
        model_version="cand_v002",
        parent_production_model="prod_v001",
        dataset_version="ds_v001",
        feature_version="v1.0.0",
        label_version="v1.0.0",
        training_period=("", ""),
        validation_period=("", ""),
        test_period=("", ""),
    )
    registry_with_prod.register_candidate(cand_model, lineage)

    # Failed validation result
    failed_val = ValidationResult(
        candidate_model_version="cand_v002",
        final_status=GateStatus.FAIL,
        warnings=["GATE_4_VALIDATION_PERFORMANCE FAILED"],
    )

    # Attempt to promote must raise PromotionError
    with pytest.raises(PromotionError, match="Cannot promote candidate"):
        promoter.promote_candidate(
            model_name="direction_classifier",
            candidate_version="cand_v002",
            validation_result=failed_val,
            approved_by="risk_committee",
            reason="Want higher returns",
        )

    # Production model remains strictly unchanged
    assert registry_with_prod.get_active_production_version() == "prod_v001"


def test_controlled_promotion_success(registry_with_prod: EnterpriseModelRegistry) -> None:
    promoter = ControlledPromotionManager(registry_with_prod)

    # Candidate model registered
    cand_model = DirectionModel(
        model_name="direction_classifier",
        model_version="cand_v002",
    )
    lineage = ModelLineageRecord(
        model_version="cand_v002",
        parent_production_model="prod_v001",
        dataset_version="ds_v001",
        feature_version="v1.0.0",
        label_version="v1.0.0",
        training_period=("", ""),
        validation_period=("", ""),
        test_period=("", ""),
    )
    registry_with_prod.register_candidate(cand_model, lineage)

    # Passing validation
    pass_val = ValidationResult(
        candidate_model_version="cand_v002",
        final_status=GateStatus.PASS,
    )

    record = promoter.promote_candidate(
        model_name="direction_classifier",
        candidate_version="cand_v002",
        validation_result=pass_val,
        approved_by="governance_lead",
        reason="Passed all 10 validation gates with higher out-of-sample accuracy",
    )

    # Assert promotion state
    assert record.new_production_model == "cand_v002"
    assert record.old_production_model == "prod_v001"
    assert registry_with_prod.get_active_production_version() == "cand_v002"

    # Old production model must be RETIRED, not deleted
    old_meta = registry_with_prod.get_metadata("direction_classifier", "prod_v001")
    assert old_meta["status"] == ModelLifecycleState.RETIRED.value

    # New model must be in PRODUCTION
    new_meta = registry_with_prod.get_metadata("direction_classifier", "cand_v002")
    assert new_meta["status"] == ModelLifecycleState.PRODUCTION.value


def test_model_rollback(registry_with_prod: EnterpriseModelRegistry) -> None:
    promoter = ControlledPromotionManager(registry_with_prod)

    # 1. Promote cand_v002
    cand_model = DirectionModel(model_name="direction_classifier", model_version="cand_v002")
    lineage = ModelLineageRecord(
        model_version="cand_v002",
        parent_production_model="prod_v001",
        dataset_version="ds_v001",
        feature_version="v1.0.0",
        label_version="v1.0.0",
        training_period=("", ""),
        validation_period=("", ""),
        test_period=("", ""),
    )
    registry_with_prod.register_candidate(cand_model, lineage)
    pass_val = ValidationResult(candidate_model_version="cand_v002", final_status=GateStatus.PASS)
    promoter.promote_candidate("direction_classifier", "cand_v002", pass_val, "lead", "Approved")

    assert registry_with_prod.get_active_production_version() == "cand_v002"

    # 2. Trigger Rollback back to prod_v001
    rbk = promoter.rollback_model(
        model_name="direction_classifier",
        problematic_version="cand_v002",
        target_version="prod_v001",
        reason="Severe live trading slippage observed",
        initiated_by="risk_officer",
    )

    assert rbk.problematic_model == "cand_v002"
    assert rbk.restored_model == "prod_v001"

    # Active production model is restored to prod_v001
    assert registry_with_prod.get_active_production_version() == "prod_v001"
    prod_meta = registry_with_prod.get_metadata("direction_classifier", "prod_v001")
    assert prod_meta["status"] == ModelLifecycleState.PRODUCTION.value

    # Problematic model is preserved in REJECTED state, NOT deleted
    prob_meta = registry_with_prod.get_metadata("direction_classifier", "cand_v002")
    assert prob_meta["status"] == ModelLifecycleState.REJECTED.value

    # Rollback history is tracked
    assert len(promoter.get_rollback_history()) == 1
