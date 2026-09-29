"""Critical Invariant and Safety Tests for Phase 8 (Sections 51, 52, 53, 54, 55).

Contains the 5 explicitly mandated critical safety invariant tests:
1. Section 51: Critical Production Protection Test
2. Section 52: Critical Leakage Test
3. Section 53: Critical Drift Test
4. Section 54: Critical Validation Test
5. Section 55: Critical Rollback Test
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from data.ingestion.models import Candle
from learning.model_governance.auditor import AuditorConfig, ModelAuditor
from learning.model_governance.drift import FeatureDriftDetector
from learning.model_governance.models import (
    AuditRecommendation,
    DriftStatus,
    GateStatus,
    ModelLifecycleState,
    ModelLineageRecord,
    ValidationResult,
)
from learning.model_governance.promotion import ControlledPromotionManager, PromotionError
from learning.model_governance.registry import EnterpriseModelRegistry
from learning.retraining.dataset_builder import CandidateDatasetBuilder, DatasetBuildConfig
from learning.retraining.decision import RetrainingDecisionEngine
from learning.retraining.trainer import CandidateTrainConfig, CandidateTrainer
from learning.validation.engine import ValidationEngine
from models.direction.model import DirectionModel
from paper_trading.models import PredictionRecord, PredictionStatus


def generate_candles(count: int = 200, start_time: datetime | None = None) -> list[Candle]:
    base_time = start_time or datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    price = 50000.0
    candles = []
    for i in range(count):
        price += (np.sin(i / 10.0) * 100.0) + 10.0
        start = base_time + timedelta(minutes=i)
        end = start + timedelta(minutes=1)
        candles.append(Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open_time=int(start.timestamp() * 1000),
            close_time=int(end.timestamp() * 1000) - 1,
            open=price - 10.0,
            high=price + 20.0,
            low=price - 20.0,
            close=price,
            volume=5.0,
            quote_volume=price * 5.0,
            trades_count=100,
            is_closed=True,
            start_time=start,
            end_time=end,
        ))
    return candles


# ==============================================================================
# 1. Critical Production Protection Test (Section 51)
# ==============================================================================
def test_critical_production_protection(tmp_path: Path) -> None:
    """CRITICAL TEST 51:
    1. Current production model = MODEL_A.
    2. Train MODEL_B.
    3. MODEL_B passes candidate training.
    4. MODEL_B is registered.
    5. Even if MODEL_B performs better in an experiment:
       production model must remain MODEL_A.
    6. Only an explicit approved promotion can change production.
    """
    registry = EnterpriseModelRegistry(tmp_path / "models")

    # 1. Current production model = MODEL_A
    model_a = DirectionModel(model_name="direction_classifier", model_version="MODEL_A")
    registry.register_initial_production(model_a)
    assert registry.get_active_production_version() == "MODEL_A"

    # 2. Train MODEL_B
    candles = generate_candles(150)
    builder = CandidateDatasetBuilder()
    splits = builder.build_dataset(candles, DatasetBuildConfig(dataset_version="ds_v001"))

    trainer = CandidateTrainer(registry=registry)
    # 3 & 4. Train and register MODEL_B
    model_b = trainer.train_candidate(
        splits=splits,
        config=CandidateTrainConfig(candidate_version="MODEL_B", dataset_version="ds_v001"),
        parent_production_model="MODEL_A",
    )

    # 5. Even if MODEL_B performs better, production model MUST remain MODEL_A!
    assert registry.get_active_production_version() == "MODEL_A"
    meta_b = registry.get_metadata("direction_classifier", "MODEL_B")
    assert meta_b["status"] == ModelLifecycleState.CANDIDATE.value

    # 6. Only an explicit approved promotion can change production
    promoter = ControlledPromotionManager(registry)
    val_result = ValidationResult(candidate_model_version="MODEL_B", final_status=GateStatus.PASS)
    promoter.promote_candidate(
        model_name="direction_classifier",
        candidate_version="MODEL_B",
        validation_result=val_result,
        approved_by="human_governance_board",
        reason="Approved after passing all required evaluation criteria.",
    )

    # Now and ONLY now does production become MODEL_B
    assert registry.get_active_production_version() == "MODEL_B"


# ==============================================================================
# 2. Critical Leakage Test (Section 52)
# ==============================================================================
def test_critical_leakage_protection() -> None:
    """CRITICAL TEST 52:
    Create TRAIN cutoff = T.
    Then add observations after T.
    Rebuild the training dataset.
    Verify: No observation after T appears in the training dataset.
    Repeat for features, labels, outcomes, preprocessing statistics.
    """
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    cutoff_t = base_time + timedelta(minutes=100)

    # Generate 150 candles (50 candles are after cutoff T)
    candles = generate_candles(count=150, start_time=base_time)

    builder = CandidateDatasetBuilder()
    cfg = DatasetBuildConfig(
        dataset_version="ds_cutoff_test",
        test_end=cutoff_t,  # Cutoff at T
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    splits = builder.build_dataset(candles, cfg)

    # Assert that no row in train, val, or test has timestamp > cutoff_t
    max_train_time = splits.train_df["end_time"].max()
    max_val_time = splits.val_df["end_time"].max()
    max_test_time = splits.test_df["end_time"].max()

    assert max_train_time < cutoff_t
    assert max_val_time < cutoff_t
    assert max_test_time <= cutoff_t

    # Features and labels in train_df must only contain rows <= cutoff_t
    assert len(splits.train_df[splits.train_df["end_time"] > cutoff_t]) == 0


# ==============================================================================
# 3. Critical Drift Test (Section 53)
# ==============================================================================
def test_critical_drift_handling(tmp_path: Path) -> None:
    """CRITICAL TEST 53:
    Create a synthetic distribution shift.
    Verify:
    - drift detector identifies the change
    - drift status is recorded
    - model is NOT automatically replaced
    - audit is generated
    - appropriate review state is created
    """
    registry = EnterpriseModelRegistry(tmp_path / "models")
    prod_model = DirectionModel(model_name="direction_classifier", model_version="MODEL_PROD")
    registry.register_initial_production(prod_model)

    detector = FeatureDriftDetector(psi_severe_threshold=0.25)
    np.random.seed(42)
    ref_df = pd.DataFrame({"feat_vol": np.random.normal(10.0, 1.0, 200)})
    # Synthetic severe distribution shift
    cur_df = pd.DataFrame({"feat_vol": np.random.normal(25.0, 5.0, 200)})

    drift_entries = detector.evaluate_dataset(ref_df, cur_df)
    assert len(drift_entries) == 1
    assert drift_entries[0].status == DriftStatus.SEVERE_DRIFT

    auditor = ModelAuditor(feature_drift_detector=detector)
    now = datetime.now(UTC)
    preds = [
        PredictionRecord(
            symbol="BTCUSDT",
            timestamp=now - timedelta(minutes=40 - i),
            model_version="MODEL_PROD",
            status=PredictionStatus.EVALUATED,
            actual_outcome="CORRECT",
            confidence=0.7,
        )
        for i in range(40)
    ]

    audit_res = auditor.audit_model(
        model_version="MODEL_PROD",
        predictions=preds,
        reference_features_df=ref_df,
        current_features_df=cur_df,
    )

    # Review/warning state created
    assert audit_res.recommendation_state in (AuditRecommendation.MONITOR, AuditRecommendation.REVIEW)
    assert any("Severe feature drift" in w for w in audit_res.warnings)

    # CRITICAL INVARIANT: Model is NOT automatically replaced
    assert registry.get_active_production_version() == "MODEL_PROD"


# ==============================================================================
# 4. Critical Validation Test (Section 54)
# ==============================================================================
def test_critical_validation_failure(tmp_path: Path) -> None:
    """CRITICAL TEST 54:
    Create a candidate that fails one required gate.
    Verify:
    - candidate status != PRODUCTION
    - promotion is rejected
    """
    registry = EnterpriseModelRegistry(tmp_path / "models")
    prod_model = DirectionModel(model_name="direction_classifier", model_version="MODEL_A")
    registry.register_initial_production(prod_model)

    cand_model = DirectionModel(model_name="direction_classifier", model_version="MODEL_CAND_FAIL")
    lineage = ModelLineageRecord(
        model_version="MODEL_CAND_FAIL",
        parent_production_model="MODEL_A",
        dataset_version="ds_v001",
        feature_version="v1.0.0",
        label_version="v1.0.0",
        training_period=("", ""),
        validation_period=("", ""),
        test_period=("", ""),
    )
    registry.register_candidate(cand_model, lineage)

    # Candidate fails Risk Constraints Gate (e.g. max drawdown 35% > 20% limit)
    validation_engine = ValidationEngine(registry=registry)
    val_data = {
        "model_name": "direction_classifier",
        "feature_version": "v1.0.0",
        "val_accuracy": 0.58,
        "test_accuracy": 0.55,
        "max_drawdown": 0.35,  # FAILS Risk Constraints Gate
        "artifact_valid": True,
    }

    val_res = validation_engine.validate_candidate("MODEL_CAND_FAIL", val_data, "MODEL_A")
    assert val_res.final_status == GateStatus.FAIL

    # Candidate status in registry must NOT be PRODUCTION
    cand_meta = registry.get_metadata("direction_classifier", "MODEL_CAND_FAIL")
    assert cand_meta["status"] != ModelLifecycleState.PRODUCTION.value
    assert cand_meta["status"] == ModelLifecycleState.REJECTED.value

    # Attempting to promote must be rejected with PromotionError
    promoter = ControlledPromotionManager(registry)
    with pytest.raises(PromotionError, match="Cannot promote candidate"):
        promoter.promote_candidate(
            model_name="direction_classifier",
            candidate_version="MODEL_CAND_FAIL",
            validation_result=val_res,
            approved_by="operator",
            reason="Override attempt",
        )

    # Production remains unchanged
    assert registry.get_active_production_version() == "MODEL_A"


# ==============================================================================
# 5. Critical Rollback Test (Section 55)
# ==============================================================================
def test_critical_rollback(tmp_path: Path) -> None:
    """CRITICAL TEST 55:
    Set MODEL_A = production.
    Promote MODEL_B.
    Then trigger rollback.
    Verify:
    - MODEL_A becomes active production again.
    - MODEL_B remains in registry.
    - Do not delete model history.
    """
    registry = EnterpriseModelRegistry(tmp_path / "models")
    promoter = ControlledPromotionManager(registry)

    # 1. Set MODEL_A = production
    model_a = DirectionModel(model_name="direction_classifier", model_version="MODEL_A")
    registry.register_initial_production(model_a)
    assert registry.get_active_production_version() == "MODEL_A"

    # 2. Promote MODEL_B
    model_b = DirectionModel(model_name="direction_classifier", model_version="MODEL_B")
    lineage = ModelLineageRecord(
        model_version="MODEL_B",
        parent_production_model="MODEL_A",
        dataset_version="ds_v001",
        feature_version="v1.0.0",
        label_version="v1.0.0",
        training_period=("", ""),
        validation_period=("", ""),
        test_period=("", ""),
    )
    registry.register_candidate(model_b, lineage)
    pass_val = ValidationResult(candidate_model_version="MODEL_B", final_status=GateStatus.PASS)
    promoter.promote_candidate("direction_classifier", "MODEL_B", pass_val, "lead", "Initial promotion")

    assert registry.get_active_production_version() == "MODEL_B"

    # 3. Trigger Rollback
    rollback_record = promoter.rollback_model(
        model_name="direction_classifier",
        problematic_version="MODEL_B",
        target_version="MODEL_A",
        reason="Realized return anomaly in live execution",
        initiated_by="risk_officer",
    )

    # 4. Verify MODEL_A becomes active production again
    assert registry.get_active_production_version() == "MODEL_A"
    meta_a = registry.get_metadata("direction_classifier", "MODEL_A")
    assert meta_a["status"] == ModelLifecycleState.PRODUCTION.value

    # 5. Verify MODEL_B remains in registry and is NOT deleted
    meta_b = registry.get_metadata("direction_classifier", "MODEL_B")
    assert meta_b["status"] == ModelLifecycleState.REJECTED.value
    assert Path(meta_b["artifact_location"]).exists()

    # 6. Verify audit history is preserved
    assert len(promoter.get_promotion_history()) == 1
    assert len(promoter.get_rollback_history()) == 1
    assert rollback_record.problematic_model == "MODEL_B"
    assert rollback_record.restored_model == "MODEL_A"
