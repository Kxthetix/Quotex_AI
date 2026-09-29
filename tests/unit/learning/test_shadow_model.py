"""Tests for Shadow Model Inference (Phase 8 Sections 28, 29)."""

from datetime import UTC, datetime
import pandas as pd
import pytest

from learning.model_governance.shadow import ShadowModelManager
from models.direction.model import DirectionModel
from paper_trading.models import PredictionRecord


def test_shadow_model_inference_and_ledger() -> None:
    shadow_mgr = ShadowModelManager()

    # Candidate model (fitted)
    cand_model = DirectionModel(
        model_name="direction_classifier",
        model_version="cand_v002",
        expected_features=["feat_1", "feat_2"],
    )
    import numpy as np
    cand_model.fit(
        pd.DataFrame([{"feat_1": 1.0, "feat_2": 2.0}, {"feat_1": -1.0, "feat_2": -2.0}]),
        np.array([1, 0]),
    )
    shadow_mgr.register_shadow_candidate(cand_model)

    now = datetime.now(UTC)
    prod_pred = PredictionRecord(
        symbol="BTCUSDT",
        timestamp=now,
        model_version="prod_v001",
        feature_version="v1.0.0",
        signal_action="BUY",
        confidence=0.72,
        expected_return=0.015,
    )

    features_df = pd.DataFrame([{"feat_1": 1.5, "feat_2": -0.8}])

    # Evaluate shadow
    shadow_records = shadow_mgr.evaluate_shadow(
        symbol="BTCUSDT",
        features_df=features_df,
        production_prediction=prod_pred,
        timestamp=now,
    )

    assert len(shadow_records) == 1
    rec = shadow_records[0]
    assert rec.production_model_version == "prod_v001"
    assert rec.candidate_model_version == "cand_v002"
    assert rec.production_action == "BUY"

    # Ground truth outcome arrival
    shadow_mgr.append_outcome(
        symbol="BTCUSDT",
        timestamp=now,
        actual_outcome="CORRECT",
        realized_return=0.012,
    )

    summary = shadow_mgr.get_comparison_summary()
    assert summary["evaluated_observations"] == 1
    assert "production_accuracy" in summary
    assert "candidate_accuracy" in summary
