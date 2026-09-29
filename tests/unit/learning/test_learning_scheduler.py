"""Tests for Learning Loop Scheduler and Reports (Phase 8 Sections 39, 40, 41, 42)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import numpy as np
import pytest

from data.ingestion.models import Candle
from learning.model_governance.models import (
    DriftStatus,
    FeatureDriftEntry,
    ModelAuditResult,
)
from learning.model_governance.registry import EnterpriseModelRegistry
from learning.model_governance.reports import (
    DriftReporter,
    ModelAuditReporter,
    ModelComparisonReporter,
)
from learning.scheduler import LearningLoopScheduler
from models.direction.model import DirectionModel
from paper_trading.models import PredictionRecord, PredictionStatus


def generate_candles(count: int = 120) -> list[Candle]:
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    price = 50000.0
    candles = []
    for i in range(count):
        price += (np.sin(i / 10.0) * 80.0) + 5.0
        start = base_time + timedelta(minutes=i)
        end = start + timedelta(minutes=1)
        candles.append(Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open_time=int(start.timestamp() * 1000),
            close_time=int(end.timestamp() * 1000) - 1,
            open=price - 5.0,
            high=price + 10.0,
            low=price - 10.0,
            close=price,
            volume=3.0,
            quote_volume=price * 3.0,
            trades_count=50,
            is_closed=True,
            start_time=start,
            end_time=end,
        ))
    return candles


def test_scheduler_cycle_nominal(tmp_path: Path) -> None:
    registry = EnterpriseModelRegistry(tmp_path / "models")
    prod_model = DirectionModel(model_name="direction_classifier", model_version="prod_v001")
    registry.register_initial_production(prod_model)

    scheduler = LearningLoopScheduler(registry=registry)

    now = datetime.now(UTC)
    preds = [
        PredictionRecord(
            symbol="BTCUSDT",
            timestamp=now - timedelta(minutes=40 - i),
            model_version="prod_v001",
            status=PredictionStatus.EVALUATED,
            actual_outcome="CORRECT" if i < 32 else "INCORRECT",
            confidence=0.85 if i < 32 else 0.20,
            regime="TREND_UP",
        )
        for i in range(40)
    ]

    res = scheduler.run_cycle(predictions=preds)
    assert res.action_taken == "NO_ACTION"
    assert not res.candidate_trained
    assert registry.get_active_production_version() == "prod_v001"


def test_scheduler_cycle_retraining_trigger(tmp_path: Path) -> None:
    registry = EnterpriseModelRegistry(tmp_path / "models")
    prod_model = DirectionModel(model_name="direction_classifier", model_version="prod_v001")
    registry.register_initial_production(prod_model)

    scheduler = LearningLoopScheduler(registry=registry)

    now = datetime.now(UTC)
    # Severely degraded predictions: only 20% accuracy
    preds = [
        PredictionRecord(
            symbol="BTCUSDT",
            timestamp=now - timedelta(minutes=50 - i),
            model_version="prod_v001",
            status=PredictionStatus.EVALUATED,
            actual_outcome="CORRECT" if i < 10 else "INCORRECT",
            confidence=0.6,
            prediction_error=0.03,
            regime="SIDEWAYS",
        )
        for i in range(50)
    ]

    candles = generate_candles(150)
    res = scheduler.run_cycle(
        predictions=preds,
        candidate_candles=candles,
    )

    assert res.action_taken == "RETRAIN_CANDIDATE"
    assert res.candidate_trained
    assert res.validation_result is not None
    # CRITICAL INVARIANT: Production model remains strictly unchanged!
    assert registry.get_active_production_version() == "prod_v001"


def test_reports_generation() -> None:
    # 1. Audit Report
    audit_res = ModelAuditResult(
        model_version="prod_v001",
        prediction_count=100,
        performance_metrics={"accuracy": 0.58, "evaluated_count": 80},
        calibration_metrics={"ece": 0.05, "brier_score": 0.18},
        regime_metrics={"TREND_UP": {"count": 50, "accuracy": 0.62, "mean_error": 0.003}},
        warnings=["Mild calibration degradation"],
    )
    audit_md = ModelAuditReporter.generate_report(audit_res)
    assert "# Model Audit Report: `prod_v001`" in audit_md
    assert "TREND_UP" in audit_md

    # 2. Drift Report
    drift_entry = FeatureDriftEntry(
        feature_name="rsi_14",
        reference_period=("2026-01-01", "2026-06-01"),
        current_period=("2026-06-01", "2026-07-01"),
        metric_name="PSI",
        metric_value=0.08,
        threshold=0.10,
        status=DriftStatus.NO_DRIFT,
    )
    drift_md = DriftReporter.generate_report([drift_entry])
    assert "Comprehensive Drift Detection Report" in drift_md
    assert "rsi_14" in drift_md

    # 3. Model Comparison Report
    comp_md = ModelComparisonReporter.generate_report(
        production_meta={"model_version": "v1.0.0", "algorithm": "random_forest", "dataset_version": "ds1"},
        candidate_meta={"model_version": "v1.1.0-cand", "algorithm": "random_forest", "dataset_version": "ds2"},
        comparison_metrics={"production_accuracy": 0.52, "candidate_accuracy": 0.58},
    )
    assert "Model Comparison: Current Production vs Candidate" in comp_md
    assert "v1.1.0-cand" in comp_md
