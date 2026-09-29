"""Tests for ExperimentTracker and MLflow tracking integration (Phase 4 Section 39)."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from learning.tracking import ExperimentTracker
from models.direction.model import DirectionModel


def test_experiment_tracker_log_run() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracking_uri = f"file:///{Path(tmp_dir).as_posix()}"
        tracker = ExperimentTracker(
            experiment_name="test_experiment",
            tracking_uri=tracking_uri,
        )

        model = DirectionModel(
            algorithm="logistic_regression",
            model_name="dir_test_model",
            model_version="v1.0.0",
            expected_features=["feat1", "feat2"],
            hyperparameters={"C": 0.5},
        )

        metrics = {"accuracy": 0.72, "f1": 0.70, "roc_auc": 0.75}
        training_period = (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 10, tzinfo=UTC))

        run_id = tracker.log_model_run(
            model=model,
            dataset_id="ds_unit_test",
            metrics=metrics,
            training_period=training_period,
        )

        assert isinstance(run_id, str)
        assert len(run_id) > 0
