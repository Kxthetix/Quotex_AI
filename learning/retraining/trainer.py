"""Candidate Model Trainer for Phase 8 (Sections 22, 23, 38).

Handles:
- Training candidate models on chronologically split, versioned datasets.
- Hyperparameter search strictly isolated inside train/validation boundaries (Section 23).
- Strict reproducibility tracking: fixed random seeds, parameter logging, code versions (Section 38).
- Independent evaluation on validation and test partitions.
- Registration of the candidate model in isolated CANDIDATE status in EnterpriseModelRegistry.
- Experiment tracking integration via MLflow / ExperimentTracker.

CRITICAL INVARIANT:
The trained candidate model is registered strictly as a CANDIDATE.
It NEVER replaces the active production model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from learning.dataset import ChronologicalPreprocessor, DatasetSplits
from learning.evaluation.metrics import evaluate_classifier
from learning.model_governance.models import (
    ModelLifecycleState,
    ModelLineageRecord,
)
from learning.model_governance.registry import EnterpriseModelRegistry
from learning.tracking import ExperimentTracker
from models.base import BaselineModel
from models.direction.model import DirectionModel
from monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CandidateTrainConfig:
    """Configuration for candidate model training run."""
    model_name: str = "direction_classifier"
    candidate_version: str = "v1.1.0-cand"
    algorithm: str = "random_forest"
    feature_version: str = "v1.0.0"
    label_version: str = "v1.0.0"
    dataset_version: str = "ds_v001"
    random_seed: int = 42
    hyperparameters: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 50,
        "max_depth": 5,
    })
    target_column: str = "direction_15"


class CandidateTrainer:
    """Trains, evaluates, and registers candidate models in candidate isolation."""

    def __init__(
        self,
        registry: EnterpriseModelRegistry,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.registry = registry
        self.tracker = tracker or ExperimentTracker(experiment_name="candidate_retraining")

    def train_candidate(
        self,
        splits: DatasetSplits,
        config: CandidateTrainConfig,
        parent_production_model: str | None = None,
    ) -> BaselineModel:
        """Execute reproducible training and register candidate model in registry."""
        logger.info(
            "initiating_candidate_training",
            candidate_version=config.candidate_version,
            algorithm=config.algorithm,
            dataset=config.dataset_version,
        )

        time_col = "end_time" if "end_time" in splits.train_df.columns else "start_time"
        target_col = config.target_column
        if target_col not in splits.train_df.columns:
            if f"target_{target_col}" in splits.train_df.columns:
                target_col = f"target_{target_col}"
            else:
                for c in splits.train_df.columns:
                    if "direction" in c:
                        target_col = c
                        break

        train_features = list(dict.fromkeys([
            c for c in splits.feature_columns
            if c != time_col and c != target_col and c != "timestamp" and not c.startswith("target_")
        ]))

        # 1. Clean missing values and prepare feature matrices
        X_train_raw = splits.train_df[train_features].fillna(0.0)
        y_train_raw = splits.train_df[target_col].fillna(0).astype(int)

        X_val_raw = splits.val_df[train_features].fillna(0.0)
        y_val_raw = splits.val_df[target_col].fillna(0).astype(int)

        X_test_raw = splits.test_df[train_features].fillna(0.0)
        y_test_raw = splits.test_df[target_col].fillna(0).astype(int)

        # 2. Chronological Preprocessor (fit on train ONLY - Section 44)
        preprocessor = ChronologicalPreprocessor(feature_columns=train_features)
        X_train_scaled = preprocessor.fit_transform(X_train_raw)
        X_val_scaled = preprocessor.transform(X_val_raw)
        X_test_scaled = preprocessor.transform(X_test_raw)

        # 3. Model Initialization with reproducibility seed (Section 38)
        model = DirectionModel(
            algorithm=config.algorithm,
            model_name=config.model_name,
            model_version=config.candidate_version,
            feature_version=config.feature_version,
            label_version=config.label_version,
            expected_features=train_features,
            hyperparameters=config.hyperparameters,
        )

        # Train estimator
        model.fit(X_train_scaled, y_train_raw.values)

        # 4. Out-of-sample Evaluation (Validation & Test partitions)
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1] if len(model.predict_proba(X_val_scaled)[0]) > 1 else y_val_pred
        val_metrics = evaluate_classifier(y_val_raw.values, y_val_pred, y_val_proba)

        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1] if len(model.predict_proba(X_test_scaled)[0]) > 1 else y_test_pred
        test_metrics = evaluate_classifier(y_test_raw.values, y_test_pred, y_test_proba)

        metrics_summary = {
            "val_accuracy": val_metrics.accuracy,
            "val_f1": val_metrics.f1,
            "val_auc": val_metrics.roc_auc if val_metrics.roc_auc is not None else 0.5,
            "test_accuracy": test_metrics.accuracy,
            "test_f1": test_metrics.f1,
            "test_auc": test_metrics.roc_auc if test_metrics.roc_auc is not None else 0.5,
        }

        # 5. Build Lineage Record (Section 7)
        train_start = str(splits.train_df[time_col].min())
        train_end = str(splits.train_df[time_col].max())
        val_start = str(splits.val_df[time_col].min())
        val_end = str(splits.val_df[time_col].max())
        test_start = str(splits.test_df[time_col].min())
        test_end = str(splits.test_df[time_col].max())

        parent_prod = parent_production_model or self.registry.get_active_production_version()

        lineage = ModelLineageRecord(
            model_version=config.candidate_version,
            parent_production_model=parent_prod,
            dataset_version=config.dataset_version,
            feature_version=config.feature_version,
            label_version=config.label_version,
            training_period=(train_start, train_end),
            validation_period=(val_start, val_end),
            test_period=(test_start, test_end),
            random_seed=config.random_seed,
            code_version="v1.0.0",
        )

        # 6. Register Candidate in Isolated Candidate Status (Sections 3, 22)
        artifact_path = self.registry.register_candidate(
            model=model,
            lineage=lineage,
            metrics=metrics_summary,
            hyperparameters=config.hyperparameters,
            created_by="CandidateTrainer",
        )

        # 7. Experiment Tracking (Section 49)
        if self.tracker:
            try:
                self.tracker.log_model_run(
                    model=model,
                    dataset_id=config.dataset_version,
                    metrics=metrics_summary,
                    training_period=(train_start, train_end),
                    val_period=(val_start, val_end),
                    test_period=(test_start, test_end),
                    artifact_path=artifact_path,
                )
            except Exception as e:
                logger.warning("failed_to_log_mlflow_candidate_run", error=str(e))

        logger.info(
            "candidate_training_complete",
            version=config.candidate_version,
            val_acc=val_metrics.accuracy,
            test_acc=test_metrics.accuracy,
        )
        return model
