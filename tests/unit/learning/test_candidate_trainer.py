"""Tests for Candidate Dataset Builder and Candidate Trainer (Phase 8 Sections 19, 20, 22, 38)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from data.ingestion.models import Candle
from learning.model_governance.models import ModelLifecycleState
from learning.model_governance.registry import EnterpriseModelRegistry
from learning.retraining.dataset_builder import CandidateDatasetBuilder, DatasetBuildConfig
from learning.retraining.trainer import CandidateTrainConfig, CandidateTrainer
from models.direction.model import DirectionModel


def generate_synthetic_candles(count: int = 150) -> list[Candle]:
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
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


def test_dataset_builder_chronological_splits(tmp_path: Path) -> None:
    builder = CandidateDatasetBuilder()
    candles = generate_synthetic_candles(150)

    cfg = DatasetBuildConfig(
        dataset_version="ds_test_001",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    splits = builder.build_dataset(candles, cfg)

    assert len(splits.train_df) > 0
    assert len(splits.val_df) > 0
    assert len(splits.test_df) > 0

    train_end = splits.train_df["end_time"].max()
    val_start = splits.val_df["end_time"].min()
    val_end = splits.val_df["end_time"].max()
    test_start = splits.test_df["end_time"].min()

    # Temporal boundary assertion: train < val < test
    assert train_end < val_start
    assert val_end < test_start


def test_candidate_training_workflow(tmp_path: Path) -> None:
    registry = EnterpriseModelRegistry(tmp_path / "models")
    # Register prod
    prod_model = DirectionModel(
        model_name="direction_classifier",
        model_version="prod_v001",
    )
    registry.register_initial_production(prod_model)

    candles = generate_synthetic_candles(150)
    builder = CandidateDatasetBuilder()
    splits = builder.build_dataset(candles, DatasetBuildConfig(dataset_version="ds_v001"))

    trainer = CandidateTrainer(registry=registry)
    train_cfg = CandidateTrainConfig(
        model_name="direction_classifier",
        candidate_version="cand_v002",
        dataset_version="ds_v001",
        random_seed=42,
        hyperparameters={"n_estimators": 20, "max_depth": 3},
    )

    candidate = trainer.train_candidate(splits=splits, config=train_cfg)

    assert candidate.model_version == "cand_v002"
    # Registry checks
    assert registry.get_active_production_version() == "prod_v001"
    meta = registry.get_metadata("direction_classifier", "cand_v002")
    assert meta["status"] == ModelLifecycleState.CANDIDATE.value
    assert meta["parent_model_version"] == "prod_v001"
    assert "val_accuracy" in meta["metrics"]
    assert "test_accuracy" in meta["metrics"]


def test_reproducibility(tmp_path: Path) -> None:
    registry = EnterpriseModelRegistry(tmp_path / "models")
    candles = generate_synthetic_candles(150)
    builder = CandidateDatasetBuilder()
    splits = builder.build_dataset(candles, DatasetBuildConfig(dataset_version="ds_v001"))

    trainer = CandidateTrainer(registry=registry)

    # Train model 1
    cfg1 = CandidateTrainConfig(
        candidate_version="cand_run_1",
        random_seed=42,
        hyperparameters={"n_estimators": 20, "max_depth": 3},
    )
    model1 = trainer.train_candidate(splits, cfg1)

    # Train model 2 with identical seed
    cfg2 = CandidateTrainConfig(
        candidate_version="cand_run_2",
        random_seed=42,
        hyperparameters={"n_estimators": 20, "max_depth": 3},
    )
    model2 = trainer.train_candidate(splits, cfg2)

    meta1 = registry.get_metadata("direction_classifier", "cand_run_1")
    meta2 = registry.get_metadata("direction_classifier", "cand_run_2")

    # Metrics should be deterministic
    assert meta1["metrics"]["val_accuracy"] == meta2["metrics"]["val_accuracy"]
    assert meta1["metrics"]["test_accuracy"] == meta2["metrics"]["test_accuracy"]
