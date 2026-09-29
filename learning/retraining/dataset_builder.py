"""Controlled Candidate Dataset Builder for Phase 8 (Sections 19, 20, 44, 45, 46).

Enforces:
- Strict temporal boundary: Training data occurs strictly BEFORE validation data,
  and validation occurs strictly BEFORE test data.
- Strict Cutoff Invariant (Section 44): Data arriving after the training cutoff
  can NEVER leak into the historical training partition.
- Explicit Dataset Versioning: Every generated dataset receives an immutable dataset_version.
- Label & Feature Versioning: Models track exact feature_version and label_version.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from data.ingestion.models import Candle
from features.engine import FeatureEngine, candles_to_dataframe
from learning.dataset import (
    ChronologicalPreprocessor,
    ChronologicalSplitter,
    DatasetMetadata,
    DatasetSplits,
)
from learning.labels.engine import LabelEngine
from monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetBuildConfig:
    """Configuration for building an auditable, versioned training dataset."""
    dataset_version: str = "ds_v001"
    feature_version: str = "v1.0.0"
    label_version: str = "v1.0.0"
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    train_start: datetime | None = None
    train_end: datetime | None = None
    val_end: datetime | None = None
    test_end: datetime | None = None
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class CandidateDatasetBuilder:
    """Constructs versioned, leakage-free training datasets for candidate models."""

    def __init__(
        self,
        feature_engine: FeatureEngine | None = None,
        label_engine: LabelEngine | None = None,
    ) -> None:
        self.feature_engine = feature_engine or FeatureEngine()
        self.label_engine = label_engine or LabelEngine()

    def build_dataset(
        self,
        candles: Sequence[Candle] | pd.DataFrame,
        config: DatasetBuildConfig,
    ) -> DatasetSplits:
        """Construct versioned DatasetSplits strictly respecting temporal ordering."""
        if isinstance(candles, pd.DataFrame):
            df_candles = candles.copy()
        else:
            df_candles = candles_to_dataframe(candles)

        if df_candles.empty:
            raise ValueError("Cannot build dataset from empty candles.")

        time_col = "end_time" if "end_time" in df_candles.columns else "start_time"
        df_candles[time_col] = pd.to_datetime(df_candles[time_col], utc=True)
        df_candles = df_candles.sort_values(time_col).reset_index(drop=True)

        # 1. Enforce Training Cutoff if specified (Section 44)
        if config.test_end is not None:
            cutoff = config.test_end if config.test_end.tzinfo else config.test_end.replace(tzinfo=UTC)
            df_candles = df_candles[df_candles[time_col] <= cutoff].reset_index(drop=True)

        # 2. Compute Features and Labels
        features_df = self.feature_engine.compute_features_dataframe(df_candles)
        labels_df = self.label_engine.compute_labels(df_candles)

        # Avoid column collision: if a label name exists in features, prefix with 'target_'
        labels_to_add = labels_df.copy()
        rename_map = {
            col: f"target_{col}" for col in labels_to_add.columns if col in features_df.columns
        }
        if rename_map:
            labels_to_add = labels_to_add.rename(columns=rename_map)

        # Combine
        combined_df = pd.concat([features_df, labels_to_add], axis=1)
        combined_df[time_col] = df_candles[time_col]
        combined_df["timestamp"] = df_candles[time_col]

        meta = DatasetMetadata(
            dataset_id=config.dataset_version,
            feature_version=config.feature_version,
            label_version=config.label_version,
            symbol=config.symbol,
            timeframe=config.timeframe,
            start_time=pd.to_datetime(combined_df["timestamp"].iloc[0]).to_pydatetime(),
            end_time=pd.to_datetime(combined_df["timestamp"].iloc[-1]).to_pydatetime(),
            total_samples=len(combined_df),
            feature_columns=[c for c in features_df.columns if c not in (time_col, "timestamp")],
            target_columns=list(labels_to_add.columns),
            created_at=datetime.now(UTC),
        )

        # 3. Chronological Split (Section 20)
        splitter = ChronologicalSplitter(
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
        )
        splits = splitter.split(combined_df, meta)

        # 4. Anti-Leakage Temporal Assertions
        train_max = pd.to_datetime(splits.train_df["timestamp"]).max()
        val_min = pd.to_datetime(splits.val_df["timestamp"]).min()
        val_max = pd.to_datetime(splits.val_df["timestamp"]).max()
        test_min = pd.to_datetime(splits.test_df["timestamp"]).min()

        if train_max >= val_min:
            raise ValueError(f"Temporal leakage detected: train_max ({train_max}) >= val_min ({val_min})")
        if val_max >= test_min:
            raise ValueError(f"Temporal leakage detected: val_max ({val_max}) >= test_min ({test_min})")

        # Set Metadata
        splits.metadata.dataset_id = config.dataset_version
        splits.metadata.feature_version = config.feature_version
        splits.metadata.label_version = config.label_version
        splits.metadata.symbol = config.symbol
        splits.metadata.timeframe = config.timeframe

        logger.info(
            "candidate_dataset_built",
            dataset_version=config.dataset_version,
            train_samples=len(splits.train_df),
            val_samples=len(splits.val_df),
            test_samples=len(splits.test_df),
        )
        return splits
