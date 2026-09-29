"""Sequence Dataset Builder for Advanced Temporal Models (Section 21).

Constructs sliding temporal feature sequences:
    [T - L + 1, ..., T] -> Target at T + h
with strict chronological ordering and zero lookahead leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SequenceDataset:
    X: np.ndarray  # Shape: (N, sequence_length, num_features)
    y: np.ndarray | None  # Shape: (N,) or (N, num_classes)
    feature_names: list[str]
    sequence_length: int
    timestamps: list[Any]
    dataset_version: str = "seq_dataset_v1.0"


class SequenceDatasetBuilder:
    """Constructs rolling temporal sequence windows strictly respecting causality."""

    def __init__(
        self,
        sequence_length: int = 15,
        stride: int = 1,
        dataset_version: str = "seq_dataset_v1.0",
    ) -> None:
        if sequence_length < 2:
            raise ValueError(f"sequence_length must be >= 2, got {sequence_length}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.sequence_length = sequence_length
        self.stride = stride
        self.dataset_version = dataset_version

    def build_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str | None = None,
        timestamp_col: str | None = None,
    ) -> SequenceDataset:
        """Construct (N, L, D) 3D array from chronological DataFrame.
        
        Section 21: Strictly guarantees that the sequence at index i ends at observation T,
        never incorporating data after T.
        """
        if len(df) < self.sequence_length:
            raise ValueError(
                f"DataFrame length ({len(df)}) is less than sequence length ({self.sequence_length})"
            )

        features_data = df[feature_cols].to_numpy(dtype=np.float32)
        target_data = df[target_col].to_numpy() if target_col and target_col in df.columns else None

        num_samples = (len(df) - self.sequence_length) // self.stride + 1
        num_features = len(feature_cols)

        X = np.zeros((num_samples, self.sequence_length, num_features), dtype=np.float32)
        y = np.zeros(num_samples, dtype=target_data.dtype) if target_data is not None else None
        timestamps: list[Any] = []

        for sample_idx in range(num_samples):
            start_idx = sample_idx * self.stride
            end_idx = start_idx + self.sequence_length
            X[sample_idx] = features_data[start_idx:end_idx]

            # The decision point is the last timestamp in the window (T)
            if timestamp_col and timestamp_col in df.columns:
                timestamps.append(df[timestamp_col].iloc[end_idx - 1])

            if target_data is not None:
                # Target corresponds to the outcome following the window end
                y[sample_idx] = target_data[end_idx - 1]

        return SequenceDataset(
            X=X,
            y=y,
            feature_names=feature_cols,
            sequence_length=self.sequence_length,
            timestamps=timestamps,
            dataset_version=self.dataset_version,
        )
