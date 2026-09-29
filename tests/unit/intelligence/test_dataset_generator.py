"""Tests for DatasetGenerator, ChronologicalSplitter, and ChronologicalPreprocessor."""

import numpy as np
import pandas as pd
import pytest

from learning.dataset import (
    ChronologicalPreprocessor,
    ChronologicalSplitter,
    DatasetGenerator,
    DatasetMetadata,
)
from tests.unit.intelligence.test_leakage_critical import _generate_synthetic_candles


class TestDatasetGenerator:
    """Tests for DatasetGenerator."""

    def test_create_dataset_success(self) -> None:
        """Verify dataset generation from synthetic candles."""
        candles = _generate_synthetic_candles(n_candles=120)
        generator = DatasetGenerator()

        df, metadata = generator.create_dataset(candles, symbol="ETH/USDT", timeframe="1m")

        assert not df.empty
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.symbol == "ETH/USDT"
        assert metadata.timeframe == "1m"
        assert metadata.feature_version == "v1.0.0"
        assert metadata.label_version == "v1.0.0"
        assert metadata.total_samples == len(df)
        assert len(metadata.feature_columns) > 0
        assert len(metadata.target_columns) > 0

        # Assert all metadata feature columns exist in df
        for col in metadata.feature_columns:
            assert col in df.columns
        for col in metadata.target_columns:
            assert col in df.columns

        # Verify no NaN values in target columns
        assert not df[metadata.target_columns].isna().any().any()

        # Check timestamp ordering
        assert df["timestamp"].is_monotonic_increasing

    def test_empty_candles_raises_error(self) -> None:
        """Verify passing empty DataFrame raises ValueError."""
        generator = DatasetGenerator()
        with pytest.raises(ValueError, match="cannot be empty"):
            generator.create_dataset(pd.DataFrame())


class TestChronologicalSplitter:
    """Tests for ChronologicalSplitter."""

    def test_splitter_ratios_validation(self) -> None:
        """Verify splitter rejects ratios that do not sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ChronologicalSplitter(train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)

    def test_chronological_splits(self) -> None:
        """Verify sequential temporal partitioning."""
        candles = _generate_synthetic_candles(n_candles=150)
        generator = DatasetGenerator()
        df, metadata = generator.create_dataset(candles, symbol="BTC/USDT", timeframe="1m")

        splitter = ChronologicalSplitter(train_ratio=0.60, val_ratio=0.20, test_ratio=0.20)
        splits = splitter.split(df, metadata)

        assert len(splits.train_df) + len(splits.val_df) + len(splits.test_df) == len(df)
        assert splits.train_df["timestamp"].max() < splits.val_df["timestamp"].min()
        assert splits.val_df["timestamp"].max() < splits.test_df["timestamp"].min()


class TestChronologicalPreprocessor:
    """Tests for ChronologicalPreprocessor."""

    def test_fit_transform_isolation(self) -> None:
        """Verify scaling and imputation learn strictly from train split."""
        feature_cols = ["feat_a", "feat_b"]
        train_df = pd.DataFrame(
            {
                "feat_a": [10.0, 20.0, 30.0, np.nan],
                "feat_b": [100.0, 200.0, 300.0, 400.0],
            }
        )
        val_df = pd.DataFrame(
            {
                "feat_a": [np.nan, 20.0],
                "feat_b": [200.0, np.inf],
            }
        )

        preprocessor = ChronologicalPreprocessor(feature_columns=feature_cols)

        # Calling transform before fit must raise RuntimeError
        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(val_df)

        X_train_scaled = preprocessor.fit_transform(train_df)
        assert preprocessor.is_fitted
        # Train median for feat_a is median([10, 20, 30]) = 20.0
        assert preprocessor.medians["feat_a"] == 20.0

        X_val_scaled = preprocessor.transform(val_df)
        assert X_val_scaled.shape == (2, 2)
        assert not np.isnan(X_val_scaled).any()
        assert not np.isinf(X_val_scaled).any()
