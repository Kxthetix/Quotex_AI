"""Dataset generation, versioning, and chronological splitting for Phase 4.

Combines Phase 3 FeatureVectors with Phase 4 forward Labels into unified, version-locked
datasets for model training and scientific validation without lookahead leakage.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from analysis.regime.engine import MarketRegimeEngine
from data.ingestion.models import Candle
from features.engine import FeatureEngine, candles_to_dataframe
from learning.labels.engine import LabelEngine


@dataclass
class DatasetMetadata:
    """Metadata tracking dataset provenance for reproducible experiments."""

    dataset_id: str
    feature_version: str
    label_version: str
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_samples: int
    feature_columns: list[str]
    target_columns: list[str]
    created_at: datetime


@dataclass
class DatasetSplits:
    """Container for chronologically partitioned training, validation, and test datasets."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]
    metadata: DatasetMetadata


class DatasetGenerator:
    """Constructs version-locked tabular datasets pairing Features(T) with Labels(T+H)."""

    def __init__(
        self,
        feature_engine: FeatureEngine | None = None,
        label_engine: LabelEngine | None = None,
        regime_engine: MarketRegimeEngine | None = None,
    ) -> None:
        self.feature_engine = feature_engine or FeatureEngine()
        self.label_engine = label_engine or LabelEngine()
        self.regime_engine = regime_engine or MarketRegimeEngine()

    def create_dataset(
        self,
        candles: Sequence[Candle] | pd.DataFrame,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
    ) -> tuple[pd.DataFrame, DatasetMetadata]:
        """Generate a complete training dataset from a sequence of raw or normalized candles."""
        if isinstance(candles, pd.DataFrame):
            df_candles = candles.copy()
        else:
            df_candles = candles_to_dataframe(candles)

        if df_candles.empty:
            raise ValueError("Input candles cannot be empty.")

        time_col = "end_time" if "end_time" in df_candles.columns else "start_time"
        df_candles[time_col] = pd.to_datetime(df_candles[time_col], utc=True)
        df_candles = df_candles.sort_values(time_col).reset_index(drop=True)

        # 1. Compute features DataFrame
        features_df = self.feature_engine.compute_features_dataframe(df_candles)

        # 2. Compute labels DataFrame
        labels_df = self.label_engine.compute_labels(df_candles)

        # 3. Classify regimes
        # Convert df_candles into feature vectors for regime classification
        vectors = self.feature_engine.compute_historical_features(df_candles, symbol=symbol)
        regimes = [self.regime_engine.classify(v).regime.value for v in vectors]

        # Avoid column collision: if a label name exists in features, prefix with 'target_'
        labels_to_add = labels_df.copy()
        rename_map = {
            col: f"target_{col}" for col in labels_to_add.columns if col in features_df.columns
        }
        if rename_map:
            labels_to_add = labels_to_add.rename(columns=rename_map)

        meta_df = pd.DataFrame(
            {
                "timestamp": df_candles[time_col],
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": regimes,
            },
            index=df_candles.index,
        )
        combined = pd.concat([meta_df, features_df, labels_to_add], axis=1)

        feature_cols = list(features_df.columns)
        target_cols = list(labels_to_add.columns)

        # Drop rows where labels are NaN (the tail bars where future horizon extends past end of data)
        valid_dataset = combined.dropna(subset=target_cols).reset_index(drop=True)

        metadata = DatasetMetadata(
            dataset_id=f"ds_{uuid.uuid4().hex[:12]}",
            feature_version=self.feature_engine.config.version,
            label_version=self.label_engine.label_version,
            symbol=symbol,
            timeframe=timeframe,
            start_time=valid_dataset["timestamp"].iloc[0].to_pydatetime(),
            end_time=valid_dataset["timestamp"].iloc[-1].to_pydatetime(),
            total_samples=len(valid_dataset),
            feature_columns=feature_cols,
            target_columns=target_cols,
            created_at=datetime.now(UTC),
        )

        return valid_dataset, metadata


class ChronologicalSplitter:
    """Partitions time series datasets into strictly sequential Train, Validation, and Test splits."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, df: pd.DataFrame, metadata: DatasetMetadata) -> DatasetSplits:
        """Split DataFrame chronologically: max(train_time) < min(val_time) < min(test_time)."""
        sorted_df = df.sort_values("timestamp").reset_index(drop=True)
        n = len(sorted_df)

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_df = sorted_df.iloc[:n_train].copy().reset_index(drop=True)
        val_df = sorted_df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
        test_df = sorted_df.iloc[n_train + n_val :].copy().reset_index(drop=True)

        return DatasetSplits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_columns=metadata.feature_columns,
            target_columns=metadata.target_columns,
            metadata=metadata,
        )


class ChronologicalPreprocessor:
    """Leakage-safe feature scaling and imputation fitted exclusively on the training split."""

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.medians: dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "ChronologicalPreprocessor":
        """Learn scaling parameters and medians ONLY from training observations."""
        x_train = train_df[self.feature_columns].copy()

        # Learn medians on train split for safe imputation
        for col in self.feature_columns:
            median_val = float(x_train[col].median(skipna=True))
            self.medians[col] = 0.0 if np.isnan(median_val) else median_val
            x_train[col] = x_train[col].fillna(self.medians[col])

        self.scaler.fit(x_train.values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform input features using previously fitted training parameters."""
        if not self.is_fitted:
            raise RuntimeError(
                "ChronologicalPreprocessor must be fitted on train split before transforming."
            )

        x_df = df[self.feature_columns].copy()
        for col in self.feature_columns:
            x_df[col] = x_df[col].fillna(self.medians.get(col, 0.0))

        # Handle any residual infinities
        x_df = x_df.replace([np.inf, -np.inf], 0.0)

        return self.scaler.transform(x_df.values)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fit on train and transform train."""
        return self.fit(train_df).transform(train_df)
