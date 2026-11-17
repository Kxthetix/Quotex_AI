"""Master Feature Engine orchestrator for Phase 3.

Coordinates deterministic, versioned, leakage-free feature calculations across
all categories (price, trend, momentum, volatility, volume, structure, multi-timeframe).
Supports both historical batch processing and live incremental calculation.
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from data.ingestion.models import Candle
from features.calculators.momentum import (
    compute_momentum_features,
    register_momentum_features,
)
from features.calculators.price import (
    compute_price_features,
    register_price_features,
)
from features.calculators.structure import (
    compute_structure_features,
    register_structure_features,
)
from features.calculators.trend import (
    compute_trend_features,
    register_trend_features,
)
from features.calculators.volatility import (
    compute_volatility_features,
    register_volatility_features,
)
from features.calculators.volume import (
    compute_volume_features,
    register_volume_features,
)
from features.config import FeatureEngineConfig, get_default_feature_config
from features.models import FeatureValidity, FeatureVector
from features.multi_timeframe.aligner import (
    MultiTimeframeAligner,
    register_multi_timeframe_features,
)
from features.registry import FeatureRegistry, default_registry
from monitoring.logging import get_logger

logger = get_logger(__name__)


def candles_to_dataframe(candles: Sequence[Candle]) -> pd.DataFrame:
    """Convert a sequence of Candle objects to a standardized OHLCV DataFrame."""
    if not candles:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "start_time", "end_time", "symbol"]
        )

    records = [
        {
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "symbol": c.symbol,
        }
        for c in candles
    ]
    df = pd.DataFrame(records)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    df = df.sort_values("start_time").reset_index(drop=True)
    return df


class FeatureEngine:
    """Deterministic, versioned Feature Engine for market data transformation."""

    def __init__(
        self,
        config: FeatureEngineConfig | None = None,
        registry: FeatureRegistry | None = None,
    ) -> None:
        self.config = config or get_default_feature_config()
        self.registry = registry or default_registry
        self.mtf_aligner = MultiTimeframeAligner(
            config=self.config.multi_timeframe,
            base_timeframe=self.config.base_timeframe,
        )

        # Register all configured features into the catalog
        self._register_all_features()

        # Operational metrics tracking (Section 41)
        self.metrics = {
            "total_vectors_calculated": 0,
            "valid_vectors_count": 0,
            "warming_up_vectors_count": 0,
            "invalid_vectors_count": 0,
            "missing_input_count": 0,
            "calculation_errors_count": 0,
        }

    def _register_all_features(self) -> None:
        """Register metadata definitions for all enabled features."""
        version = self.config.version
        register_price_features(self.registry, self.config.price, version=version)
        register_trend_features(self.registry, self.config.trend, version=version)
        register_momentum_features(self.registry, self.config.momentum, version=version)
        register_volatility_features(self.registry, self.config.volatility, version=version)
        register_volume_features(self.registry, self.config.volume, version=version)
        register_structure_features(self.registry, self.config.structure, version=version)
        register_multi_timeframe_features(
            self.registry,
            self.config.multi_timeframe,
            base_timeframe=self.config.base_timeframe,
            version=version,
        )

    def compute_features_dataframe(
        self,
        df: pd.DataFrame,
        htf_candles_by_tf: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Vectorized computation of all enabled feature categories.

        Returns a consolidated DataFrame where each column is an exact feature name.
        """
        if df.empty:
            return pd.DataFrame()

        # 1. Price
        price_df = compute_price_features(df, self.config.price)

        # 2. Trend
        trend_df = compute_trend_features(df, self.config.trend)

        # 3. Momentum
        momentum_df = compute_momentum_features(df, self.config.momentum)

        # 4. Volatility
        volatility_df = compute_volatility_features(df, self.config.volatility)

        # 5. Volume
        volume_df = compute_volume_features(df, self.config.volume)

        # 6. Structure
        structure_df = compute_structure_features(df, self.config.structure)

        feature_blocks = [
            price_df,
            trend_df,
            momentum_df,
            volatility_df,
            volume_df,
            structure_df,
        ]

        # 7. Multi-timeframe features (if enabled)
        if self.config.multi_timeframe.enabled:
            mtf_df = self.mtf_aligner.align_historical(df, htf_candles_by_tf=htf_candles_by_tf)
            if not mtf_df.empty:
                feature_blocks.append(mtf_df)

        combined = pd.concat(feature_blocks, axis=1)
        return combined

    def compute_historical_features(
        self,
        candles: Sequence[Candle] | pd.DataFrame,
        symbol: str | None = None,
        htf_candles_by_tf: dict[str, pd.DataFrame] | None = None,
    ) -> list[FeatureVector]:
        """Compute versioned feature vectors for a historical sequence of candles."""
        if isinstance(candles, pd.DataFrame):
            df = candles.copy()
            if "symbol" in df.columns and len(df) > 0:
                sym = symbol or str(df["symbol"].iloc[0])
            else:
                sym = symbol or "UNKNOWN"
        else:
            if not candles:
                return []
            sym = symbol or candles[0].symbol
            df = candles_to_dataframe(candles)

        if df.empty:
            return []

        # Validate time column
        if "end_time" in df.columns:
            time_series = pd.to_datetime(df["end_time"], utc=True)
        elif "timestamp" in df.columns:
            time_series = pd.to_datetime(df["timestamp"], utc=True)
        elif "start_time" in df.columns:
            time_series = pd.to_datetime(df["start_time"], utc=True)
        else:
            raise ValueError(
                "Input candle data must contain 'end_time', 'timestamp', or 'start_time'."
            )

        try:
            feat_df = self.compute_features_dataframe(df, htf_candles_by_tf=htf_candles_by_tf)
        except Exception as e:
            logger.error("Error calculating feature dataframe: %s", e)
            self.metrics["calculation_errors_count"] += 1
            raise

        vectors: list[FeatureVector] = []
        feature_names = list(feat_df.columns)

        # Pre-query lookbacks from registry for fast validity assessment
        lookback_map: dict[str, int] = {}
        for name in feature_names:
            defn = self.registry.get(name)
            lookback_map[name] = defn.lookback if defn else 1

        n_rows = len(df)
        feat_matrix = feat_df.to_numpy()
        timestamps = [t.to_pydatetime() if hasattr(t, "to_pydatetime") else t for t in time_series]
        lookback_arr = [lookback_map[name] for name in feature_names]
        now_utc = datetime.now(UTC)
        base_tf = self.config.base_timeframe
        feat_ver = self.config.version

        for i in range(n_rows):
            ts = timestamps[i]
            row_vals = feat_matrix[i]
            row_dict: dict[str, float | None] = {}
            val_dict: dict[str, FeatureValidity] = {}
            has_warming_up = False
            has_invalid = False

            for j, name in enumerate(feature_names):
                val = row_vals[j]
                if np.isnan(val) or np.isinf(val):
                    row_dict[name] = None
                    if i < lookback_arr[j]:
                        val_dict[name] = FeatureValidity.WARMING_UP
                        has_warming_up = True
                    else:
                        val_dict[name] = FeatureValidity.INVALID
                        has_invalid = True
                else:
                    row_dict[name] = float(val)
                    val_dict[name] = FeatureValidity.VALID

            is_vec_valid = not (has_warming_up or has_invalid)

            # Update metrics
            self.metrics["total_vectors_calculated"] += 1
            if is_vec_valid:
                self.metrics["valid_vectors_count"] += 1
            elif has_invalid:
                self.metrics["invalid_vectors_count"] += 1
            elif has_warming_up:
                self.metrics["warming_up_vectors_count"] += 1

            vectors.append(
                FeatureVector(
                    symbol=sym,
                    timestamp=ts,
                    timeframe=base_tf,
                    feature_version=feat_ver,
                    values=row_dict,
                    validity=val_dict,
                    is_valid=is_vec_valid,
                    calculated_at=now_utc,
                )
            )

        return vectors

    def compute_live_features(
        self,
        candle_buffer: Sequence[Candle] | pd.DataFrame,
        symbol: str | None = None,
        htf_candles_by_tf: dict[str, pd.DataFrame] | None = None,
    ) -> FeatureVector:
        """Compute the current feature vector for the latest candle in an active rolling buffer.

        Guaranteed consistency: Batch computation and incremental live computation over
        the same completed historical bars yield identical results.
        """
        if isinstance(candle_buffer, pd.DataFrame):
            df = candle_buffer.copy()
            sym = symbol or (str(df["symbol"].iloc[-1]) if "symbol" in df.columns else "UNKNOWN")
        else:
            if not candle_buffer:
                raise ValueError("Candle buffer cannot be empty for live feature calculation.")
            sym = symbol or candle_buffer[-1].symbol
            df = candles_to_dataframe(candle_buffer)

        # Batch compute on the rolling window
        vectors = self.compute_historical_features(
            df, symbol=sym, htf_candles_by_tf=htf_candles_by_tf
        )
        if not vectors:
            raise ValueError("No feature vectors could be computed from candle buffer.")

        # Return the vector corresponding to the latest bar
        return vectors[-1]

    def get_metrics(self) -> dict[str, int]:
        """Return operational feature calculation metrics."""
        return dict(self.metrics)

    def reset_metrics(self) -> None:
        """Reset operational metrics."""
        for k in self.metrics:
            self.metrics[k] = 0
