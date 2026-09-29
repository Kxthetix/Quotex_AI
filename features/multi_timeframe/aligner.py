"""Multi-timeframe alignment engine for Phase 3.

Guarantees 100% leakage-safe alignment of higher-timeframe features with lower-timeframe
target timestamps using backward matching against completed bar availability times.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from features.config import MultiTimeframeConfig
from features.models import FeatureDefinition
from features.multi_timeframe.resampler import (
    parse_timeframe_to_timedelta,
    resample_candles_to_timeframe,
)
from features.registry import FeatureRegistry
from monitoring.logging import get_logger

logger = get_logger(__name__)


def compute_htf_indicators(htf_df: pd.DataFrame, config: MultiTimeframeConfig) -> pd.DataFrame:
    """Calculate core trend, momentum, and volatility indicators directly on higher-timeframe candles."""
    if htf_df.empty or len(htf_df) == 0:
        return pd.DataFrame()

    res = pd.DataFrame(index=htf_df.index)
    res["available_at"] = htf_df["available_at"]

    close = htf_df["close"]
    high = htf_df["high"]
    low = htf_df["low"]

    # 1. Higher-timeframe EMAs
    for p in config.higher_tf_ema_periods:
        res[f"ema_{p}"] = close.ewm(span=p, min_periods=p, adjust=False).mean()

    # 2. Higher-timeframe RSI
    rsi_p = config.higher_tf_rsi_period
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / rsi_p
    avg_gain = gain.ewm(alpha=alpha, min_periods=rsi_p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=rsi_p, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi)
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, rsi)
    res[f"rsi_{rsi_p}"] = rsi

    # 3. Higher-timeframe ATR
    atr_p = config.higher_tf_atr_period
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_p, min_periods=atr_p, adjust=False).mean()
    res[f"atr_{atr_p}"] = atr

    # 4. Higher-timeframe 1-bar return
    res["return_1"] = (close - close.shift(1)) / close.shift(1)

    return res


class MultiTimeframeAligner:
    """Aligns higher-timeframe features with base-timeframe prediction timestamps without future data leakage."""

    def __init__(self, config: MultiTimeframeConfig, base_timeframe: str = "1m") -> None:
        self.config = config
        self.base_timeframe = base_timeframe

    def align_historical(
        self,
        base_df: pd.DataFrame,
        htf_candles_by_tf: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Align all enabled higher-timeframe features to the base DataFrame.

        If `htf_candles_by_tf` is not provided, resamples base_df directly.
        Uses pd.merge_asof with direction='backward' strictly on completed bar end times.
        """
        if not self.config.enabled or not self.config.enabled_timeframes or base_df.empty:
            return pd.DataFrame(index=base_df.index)

        # Ensure base time column
        work_base = base_df.copy()
        if "end_time" in work_base.columns:
            target_col = "end_time"
        elif "timestamp" in work_base.columns:
            target_col = "timestamp"
        elif "start_time" in work_base.columns:
            delta = parse_timeframe_to_timedelta(self.base_timeframe)
            work_base["end_time"] = pd.to_datetime(work_base["start_time"], utc=True) + delta
            target_col = "end_time"
        else:
            raise ValueError(
                "Base DataFrame must contain 'end_time', 'timestamp', or 'start_time'."
            )

        work_base[target_col] = pd.to_datetime(work_base[target_col], utc=True)
        work_base = work_base.sort_values(target_col)

        combined_mtf = pd.DataFrame(index=work_base.index)

        for tf in self.config.enabled_timeframes:
            # Skip if timeframe is smaller or equal to base
            if parse_timeframe_to_timedelta(tf) <= parse_timeframe_to_timedelta(
                self.base_timeframe
            ):
                continue

            # Obtain HTF candles
            if htf_candles_by_tf and tf in htf_candles_by_tf:
                htf_df = htf_candles_by_tf[tf].copy()
                if "available_at" not in htf_df.columns:
                    htf_df["available_at"] = pd.to_datetime(htf_df["end_time"], utc=True)
            else:
                htf_df = resample_candles_to_timeframe(
                    work_base, target_timeframe=tf, base_timeframe=self.base_timeframe
                )

            if htf_df.empty:
                # Fill MTF columns with NaN if no completed candles exist
                for col in self._expected_feature_names(tf):
                    combined_mtf[col] = np.nan
                continue

            # Calculate indicators on the HTF bars
            indicators_df = compute_htf_indicators(htf_df, self.config)
            if indicators_df.empty:
                for col in self._expected_feature_names(tf):
                    combined_mtf[col] = np.nan
                continue

            indicators_df["available_at"] = pd.to_datetime(indicators_df["available_at"], utc=True)
            indicators_df = indicators_df.sort_values("available_at")

            # Rename indicator columns with mtf prefix
            rename_map = {
                col: f"mtf_{tf}_{col}" for col in indicators_df.columns if col != "available_at"
            }
            renamed_indicators = indicators_df.rename(columns=rename_map)

            # Strict backward merge: matches the latest HTF candle where available_at <= target_col
            merged = pd.merge_asof(
                work_base[[target_col]],
                renamed_indicators,
                left_on=target_col,
                right_on="available_at",
                direction="backward",
            )
            merged = merged.drop(columns=[target_col, "available_at"], errors="ignore")

            for col in merged.columns:
                combined_mtf[col] = merged[col].values

        return combined_mtf

    def align_live(
        self,
        target_timestamp: datetime,
        htf_candles_by_tf: dict[str, pd.DataFrame],
    ) -> dict[str, float | None]:
        """Align MTF features incrementally for a single timestamp T without looking ahead.

        Invariant: Only completed HTF candles with end_time <= target_timestamp are used.
        """
        results: dict[str, float | None] = {}
        target_dt = pd.to_datetime(target_timestamp, utc=True)

        for tf in self.config.enabled_timeframes:
            if parse_timeframe_to_timedelta(tf) <= parse_timeframe_to_timedelta(
                self.base_timeframe
            ):
                continue

            expected_names = self._expected_feature_names(tf)

            if tf not in htf_candles_by_tf or htf_candles_by_tf[tf].empty:
                for name in expected_names:
                    results[name] = None
                continue

            htf_df = htf_candles_by_tf[tf].copy()
            if "available_at" not in htf_df.columns:
                htf_df["available_at"] = pd.to_datetime(htf_df["end_time"], utc=True)
            else:
                htf_df["available_at"] = pd.to_datetime(htf_df["available_at"], utc=True)

            # Strictly filter completed candles <= target_dt
            valid_completed = htf_df[htf_df["available_at"] <= target_dt]
            if valid_completed.empty:
                for name in expected_names:
                    results[name] = None
                continue

            # Compute features up to this completed bar
            ind_df = compute_htf_indicators(valid_completed, self.config)
            latest_row = ind_df.iloc[-1]

            for raw_col in ind_df.columns:
                if raw_col == "available_at":
                    continue
                feature_name = f"mtf_{tf}_{raw_col}"
                val = latest_row[raw_col]
                results[feature_name] = float(val) if pd.notna(val) else None

        return results

    def _expected_feature_names(self, tf: str) -> list[str]:
        """List all expected feature names for a given higher timeframe."""
        names = []
        for p in self.config.higher_tf_ema_periods:
            names.append(f"mtf_{tf}_ema_{p}")
        names.append(f"mtf_{tf}_rsi_{self.config.higher_tf_rsi_period}")
        names.append(f"mtf_{tf}_atr_{self.config.higher_tf_atr_period}")
        names.append(f"mtf_{tf}_return_1")
        return names


def register_multi_timeframe_features(
    registry: FeatureRegistry,
    config: MultiTimeframeConfig,
    base_timeframe: str = "1m",
    version: str = "v1.0.0",
) -> None:
    """Register all multi-timeframe feature definitions into the FeatureRegistry."""
    if not config.enabled:
        return

    base_delta = parse_timeframe_to_timedelta(base_timeframe)

    for tf in config.enabled_timeframes:
        tf_delta = parse_timeframe_to_timedelta(tf)
        if tf_delta <= base_delta:
            continue

        ratio = int(tf_delta.total_seconds() // base_delta.total_seconds())

        for p in config.higher_tf_ema_periods:
            registry.register(
                FeatureDefinition(
                    name=f"mtf_{tf}_ema_{p}",
                    category="multi_timeframe",
                    version=version,
                    parameters={"timeframe": tf, "period": p},
                    lookback=p * ratio,
                    timeframe=tf,
                    description=f"Higher-timeframe {tf} EMA({p}) aligned at completed candle boundaries",
                )
            )

        rsi_p = config.higher_tf_rsi_period
        registry.register(
            FeatureDefinition(
                name=f"mtf_{tf}_rsi_{rsi_p}",
                category="multi_timeframe",
                version=version,
                parameters={"timeframe": tf, "period": rsi_p},
                lookback=(rsi_p + 1) * ratio,
                timeframe=tf,
                description=f"Higher-timeframe {tf} RSI({rsi_p}) aligned at completed candle boundaries",
            )
        )

        atr_p = config.higher_tf_atr_period
        registry.register(
            FeatureDefinition(
                name=f"mtf_{tf}_atr_{atr_p}",
                category="multi_timeframe",
                version=version,
                parameters={"timeframe": tf, "period": atr_p},
                lookback=(atr_p + 1) * ratio,
                timeframe=tf,
                description=f"Higher-timeframe {tf} ATR({atr_p}) aligned at completed candle boundaries",
            )
        )

        registry.register(
            FeatureDefinition(
                name=f"mtf_{tf}_return_1",
                category="multi_timeframe",
                version=version,
                parameters={"timeframe": tf, "period": 1},
                lookback=2 * ratio,
                timeframe=tf,
                description=f"Higher-timeframe {tf} 1-bar return aligned at completed candle boundaries",
            )
        )
