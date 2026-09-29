"""Volume feature calculators for Phase 3.

Implements deterministic volume metrics:
- Relative Volume (current volume / baseline rolling average volume)
- Volume Simple Moving Averages (Volume SMA)
- Volume Spikes (binary state where volume exceeds threshold multiplier)
- Volume Trends (normalized slope of volume over lookback)
"""

import numpy as np
import pandas as pd

from features.config import VolumeFeatureConfig
from features.models import FeatureDefinition
from features.registry import FeatureRegistry


def compute_volume_features(df: pd.DataFrame, config: VolumeFeatureConfig) -> pd.DataFrame:
    """Compute all configured volume features from an OHLCV DataFrame."""
    res = pd.DataFrame(index=df.index)

    if "volume" not in df.columns:
        return res

    volume = df["volume"]

    # 1. Volume Simple Moving Averages
    for period in config.volume_ma_periods:
        res[f"volume_ma_{period}"] = volume.rolling(window=period, min_periods=period).mean()

    # 2. Relative Volume (current volume / prior N-period average volume)
    for period in config.relative_vol_periods:
        prior_ma = volume.shift(1).rolling(window=period, min_periods=period).mean()
        # Avoid division by zero
        rel_vol = volume / prior_ma.replace(0, np.nan)
        res[f"relative_vol_{period}"] = rel_vol

    # 3. Volume Spikes (volume > threshold * baseline volume)
    base_ma_20 = volume.shift(1).rolling(window=20, min_periods=20).mean()
    spike_thresh = config.volume_spike_threshold
    spike = np.where(volume > (spike_thresh * base_ma_20), 1.0, 0.0)
    spike = np.where(base_ma_20.isna(), np.nan, spike)
    res["volume_spike"] = spike

    # 4. Volume Trends (normalized slope over lookback)
    for period in config.volume_trend_periods:
        res[f"volume_trend_{period}"] = _compute_volume_trend(volume, period=period)

    return res


def _compute_volume_trend(volume: pd.Series, period: int) -> pd.Series:
    """Calculate normalized slope of volume over lookback window."""
    if len(volume) < period:
        return pd.Series(np.nan, index=volume.index)

    x = np.arange(period)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def calc_slope(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        y_mean = window.mean()
        if y_mean <= 0:
            return 0.0
        cov = ((x - x_mean) * (window - y_mean)).sum()
        slope = cov / x_var
        return float(slope / y_mean)

    return volume.rolling(window=period, min_periods=period).apply(calc_slope, raw=True)


def register_volume_features(
    registry: FeatureRegistry, config: VolumeFeatureConfig, version: str = "v1.0.0"
) -> None:
    """Register all volume feature definitions into the FeatureRegistry."""
    for p in config.volume_ma_periods:
        registry.register(
            FeatureDefinition(
                name=f"volume_ma_{p}",
                category="volume",
                version=version,
                parameters={"period": p},
                lookback=p,
                description=f"Simple Moving Average of volume over {p} bars",
            )
        )

    for p in config.relative_vol_periods:
        registry.register(
            FeatureDefinition(
                name=f"relative_vol_{p}",
                category="volume",
                version=version,
                parameters={"period": p},
                lookback=p + 1,
                description=f"Current volume relative to prior {p}-period average volume",
            )
        )

    registry.register(
        FeatureDefinition(
            name="volume_spike",
            category="volume",
            version=version,
            parameters={"multiplier": config.volume_spike_threshold, "lookback": 20},
            lookback=21,
            description="Binary indicator if volume exceeds 2x 20-period baseline",
        )
    )

    for p in config.volume_trend_periods:
        registry.register(
            FeatureDefinition(
                name=f"volume_trend_{p}",
                category="volume",
                version=version,
                parameters={"period": p},
                lookback=p,
                description=f"Normalized linear regression slope of volume over {p} bars",
            )
        )
