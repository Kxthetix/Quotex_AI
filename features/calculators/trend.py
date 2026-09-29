"""Trend feature calculators for Phase 3.

Implements deterministic trend metrics:
- Exponential Moving Average (EMA) families
- Simple Moving Average (SMA) families
- Average Directional Index (ADX, +DI, -DI)
- Linear regression slope of price
- Higher-high, lower-low, higher-low, and lower-high structural indicators
"""

import numpy as np
import pandas as pd

from features.config import TrendFeatureConfig
from features.models import FeatureDefinition
from features.registry import FeatureRegistry


def compute_trend_features(df: pd.DataFrame, config: TrendFeatureConfig) -> pd.DataFrame:
    """Compute all configured trend features from an OHLCV DataFrame."""
    res = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 1. EMA families
    for period in config.ema_periods:
        # Standard EMA with min_periods to enforce proper warm-up
        res[f"ema_{period}"] = close.ewm(span=period, min_periods=period, adjust=False).mean()

    # 2. SMA families
    for period in config.sma_periods:
        res[f"sma_{period}"] = close.rolling(window=period, min_periods=period).mean()

    # 3. ADX, +DI, -DI
    adx_df = _compute_adx(high, low, close, period=config.adx_period)
    res = pd.concat([res, adx_df], axis=1)

    # 4. Normalized Linear Regression Slopes
    for period in config.slope_periods:
        res[f"slope_{period}"] = _compute_rolling_slope(close, period=period)

    # 5. Trend Structure (Higher-High, Lower-Low, etc.)
    struct_df = _compute_trend_structure(high, low, lookback=config.structure_lookback)
    res = pd.concat([res, struct_df], axis=1)

    return res


def _compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """Calculate ADX, +DI, and -DI using Wilder's smoothing method."""
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    # Wilder smoothing: alpha = 1 / period
    alpha = 1.0 / period
    tr_smooth = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * (di_diff / di_sum.replace(0, np.nan))

    # ADX is Wilder smooth of DX (requires 2 * period for complete warm-up)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {
            f"adx_{period}": adx,
            f"plus_di_{period}": plus_di,
            f"minus_di_{period}": minus_di,
        },
        index=high.index,
    )


def _compute_rolling_slope(series: pd.Series, period: int) -> pd.Series:
    """Calculate normalized linear regression slope over rolling window."""
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)

    x = np.arange(period)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def calc_slope(window: np.ndarray) -> float:
        if np.isnan(window).any():
            return np.nan
        y_mean = window.mean()
        if y_mean == 0:
            return 0.0
        cov = ((x - x_mean) * (window - y_mean)).sum()
        slope = cov / x_var
        return float(slope / y_mean)  # Normalize by price level

    # Vectorized / rolling apply with raw numpy
    return series.rolling(window=period, min_periods=period).apply(calc_slope, raw=True)


def _compute_trend_structure(high: pd.Series, low: pd.Series, lookback: int = 20) -> pd.DataFrame:
    """Detect higher-high, lower-low, higher-low, and lower-high states."""
    # Previous N completed candles (shift by 1 to exclude current candle)
    prior_high_max = high.shift(1).rolling(window=lookback, min_periods=lookback).max()
    prior_high_min = high.shift(1).rolling(window=lookback, min_periods=lookback).min()
    prior_low_min = low.shift(1).rolling(window=lookback, min_periods=lookback).min()
    prior_low_max = low.shift(1).rolling(window=lookback, min_periods=lookback).max()

    higher_high = np.where(high > prior_high_max, 1.0, 0.0)
    lower_low = np.where(low < prior_low_min, 1.0, 0.0)
    higher_low = np.where(low > prior_low_max, 1.0, 0.0)
    lower_high = np.where(high < prior_high_min, 1.0, 0.0)

    # Where prior values are NaN (warming up), structure must be NaN
    mask = prior_high_max.isna()
    higher_high = np.where(mask, np.nan, higher_high)
    lower_low = np.where(mask, np.nan, lower_low)
    higher_low = np.where(mask, np.nan, higher_low)
    lower_high = np.where(mask, np.nan, lower_high)

    return pd.DataFrame(
        {
            f"higher_high_{lookback}": higher_high,
            f"lower_low_{lookback}": lower_low,
            f"higher_low_{lookback}": higher_low,
            f"lower_high_{lookback}": lower_high,
        },
        index=high.index,
    )


def register_trend_features(
    registry: FeatureRegistry, config: TrendFeatureConfig, version: str = "v1.0.0"
) -> None:
    """Register all trend feature definitions into the FeatureRegistry."""
    for p in config.ema_periods:
        registry.register(
            FeatureDefinition(
                name=f"ema_{p}",
                category="trend",
                version=version,
                parameters={"period": p},
                lookback=p,
                description=f"Exponential Moving Average with period {p}",
            )
        )

    for p in config.sma_periods:
        registry.register(
            FeatureDefinition(
                name=f"sma_{p}",
                category="trend",
                version=version,
                parameters={"period": p},
                lookback=p,
                description=f"Simple Moving Average with period {p}",
            )
        )

    adx_p = config.adx_period
    registry.register(
        FeatureDefinition(
            name=f"adx_{adx_p}",
            category="trend",
            version=version,
            parameters={"period": adx_p},
            lookback=adx_p * 2,
            description=f"Average Directional Index with period {adx_p}",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"plus_di_{adx_p}",
            category="trend",
            version=version,
            parameters={"period": adx_p},
            lookback=adx_p,
            description=f"Positive Directional Indicator (+DI) with period {adx_p}",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"minus_di_{adx_p}",
            category="trend",
            version=version,
            parameters={"period": adx_p},
            lookback=adx_p,
            description=f"Negative Directional Indicator (-DI) with period {adx_p}",
        )
    )

    for p in config.slope_periods:
        registry.register(
            FeatureDefinition(
                name=f"slope_{p}",
                category="trend",
                version=version,
                parameters={"period": p},
                lookback=p,
                description=f"Normalized linear regression slope over {p} periods",
            )
        )

    lb = config.structure_lookback
    registry.register(
        FeatureDefinition(
            name=f"higher_high_{lb}",
            category="trend",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description=f"Binary indicator if high exceeds prior {lb}-period max high",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"lower_low_{lb}",
            category="trend",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description=f"Binary indicator if low is below prior {lb}-period min low",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"higher_low_{lb}",
            category="trend",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description=f"Binary indicator if low is above prior {lb}-period max low",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"lower_high_{lb}",
            category="trend",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description=f"Binary indicator if high is below prior {lb}-period min high",
        )
    )
