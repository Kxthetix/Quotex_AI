"""Volatility feature calculators for Phase 3.

Implements deterministic volatility metrics:
- Average True Range (ATR) using Wilder's smoothing
- Rolling volatility (sample standard deviation of percentage returns)
- Historical volatility (annualized standard deviation of log returns)
"""

import numpy as np
import pandas as pd

from features.config import VolatilityFeatureConfig
from features.models import FeatureDefinition
from features.registry import FeatureRegistry


def compute_volatility_features(df: pd.DataFrame, config: VolatilityFeatureConfig) -> pd.DataFrame:
    """Compute all configured volatility features from an OHLCV DataFrame."""
    res = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 1. Average True Range (ATR)
    for period in config.atr_periods:
        res[f"atr_{period}"] = _compute_atr(high, low, close, period=period)

    # 2. Rolling Volatility (std of 1-period simple returns)
    returns = close.pct_change()
    for period in config.rolling_vol_periods:
        res[f"rolling_vol_{period}"] = returns.rolling(window=period, min_periods=period).std()

    # 3. Historical Volatility (std of log returns, annualized for minute data: sqrt(252 * 1440))
    log_returns = np.log(close / close.shift(1))
    # Annualization factor for 24/7 minute bars: 365 days * 1440 minutes/day = 525,600
    annualize_factor = np.sqrt(365 * 1440)
    for period in config.historical_vol_periods:
        res[f"historical_vol_{period}"] = (
            log_returns.rolling(window=period, min_periods=period).std() * annualize_factor
        )

    return res


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range using Wilder's exponential smoothing."""
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return pd.Series(atr, index=close.index)


def register_volatility_features(
    registry: FeatureRegistry, config: VolatilityFeatureConfig, version: str = "v1.0.0"
) -> None:
    """Register all volatility feature definitions into the FeatureRegistry."""
    for p in config.atr_periods:
        registry.register(
            FeatureDefinition(
                name=f"atr_{p}",
                category="volatility",
                version=version,
                parameters={"period": p},
                lookback=p,
                description=f"Average True Range with period {p}",
            )
        )

    for p in config.rolling_vol_periods:
        registry.register(
            FeatureDefinition(
                name=f"rolling_vol_{p}",
                category="volatility",
                version=version,
                parameters={"period": p},
                lookback=p,
                description=f"Sample standard deviation of 1-period returns over {p} bars",
            )
        )

    for p in config.historical_vol_periods:
        registry.register(
            FeatureDefinition(
                name=f"historical_vol_{p}",
                category="volatility",
                version=version,
                parameters={"period": p},
                lookback=p,
                description=f"Annualized historical volatility of log returns over {p} bars",
            )
        )
