"""Momentum feature calculators for Phase 3.

Implements deterministic momentum oscillators:
- Relative Strength Index (RSI) using Wilder's smoothing
- Moving Average Convergence Divergence (MACD line, signal line, histogram)
- Rate of Change (ROC)
- Fast and Slow Stochastic (%K, %D)
"""

import numpy as np
import pandas as pd

from features.config import MomentumFeatureConfig
from features.models import FeatureDefinition
from features.registry import FeatureRegistry


def compute_momentum_features(df: pd.DataFrame, config: MomentumFeatureConfig) -> pd.DataFrame:
    """Compute all configured momentum features from an OHLCV DataFrame."""
    res = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 1. RSI families
    for period in config.rsi_periods:
        res[f"rsi_{period}"] = _compute_rsi(close, period=period)

    # 2. MACD (Line, Signal, Histogram)
    macd_df = _compute_macd(
        close,
        fast=config.macd_fast,
        slow=config.macd_slow,
        signal=config.macd_signal,
    )
    res = pd.concat([res, macd_df], axis=1)

    # 3. Rate of Change (ROC)
    for period in config.roc_periods:
        res[f"roc_{period}"] = 100.0 * (close - close.shift(period)) / close.shift(period)

    # 4. Stochastic (%K, %D)
    stoch_df = _compute_stochastic(
        high,
        low,
        close,
        k_period=config.stoch_k_period,
        d_period=config.stoch_d_period,
        smooth_k=config.stoch_smooth_k,
    )
    res = pd.concat([res, stoch_df], axis=1)

    return res


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index using Wilder's exponential smoothing."""
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing: alpha = 1 / period
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases: when avg_loss is 0 and avg_gain > 0, RSI = 100; if both 0, RSI = 50
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi)
    rsi = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, rsi)

    return pd.Series(rsi, index=close.index)


def _compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_hist,
        },
        index=close.index,
    )


def _compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """Calculate Stochastic %K and %D oscillators."""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    denom = highest_high - lowest_low
    # Avoid divide-by-zero on flat periods
    raw_k = 100.0 * (close - lowest_low) / denom.replace(0, np.nan)
    raw_k = np.where(denom == 0, 50.0, raw_k)
    raw_k_series = pd.Series(raw_k, index=close.index)

    # Smooth %K if smooth_k > 1 (Slow Stochastic)
    stoch_k = (
        raw_k_series.rolling(window=smooth_k, min_periods=smooth_k).mean()
        if smooth_k > 1
        else raw_k_series
    )
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame(
        {
            f"stoch_k_{k_period}": stoch_k,
            f"stoch_d_{k_period}": stoch_d,
        },
        index=close.index,
    )


def register_momentum_features(
    registry: FeatureRegistry, config: MomentumFeatureConfig, version: str = "v1.0.0"
) -> None:
    """Register all momentum feature definitions into the FeatureRegistry."""
    for p in config.rsi_periods:
        registry.register(
            FeatureDefinition(
                name=f"rsi_{p}",
                category="momentum",
                version=version,
                parameters={"period": p},
                lookback=p,
                description=f"Relative Strength Index with Wilder smoothing ({p} periods)",
            )
        )

    registry.register(
        FeatureDefinition(
            name="macd_line",
            category="momentum",
            version=version,
            parameters={"fast": config.macd_fast, "slow": config.macd_slow},
            lookback=config.macd_slow,
            description="MACD line: EMA(fast) - EMA(slow)",
        )
    )
    registry.register(
        FeatureDefinition(
            name="macd_signal",
            category="momentum",
            version=version,
            parameters={"signal": config.macd_signal},
            lookback=config.macd_slow + config.macd_signal,
            description="MACD signal line: EMA(MACD line, signal)",
        )
    )
    registry.register(
        FeatureDefinition(
            name="macd_hist",
            category="momentum",
            version=version,
            parameters={
                "fast": config.macd_fast,
                "slow": config.macd_slow,
                "signal": config.macd_signal,
            },
            lookback=config.macd_slow + config.macd_signal,
            description="MACD histogram: MACD line - Signal line",
        )
    )

    for p in config.roc_periods:
        registry.register(
            FeatureDefinition(
                name=f"roc_{p}",
                category="momentum",
                version=version,
                parameters={"period": p},
                lookback=p,
                description=f"Rate of Change percentage over {p} periods",
            )
        )

    kp = config.stoch_k_period
    registry.register(
        FeatureDefinition(
            name=f"stoch_k_{kp}",
            category="momentum",
            version=version,
            parameters={"k_period": kp, "smooth_k": config.stoch_smooth_k},
            lookback=kp + config.stoch_smooth_k - 1,
            description=f"Stochastic %K oscillator ({kp} periods)",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"stoch_d_{kp}",
            category="momentum",
            version=version,
            parameters={"k_period": kp, "d_period": config.stoch_d_period},
            lookback=kp + config.stoch_smooth_k + config.stoch_d_period - 2,
            description=f"Stochastic %D oscillator ({kp} periods)",
        )
    )
