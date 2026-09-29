"""Price feature calculators for Phase 3.

Implements deterministic price metrics:
- Returns over configurable periods
- Log returns over configurable periods
- Candle body (close - open)
- Upper wick (high - max(open, close))
- Lower wick (min(open, close) - low)
- Candle range (high - low)
- Price gaps (open_t - close_{t-1})
"""

import numpy as np
import pandas as pd

from features.config import PriceFeatureConfig
from features.models import FeatureDefinition
from features.registry import FeatureRegistry


def compute_price_features(df: pd.DataFrame, config: PriceFeatureConfig) -> pd.DataFrame:
    """Compute all configured price features from an OHLCV DataFrame."""
    res = pd.DataFrame(index=df.index)

    close = df["close"]
    open_s = df["open"]
    high = df["high"]
    low = df["low"]

    # 1. Simple returns
    for n in config.returns_periods:
        res[f"return_{n}"] = (close - close.shift(n)) / close.shift(n)

    # 2. Log returns
    for n in config.log_returns_periods:
        res[f"log_return_{n}"] = np.log(close / close.shift(n))

    # 3. Candle body
    if config.include_candle_body:
        res["candle_body"] = close - open_s

    # 4. Upper & lower wicks
    if config.include_wicks:
        res["upper_wick"] = high - np.maximum(open_s, close)
        res["lower_wick"] = np.minimum(open_s, close) - low

    # 5. Range
    if config.include_range:
        res["candle_range"] = high - low

    # 6. Gap from previous close
    if config.include_gap:
        res["gap"] = open_s - close.shift(1)

    return res


def register_price_features(
    registry: FeatureRegistry, config: PriceFeatureConfig, version: str = "v1.0.0"
) -> None:
    """Register all price feature definitions into the FeatureRegistry."""
    for n in config.returns_periods:
        registry.register(
            FeatureDefinition(
                name=f"return_{n}",
                category="price",
                version=version,
                parameters={"period": n},
                lookback=n,
                description=f"Simple percentage return over {n} periods",
            )
        )

    for n in config.log_returns_periods:
        registry.register(
            FeatureDefinition(
                name=f"log_return_{n}",
                category="price",
                version=version,
                parameters={"period": n},
                lookback=n,
                description=f"Natural logarithmic return over {n} periods",
            )
        )

    if config.include_candle_body:
        registry.register(
            FeatureDefinition(
                name="candle_body",
                category="price",
                version=version,
                lookback=1,
                description="Candle body difference (close - open)",
            )
        )

    if config.include_wicks:
        registry.register(
            FeatureDefinition(
                name="upper_wick",
                category="price",
                version=version,
                lookback=1,
                description="Upper wick length: high - max(open, close)",
            )
        )
        registry.register(
            FeatureDefinition(
                name="lower_wick",
                category="price",
                version=version,
                lookback=1,
                description="Lower wick length: min(open, close) - low",
            )
        )

    if config.include_range:
        registry.register(
            FeatureDefinition(
                name="candle_range",
                category="price",
                version=version,
                lookback=1,
                description="Candle total range: high - low",
            )
        )

    if config.include_gap:
        registry.register(
            FeatureDefinition(
                name="gap",
                category="price",
                version=version,
                lookback=2,
                description="Gap between current open and previous close",
            )
        )
