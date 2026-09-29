"""Market structure feature calculators for Phase 3.

Implements deterministic market structure metrics:
- Support level (rolling minimum low over prior N completed bars)
- Resistance level (rolling maximum high over prior N completed bars)
- Breakout state (close > prior resistance)
- Breakdown state (close < prior support)
- Retest state (measurable proximity to recently broken levels)
- Consolidation state (range compression ratio below threshold)
- Range position (normalized 0.0-1.0 position within current support-resistance channel)
"""

import numpy as np
import pandas as pd

from features.config import StructureFeatureConfig
from features.models import FeatureDefinition
from features.registry import FeatureRegistry


def compute_structure_features(df: pd.DataFrame, config: StructureFeatureConfig) -> pd.DataFrame:
    """Compute all configured market structure features from an OHLCV DataFrame."""
    res = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    lb = config.support_resistance_lookback

    # 1. Support & Resistance levels (Shift by 1 to strictly exclude current bar - Anti-Leakage Invariant)
    support = low.shift(1).rolling(window=lb, min_periods=lb).min()
    resistance = high.shift(1).rolling(window=lb, min_periods=lb).max()

    res[f"support_level_{lb}"] = support
    res[f"resistance_level_{lb}"] = resistance

    # 2. Breakout and Breakdown states
    breakout = np.where(close > resistance, 1.0, 0.0)
    breakdown = np.where(close < support, 1.0, 0.0)

    # In warm-up periods where support/resistance are NaN, states must be NaN
    mask = support.isna() | resistance.isna()
    breakout = np.where(mask, np.nan, breakout)
    breakdown = np.where(mask, np.nan, breakdown)

    res[f"breakout_{lb}"] = breakout
    res[f"breakdown_{lb}"] = breakdown

    # 3. Consolidation state (Range compression ratio)
    channel_width = resistance - support
    mean_price = (resistance + support) / 2.0
    range_compression = channel_width / mean_price.replace(0, np.nan)
    res[f"range_compression_{lb}"] = range_compression

    is_consolidating = np.where(range_compression < config.consolidation_threshold, 1.0, 0.0)
    is_consolidating = np.where(mask, np.nan, is_consolidating)
    res[f"is_consolidating_{lb}"] = is_consolidating

    # 4. Range position (0.0 at support, 1.0 at resistance, clamped)
    pos = (close - support) / channel_width.replace(0, np.nan)
    pos = np.clip(pos, 0.0, 1.0)
    pos = np.where(mask, np.nan, pos)
    res[f"range_position_{lb}"] = pos

    # 5. Retest state: proximity to broken level within tolerance
    # Distance from current close to resistance (for potential breakout retest)
    dist_to_res = (close - resistance).abs() / resistance.replace(0, np.nan)
    dist_to_sup = (close - support).abs() / support.replace(0, np.nan)
    res[f"retest_dist_resistance_{lb}"] = dist_to_res
    res[f"retest_dist_support_{lb}"] = dist_to_sup

    return res


def register_structure_features(
    registry: FeatureRegistry, config: StructureFeatureConfig, version: str = "v1.0.0"
) -> None:
    """Register all market structure feature definitions into the FeatureRegistry."""
    lb = config.support_resistance_lookback

    registry.register(
        FeatureDefinition(
            name=f"support_level_{lb}",
            category="structure",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description=f"Support level: rolling minimum low over prior {lb} completed bars",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"resistance_level_{lb}",
            category="structure",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description=f"Resistance level: rolling maximum high over prior {lb} completed bars",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"breakout_{lb}",
            category="structure",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description=f"Binary indicator: close > resistance_level_{lb}",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"breakdown_{lb}",
            category="structure",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description=f"Binary indicator: close < support_level_{lb}",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"range_compression_{lb}",
            category="structure",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description=f"Range compression ratio: (resistance - support) / mean_price over {lb} bars",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"is_consolidating_{lb}",
            category="structure",
            version=version,
            parameters={"lookback": lb, "threshold": config.consolidation_threshold},
            lookback=lb + 1,
            description="Binary indicator: range_compression below threshold",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"range_position_{lb}",
            category="structure",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description="Normalized relative position of close within support-resistance channel [0.0, 1.0]",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"retest_dist_resistance_{lb}",
            category="structure",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description="Proximity distance percentage from close to resistance level",
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"retest_dist_support_{lb}",
            category="structure",
            version=version,
            parameters={"lookback": lb},
            lookback=lb + 1,
            description="Proximity distance percentage from close to support level",
        )
    )
