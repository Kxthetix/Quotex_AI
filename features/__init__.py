"""Feature Engine package for Trading Intelligence System (Phase 3)."""

from features.config import (
    FeatureEngineConfig,
    MomentumFeatureConfig,
    MultiTimeframeConfig,
    PriceFeatureConfig,
    StructureFeatureConfig,
    TrendFeatureConfig,
    VolatilityFeatureConfig,
    VolumeFeatureConfig,
    get_default_feature_config,
)
from features.engine import FeatureEngine, candles_to_dataframe
from features.models import FeatureDefinition, FeatureValidity, FeatureVector
from features.multi_timeframe.aligner import MultiTimeframeAligner
from features.multi_timeframe.resampler import (
    parse_timeframe_to_timedelta,
    resample_candles_to_timeframe,
)
from features.registry import FeatureRegistry, default_registry

__all__ = [
    "FeatureDefinition",
    "FeatureEngine",
    "FeatureEngineConfig",
    "FeatureRegistry",
    "FeatureValidity",
    "FeatureVector",
    "MomentumFeatureConfig",
    "MultiTimeframeAligner",
    "MultiTimeframeConfig",
    "PriceFeatureConfig",
    "StructureFeatureConfig",
    "TrendFeatureConfig",
    "VolatilityFeatureConfig",
    "VolumeFeatureConfig",
    "candles_to_dataframe",
    "default_registry",
    "get_default_feature_config",
    "parse_timeframe_to_timedelta",
    "resample_candles_to_timeframe",
]
