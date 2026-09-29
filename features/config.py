"""Configuration models and loader for the Feature Engine (Phase 3).

Centralizes all lookbacks, parameters, and timeframes to guarantee
that feature calculations are deterministic and version-locked without hardcoded constants.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PriceFeatureConfig(BaseModel):
    """Configuration for price-based features."""

    returns_periods: list[int] = Field(default_factory=lambda: [1, 3, 5, 15, 30, 60])
    log_returns_periods: list[int] = Field(default_factory=lambda: [1, 5, 15])
    include_candle_body: bool = True
    include_wicks: bool = True
    include_range: bool = True
    include_gap: bool = True


class TrendFeatureConfig(BaseModel):
    """Configuration for trend-following features."""

    ema_periods: list[int] = Field(default_factory=lambda: [9, 20, 50, 100, 200])
    sma_periods: list[int] = Field(default_factory=lambda: [20, 50, 100, 200])
    adx_period: int = 14
    slope_periods: list[int] = Field(default_factory=lambda: [10, 20])
    structure_lookback: int = 20  # Lookback for HH, LH, HL, LL


class MomentumFeatureConfig(BaseModel):
    """Configuration for momentum and oscillator features."""

    rsi_periods: list[int] = Field(default_factory=lambda: [7, 14, 21])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    roc_periods: list[int] = Field(default_factory=lambda: [9, 14, 21])
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth_k: int = 3


class VolatilityFeatureConfig(BaseModel):
    """Configuration for volatility and dispersion features."""

    atr_periods: list[int] = Field(default_factory=lambda: [7, 14])
    rolling_vol_periods: list[int] = Field(default_factory=lambda: [10, 20, 30])
    historical_vol_periods: list[int] = Field(default_factory=lambda: [20, 60])


class VolumeFeatureConfig(BaseModel):
    """Configuration for volume-based features."""

    relative_vol_periods: list[int] = Field(default_factory=lambda: [10, 20, 50])
    volume_ma_periods: list[int] = Field(default_factory=lambda: [20, 50])
    volume_spike_threshold: float = 2.0  # multiple of rolling volume MA
    volume_trend_periods: list[int] = Field(default_factory=lambda: [10, 20])


class StructureFeatureConfig(BaseModel):
    """Configuration for market structure features."""

    support_resistance_lookback: int = 20
    breakout_lookback: int = 20
    consolidation_lookback: int = 20
    consolidation_threshold: float = 0.02  # max range-to-price ratio for consolidation


class MultiTimeframeConfig(BaseModel):
    """Configuration for multi-timeframe feature aggregation."""

    enabled: bool = True
    enabled_timeframes: list[str] = Field(default_factory=lambda: ["5m", "15m", "1h", "4h", "1d"])
    higher_tf_ema_periods: list[int] = Field(default_factory=lambda: [20, 50])
    higher_tf_rsi_period: int = 14
    higher_tf_atr_period: int = 14


class FeatureEngineConfig(BaseModel):
    """Master configuration for the Feature Engine."""

    version: str = "v1.0.0"
    base_timeframe: str = "1m"
    price: PriceFeatureConfig = Field(default_factory=PriceFeatureConfig)
    trend: TrendFeatureConfig = Field(default_factory=TrendFeatureConfig)
    momentum: MomentumFeatureConfig = Field(default_factory=MomentumFeatureConfig)
    volatility: VolatilityFeatureConfig = Field(default_factory=VolatilityFeatureConfig)
    volume: VolumeFeatureConfig = Field(default_factory=VolumeFeatureConfig)
    structure: StructureFeatureConfig = Field(default_factory=StructureFeatureConfig)
    multi_timeframe: MultiTimeframeConfig = Field(default_factory=MultiTimeframeConfig)

    @classmethod
    def load_from_yaml(cls, path: Path | str) -> "FeatureEngineConfig":
        """Load configuration from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Feature config file not found: {file_path}")
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def save_to_yaml(self, path: Path | str) -> None:
        """Serialize configuration to a YAML file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


def get_default_feature_config() -> FeatureEngineConfig:
    """Return default v1.0.0 feature configuration."""
    return FeatureEngineConfig()
