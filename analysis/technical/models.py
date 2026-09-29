"""Data models for technical rule evaluations (Phase 4).

Represents structured, objective technical observations derived deterministically
from versioned FeatureVectors without discretionary opinions or trading advice.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TrendState(StrEnum):
    """Categorical classification of price trend alignment."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class MomentumState(StrEnum):
    """Categorical classification of oscillator velocity and extremes."""

    OVERBOUGHT = "OVERBOUGHT"
    OVERSOLD = "OVERSOLD"
    ACCELERATING = "ACCELERATING"
    DECELERATING = "DECELERATING"
    NEUTRAL = "NEUTRAL"


class VolatilityState(StrEnum):
    """Categorical classification of price range dispersion and compression."""

    EXPANDING = "EXPANDING"
    COMPRESSING = "COMPRESSING"
    NORMAL = "NORMAL"
    EXTREME = "EXTREME"


class VolumeState(StrEnum):
    """Categorical classification of trading activity and participation."""

    SPIKE = "SPIKE"
    HIGH_VOLUME = "HIGH_VOLUME"
    LOW_VOLUME = "LOW_VOLUME"
    NORMAL = "NORMAL"


class StructureState(StrEnum):
    """Categorical classification of price action relative to key boundaries."""

    BREAKOUT_UP = "BREAKOUT_UP"
    BREAKDOWN_DOWN = "BREAKDOWN_DOWN"
    CONSOLIDATION = "CONSOLIDATION"
    CHANNEL_RANGE = "CHANNEL_RANGE"


class MultiTimeframeState(StrEnum):
    """Categorical classification of directional consensus across timeframes."""

    ALIGNED_BULLISH = "ALIGNED_BULLISH"
    ALIGNED_BEARISH = "ALIGNED_BEARISH"
    MIXED = "MIXED"
    WARMING_UP = "WARMING_UP"


class TechnicalAssessment(BaseModel):
    """Structured, reproducible technical state assessment at timestamp T.

    Invariant: Purely descriptive market state. Does NOT provide trade signals.
    """

    symbol: str
    timestamp: datetime
    timeframe: str = "1m"
    feature_version: str = "v1.0.0"
    rule_version: str = "v1.0.0"
    trend_state: TrendState
    momentum_state: MomentumState
    volatility_state: VolatilityState
    volume_state: VolumeState
    structure_state: StructureState
    multi_timeframe_state: MultiTimeframeState
    evidence: dict[str, Any] = Field(default_factory=dict)
    is_valid: bool = True
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
