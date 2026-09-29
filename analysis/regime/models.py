"""Data models for Market Regime Engine (Phase 4).

Represents quantitative market regime classifications derived from objective feature evidence.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MarketRegime(StrEnum):
    """Categorical classification of prevailing market regimes."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    CONSOLIDATION = "CONSOLIDATION"
    REVERSAL = "REVERSAL"


class RegimeAssessment(BaseModel):
    """Versioned market regime classification result at timestamp T."""

    symbol: str
    timestamp: datetime
    timeframe: str = "1m"
    regime: MarketRegime
    confidence: float = Field(ge=0.0, le=1.0)
    regime_version: str = "v1.0.0"
    supporting_features: dict[str, Any] = Field(default_factory=dict)
    active_subregimes: list[MarketRegime] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
