"""Data models for Historical Pattern Engine (Phase 4).

Represents pattern search queries, matches, future outcome distributions, and historical sequence windows.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from analysis.regime.models import MarketRegime


class HistoricalMatch(BaseModel):
    """An individual historical market state match with evaluated forward outcome."""

    matched_timestamp: datetime
    similarity_score: float = Field(ge=0.0, le=1.0)
    distance: float = Field(ge=0.0)
    regime: MarketRegime
    forward_return: float
    mfe: float  # Maximum Favorable Excursion
    mae: float  # Maximum Adverse Excursion
    horizon_bars: int
    supporting_features: dict[str, float] = Field(default_factory=dict)


class PatternDistribution(BaseModel):
    """Conditional statistical distribution of forward outcomes from historical matches."""

    match_count: int
    horizon_bars: int
    mean_forward_return: float
    median_forward_return: float
    std_forward_return: float
    positive_outcome_ratio: float
    negative_outcome_ratio: float
    quantiles: dict[str, float] = Field(default_factory=dict)
    mean_mfe: float
    mean_mae: float


class PatternSearchResult(BaseModel):
    """Full pattern engine query output at timestamp T."""

    symbol: str
    timestamp: datetime
    timeframe: str = "1m"
    feature_version: str = "v1.0.0"
    regime: MarketRegime
    matches: list[HistoricalMatch] = Field(default_factory=list)
    distribution: PatternDistribution
    metadata: dict[str, Any] = Field(default_factory=dict)
