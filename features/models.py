"""Core data models for the Feature Engine (Phase 3).

Defines deterministic, versioned feature vectors, feature definitions,
and validity states to prevent data leakage and provide auditability.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FeatureValidity(StrEnum):
    """Validity state for an individual feature or feature vector."""

    VALID = "VALID"
    WARMING_UP = "WARMING_UP"
    INVALID = "INVALID"


class FeatureDefinition(BaseModel):
    """Metadata specification for a registered feature."""

    name: str
    category: str  # price, trend, momentum, volatility, volume, structure, multi_timeframe
    version: str = "v1.0.0"
    parameters: dict[str, Any] = Field(default_factory=dict)
    lookback: int = 1
    timeframe: str = "1m"
    description: str = ""
    output_type: str = "float"  # float, int, bool
    warmup_behavior: str = "returns_none"

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip().lower()


class FeatureVector(BaseModel):
    """Canonical versioned feature vector representing measurable market state at a specific timestamp.

    Invariant: feature values at timestamp T must only be derived from market data available at or before T.
    """

    symbol: str
    timestamp: datetime
    timeframe: str = "1m"
    feature_version: str = "v1.0.0"
    values: dict[str, float | None] = Field(default_factory=dict)
    validity: dict[str, FeatureValidity] = Field(default_factory=dict)
    is_valid: bool = True
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()

    def get(self, feature_name: str, default: float | None = None) -> float | None:
        """Safe getter for feature value."""
        return self.values.get(feature_name.lower(), default)

    def is_feature_valid(self, feature_name: str) -> bool:
        """Check if an individual feature is in the VALID state."""
        return self.validity.get(feature_name.lower()) == FeatureValidity.VALID
