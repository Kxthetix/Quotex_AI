"""Data models for Label Engine (Phase 4).

Defines versioned target labels for statistical and machine-learning models.
Strict invariant: Labels measure 'What happened after T?' and must never enter feature vectors.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LabelDefinition(BaseModel):
    """Metadata specification for a versioned prediction target."""

    name: str
    version: str = "v1.0.0"
    target_type: str  # "binary_classification", "multiclass_classification", "regression"
    horizon_bars: int
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class LabelSet(BaseModel):
    """Set of computed forward labels for a specific bar timestamp."""

    symbol: str
    timestamp: datetime
    label_version: str = "v1.0.0"
    targets: dict[str, float | int | None] = Field(default_factory=dict)
    is_complete: bool = True
