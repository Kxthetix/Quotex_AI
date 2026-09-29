"""Core data models for the Signal Engine (Phase 6).

Defines the canonical signal contracts:
1. SignalAction — The three permitted final signal states (BUY, SELL, NO_TRADE).
2. EvidenceType — Categorical evidence stream identifiers.
3. Evidence — A single scored evidence observation from any intelligence source.
4. TradingSignal — The versioned, auditable final signal output.

Invariant: TradingSignal is a decision artifact, NOT a trade order.
It does not imply execution, position sizing, or risk management.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class SignalAction(StrEnum):
    """The three permitted final signal states."""

    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class EvidenceType(StrEnum):
    """Categorical identifiers for intelligence evidence streams."""

    TECHNICAL = "TECHNICAL"
    PATTERN = "PATTERN"
    ML_DIRECTION = "ML_DIRECTION"
    ML_RETURN = "ML_RETURN"
    ML_TRADE_QUALITY = "ML_TRADE_QUALITY"
    REGIME = "REGIME"
    MULTI_TIMEFRAME = "MULTI_TIMEFRAME"
    VOLUME = "VOLUME"
    VOLATILITY = "VOLATILITY"
    RISK_REWARD = "RISK_REWARD"


class Evidence(BaseModel):
    """A single scored evidence observation from an intelligence source.

    Each evidence record carries:
    - A directional vote (+1 = bullish, -1 = bearish, 0 = neutral)
    - A confidence score [0.0, 1.0] representing the source's certainty
    - A weight [0.0, 1.0] representing the evidence type's relative importance in fusion
    """

    evidence_type: EvidenceType
    vote: float = Field(ge=-1.0, le=1.0, description="Directional vote: +1 bullish, -1 bearish, 0 neutral")
    confidence: float = Field(ge=0.0, le=1.0, description="Source certainty in its vote")
    weight: float = Field(ge=0.0, le=1.0, description="Relative importance weight for fusion")
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        """Compute the weighted contribution of this evidence to the fusion score."""
        return self.vote * self.confidence * self.weight


class TradingSignal(BaseModel):
    """The canonical, versioned Signal Engine output.

    Invariant: Purely a structured decision artifact.
    Does NOT imply execution, position sizing, or order submission.
    """

    signal_id: str = Field(default_factory=lambda: f"sig_{uuid4().hex[:10]}")
    symbol: str
    timestamp: datetime
    action: SignalAction
    conviction: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the signal action [0.0, 1.0]",
    )
    expected_return: float = Field(
        default=0.0,
        description="Expected forward return from ML return model",
    )
    expected_volatility: float = Field(
        default=0.0,
        description="Expected forward volatility from ML volatility model",
    )
    risk_reward_ratio: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated reward-to-risk ratio",
    )
    regime: str = Field(
        default="SIDEWAYS",
        description="Current market regime classification",
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Complete evidence audit trail",
    )
    fusion_score: float = Field(
        default=0.0,
        description="Raw weighted fusion score before thresholding",
    )
    signal_version: str = "v1.0.0"
    feature_version: str = "v1.0.0"
    model_version: str = "v1.0.0"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @property
    def evidence_summary(self) -> dict[str, float]:
        """Aggregate weighted scores by evidence type for diagnostics."""
        summary: dict[str, float] = {}
        for e in self.evidence:
            summary[e.evidence_type.value] = summary.get(e.evidence_type.value, 0.0) + e.weighted_score
        return summary
