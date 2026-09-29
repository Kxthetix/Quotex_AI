"""Event schemas and event types for event-driven chronological simulation (Phase 5).

Strict anti-leakage guarantee: Every event carries a precise timestamp. Handlers only
process events whose event_time <= current simulation time.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from analysis.regime.models import RegimeAssessment
from analysis.technical.models import TechnicalAssessment
from backtesting.models import Fill, Order, Position
from data.ingestion.models import Candle
from features.models import FeatureVector


class EventType(StrEnum):
    """Categorical types of simulation events."""

    CANDLE = "CANDLE"
    FEATURE_UPDATE = "FEATURE_UPDATE"
    SIGNAL = "SIGNAL"
    ORDER_SUBMIT = "ORDER_SUBMIT"
    ORDER_EXECUTE = "ORDER_EXECUTE"
    POSITION_UPDATE = "POSITION_UPDATE"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"


@dataclass(kw_only=True)
class BacktestEvent:
    """Base class for all backtesting events."""

    timestamp: datetime
    event_type: EventType


@dataclass(kw_only=True)
class CandleEvent(BacktestEvent):
    """Event emitted when a historical candle completes."""

    candle: Candle
    event_type: EventType = EventType.CANDLE


@dataclass(kw_only=True)
class FeatureUpdateEvent(BacktestEvent):
    """Event emitted after feature engine updates at time T."""

    feature_vector: FeatureVector
    technical_assessment: TechnicalAssessment | None = None
    regime_assessment: RegimeAssessment | None = None
    event_type: EventType = EventType.FEATURE_UPDATE


@dataclass(kw_only=True)
class SignalEvent(BacktestEvent):
    """Standardized signal input event produced by intelligence adapters."""

    symbol: str
    direction_prob: float  # e.g. P(UP)
    expected_return: float  # e.g. +0.005
    expected_volatility: float  # e.g. 0.35
    trade_quality_prob: float  # e.g. 0.65
    regime: str  # e.g. "TREND_UP"
    suggested_action: str  # "ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"
    strength: float = 1.0
    model_version: str = "v1.0.0"
    feature_version: str = "v1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)
    event_type: EventType = EventType.SIGNAL


@dataclass(kw_only=True)
class OrderEvent(BacktestEvent):
    """Event emitted when an order is created and submitted."""

    order: Order
    event_type: EventType = EventType.ORDER_SUBMIT


@dataclass(kw_only=True)
class ExecutionEvent(BacktestEvent):
    """Event emitted when an order is executed (filled or partially filled)."""

    fill: Fill
    order: Order
    event_type: EventType = EventType.ORDER_EXECUTE


@dataclass(kw_only=True)
class PositionUpdateEvent(BacktestEvent):
    """Event emitted when position state changes."""

    position: Position
    event_type: EventType = EventType.POSITION_UPDATE


@dataclass(kw_only=True)
class PortfolioUpdateEvent(BacktestEvent):
    """Event emitted when portfolio balance, equity, or margin updates."""

    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    event_type: EventType = EventType.PORTFOLIO_UPDATE
