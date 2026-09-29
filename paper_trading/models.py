"""Core data models for Phase 7 — Paper Trading.

Defines all paper-trading entities: sessions, orders, fills, portfolio,
predictions, outcomes, monitoring events, market state, and system health.

ABSOLUTE SAFETY INVARIANT: No model in this module can represent a real-money order.
All order and fill models are explicitly paper-only artifacts.
Even if real broker credentials exist, execution is simulated.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


# ==============================================================================
# System Health States
# ==============================================================================

class SystemState(StrEnum):
    """Overall system health classification."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class ComponentStatus(StrEnum):
    """Individual component health."""
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"
    UNAVAILABLE = "UNAVAILABLE"


# ==============================================================================
# Connection State
# ==============================================================================

class ConnectionState(StrEnum):
    """Lifecycle states of live market data connections."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


# ==============================================================================
# Paper Trading Session
# ==============================================================================

class SessionStatus(StrEnum):
    """Lifecycle of a paper-trading session."""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass
class PaperTradingSession:
    """Identifies a single paper-trading run with full versioning."""
    session_id: str = field(default_factory=lambda: f"pts_{uuid.uuid4().hex[:12]}")
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    configuration_version: str = "v1.0.0"
    model_versions: dict[str, str] = field(default_factory=dict)
    feature_version: str = "v1.0.0"
    signal_version: str = "v1.0.0"
    initial_capital: float = 10000.0
    status: SessionStatus = SessionStatus.INITIALIZING
    shutdown_reason: str = ""
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "configuration_version": self.configuration_version,
            "model_versions": self.model_versions,
            "feature_version": self.feature_version,
            "signal_version": self.signal_version,
            "initial_capital": self.initial_capital,
            "status": str(self.status),
            "shutdown_reason": self.shutdown_reason,
            "symbols": self.symbols,
            "metadata": self.metadata,
        }


# ==============================================================================
# Market State (In-Memory)
# ==============================================================================

@dataclass
class MarketState:
    """In-memory market state for a symbol updated from validated live events."""
    symbol: str = ""
    latest_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    latest_volume: float = 0.0
    market_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    receive_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    data_quality_status: str = "VALID"
    candles: dict[str, Any] = field(default_factory=dict)  # timeframe -> latest Candle
    active_candles: dict[str, Any] = field(default_factory=dict)  # timeframe -> in-flight Candle
    event_count: int = 0

    @property
    def spread(self) -> float:
        if self.bid > 0 and self.ask > 0 and self.ask >= self.bid:
            return self.ask - self.bid
        return 0.0

    @property
    def mid_price(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.latest_price


# ==============================================================================
# Paper Order (NEVER real-money)
# ==============================================================================

class PaperOrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class PaperOrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class PaperOrderStatus(StrEnum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class PaperOrder:
    """Simulated order — NEVER transmitted to a real exchange.

    Safety invariant: This object has no broker API reference, no exchange
    connection, and no method that can place a real order.
    """
    order_id: str = field(default_factory=lambda: f"pord_{uuid.uuid4().hex[:10]}")
    signal_id: str = ""
    symbol: str = ""
    side: PaperOrderSide = PaperOrderSide.BUY
    requested_quantity: float = 0.0
    order_type: PaperOrderType = PaperOrderType.MARKET
    requested_price: float = 0.0
    limit_price: float | None = None
    creation_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    submitted_time: datetime | None = None
    status: PaperOrderStatus = PaperOrderStatus.CREATED
    simulated_execution_time: datetime | None = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    cumulative_fees: float = 0.0
    cumulative_slippage: float = 0.0
    rejection_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": str(self.side),
            "requested_quantity": self.requested_quantity,
            "order_type": str(self.order_type),
            "requested_price": self.requested_price,
            "limit_price": self.limit_price,
            "creation_time": self.creation_time.isoformat() if self.creation_time else None,
            "submitted_time": self.submitted_time.isoformat() if self.submitted_time else None,
            "status": str(self.status),
            "simulated_execution_time": self.simulated_execution_time.isoformat() if self.simulated_execution_time else None,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "cumulative_fees": self.cumulative_fees,
            "cumulative_slippage": self.cumulative_slippage,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }


# ==============================================================================
# Paper Fill
# ==============================================================================

@dataclass
class PaperFill:
    """Record of a simulated execution fill — NEVER a real exchange fill."""
    fill_id: str = field(default_factory=lambda: f"pfill_{uuid.uuid4().hex[:10]}")
    order_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    symbol: str = ""
    side: PaperOrderSide = PaperOrderSide.BUY
    quantity: float = 0.0
    execution_price: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    spread_cost: float = 0.0
    simulated_latency_ms: float = 0.0
    is_partial: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "symbol": self.symbol,
            "side": str(self.side),
            "quantity": self.quantity,
            "execution_price": self.execution_price,
            "fees": self.fees,
            "slippage": self.slippage,
            "spread_cost": self.spread_cost,
            "simulated_latency_ms": self.simulated_latency_ms,
            "is_partial": self.is_partial,
        }


# ==============================================================================
# Paper Portfolio
# ==============================================================================

@dataclass
class PaperPosition:
    """A single simulated position — NEVER connected to a real account."""
    symbol: str = ""
    side: str = "FLAT"  # LONG, SHORT, FLAT
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    entry_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))
    realized_pnl: float = 0.0
    cumulative_fees: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0.0 or self.side == "FLAT":
            return 0.0
        if self.side == "LONG":
            return self.quantity * (self.current_price - self.avg_entry_price)
        return self.quantity * (self.avg_entry_price - self.current_price)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "market_value": self.market_value,
            "cumulative_fees": self.cumulative_fees,
        }


@dataclass
class PaperPortfolio:
    """Simulated portfolio — NEVER connected to a real brokerage account."""
    initial_capital: float = 10000.0
    cash: float = 10000.0
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    total_fees: float = 0.0
    total_slippage: float = 0.0
    peak_equity: float = 10000.0
    cumulative_realized_pnl: float = 0.0

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values() if p.side != "FLAT")

    @property
    def total_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values() if p.side != "FLAT")

    @property
    def total_equity(self) -> float:
        """Total equity = Cash + Total Market Value (asset accounting)."""
        market_val = sum(
            p.market_value if p.side == "LONG" else -p.market_value
            for p in self.positions.values() if p.side != "FLAT"
        )
        return self.cash + market_val

    @property
    def equity(self) -> float:
        return self.total_equity

    @property
    def drawdown(self) -> float:
        eq = self.total_equity
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - eq) / self.peak_equity)

    def update_peak(self) -> None:
        eq = self.total_equity
        if eq > self.peak_equity:
            self.peak_equity = eq

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "equity": self.total_equity,
            "total_exposure": self.total_exposure,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "realized_pnl": self.cumulative_realized_pnl,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "peak_equity": self.peak_equity,
            "drawdown": self.drawdown,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
        }


# ==============================================================================
# Paper Trade Ledger Record
# ==============================================================================

class ExitReason(StrEnum):
    SIGNAL = "SIGNAL"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TIME_LIMIT = "TIME_LIMIT"
    RISK_LIMIT = "RISK_LIMIT"
    SESSION_END = "SESSION_END"
    MANUAL = "MANUAL"


@dataclass
class PaperTradeRecord:
    """Completed trade cycle in paper trading."""
    trade_id: str = field(default_factory=lambda: f"ptrd_{uuid.uuid4().hex[:10]}")
    signal_id: str = ""
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    entry_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    exit_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    holding_period_seconds: float = 0.0
    exit_reason: ExitReason = ExitReason.SIGNAL
    regime: str = "SIDEWAYS"
    model_version: str = "v1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "signal_id": self.signal_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "fees": self.fees,
            "slippage": self.slippage,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "holding_period_seconds": self.holding_period_seconds,
            "exit_reason": str(self.exit_reason),
            "regime": self.regime,
            "model_version": self.model_version,
        }


# ==============================================================================
# Prediction Ledger
# ==============================================================================

class PredictionStatus(StrEnum):
    PENDING = "PENDING"
    EVALUATED = "EVALUATED"
    EXPIRED = "EXPIRED"


@dataclass
class PredictionRecord:
    """Immutable prediction record — original fields never overwritten.

    Evaluation data is appended, not mutated.
    """
    prediction_id: str = field(default_factory=lambda: f"pred_{uuid.uuid4().hex[:10]}")
    symbol: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    timeframe: str = "1m"
    model_version: str = "v1.0.0"
    feature_version: str = "v1.0.0"
    signal_version: str = "v1.0.0"
    signal_action: str = "NO_TRADE"
    confidence: float = 0.0
    regime: str = "SIDEWAYS"
    setup: str = "STANDARD"
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    expected_duration_bars: int = 15
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    evaluation_horizon_seconds: float = 900.0  # 15 min default
    # Evaluation fields (appended later, NOT overwriting original prediction)
    status: PredictionStatus = PredictionStatus.PENDING
    evaluation_time: datetime | None = None
    realized_return: float | None = None
    mfe: float | None = None  # Maximum Favorable Excursion
    mae: float | None = None  # Maximum Adverse Excursion
    actual_outcome: str | None = None  # "CORRECT", "INCORRECT", "NEUTRAL"
    prediction_error: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "timeframe": self.timeframe,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "signal_version": self.signal_version,
            "signal_action": self.signal_action,
            "confidence": self.confidence,
            "regime": self.regime,
            "setup": self.setup,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "expected_duration_bars": self.expected_duration_bars,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "evaluation_horizon_seconds": self.evaluation_horizon_seconds,
            "status": str(self.status),
            "evaluation_time": self.evaluation_time.isoformat() if self.evaluation_time else None,
            "realized_return": self.realized_return,
            "mfe": self.mfe,
            "mae": self.mae,
            "actual_outcome": self.actual_outcome,
            "prediction_error": self.prediction_error,
            "metadata": self.metadata,
        }


# ==============================================================================
# Risk Engine Models
# ==============================================================================

class RiskDecision(StrEnum):
    APPROVE = "APPROVE"
    MODIFY = "MODIFY"
    VETO = "VETO"


@dataclass
class RiskEvaluation:
    """Result of risk engine evaluation for a signal."""
    decision: RiskDecision = RiskDecision.APPROVE
    original_quantity: float = 0.0
    approved_quantity: float = 0.0
    reason: str = ""
    checks: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": str(self.decision),
            "original_quantity": self.original_quantity,
            "approved_quantity": self.approved_quantity,
            "reason": self.reason,
            "checks": self.checks,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# ==============================================================================
# Monitoring Events
# ==============================================================================

class MonitoringSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class MonitoringEventType(StrEnum):
    CONNECTION = "CONNECTION"
    DISCONNECTION = "DISCONNECTION"
    RECONNECT = "RECONNECT"
    STALE_DATA = "STALE_DATA"
    DATA_QUALITY = "DATA_QUALITY"
    INVALID_DATA = "INVALID_DATA"
    MODEL_ANOMALY = "MODEL_ANOMALY"
    RISK_VETO = "RISK_VETO"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    PAPER_ORDER = "PAPER_ORDER"
    PAPER_FILL = "PAPER_FILL"
    PREDICTION_EVALUATED = "PREDICTION_EVALUATED"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    SESSION_START = "SESSION_START"
    SESSION_STOP = "SESSION_STOP"
    MODEL_LOADED = "MODEL_LOADED"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"


@dataclass
class MonitoringEvent:
    """Structured monitoring event for audit and alerting."""
    event_id: str = field(default_factory=lambda: f"mon_{uuid.uuid4().hex[:10]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    severity: MonitoringSeverity = MonitoringSeverity.INFO
    component: str = ""
    event_type: MonitoringEventType = MonitoringEventType.CONNECTION
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "severity": str(self.severity),
            "component": self.component,
            "event_type": str(self.event_type),
            "message": self.message,
            "metadata": self.metadata,
        }


# ==============================================================================
# Latency & Stale Data State
# ==============================================================================

@dataclass
class StaleDataState:
    """Tracks freshness of live incoming market data."""
    is_stale: bool = False
    last_event_time: datetime | None = None
    last_receive_time: datetime | None = None
    last_candle_time: datetime | None = None
    expected_interval_seconds: float = 5.0
    stale_threshold_seconds: float = 15.0
    reason: str = ""


@dataclass
class LatencyMeasurement:
    """Latency timings for an individual processed event."""
    event_time: datetime | None = None
    receive_time: datetime | None = None
    normalize_time: datetime | None = None
    candle_time: datetime | None = None
    feature_time: datetime | None = None
    prediction_time: datetime | None = None
    signal_time: datetime | None = None
    risk_time: datetime | None = None
    order_time: datetime | None = None
    execution_time: datetime | None = None

    @property
    def total_pipeline_ms(self) -> float:
        if self.receive_time and self.execution_time:
            return (self.execution_time - self.receive_time).total_seconds() * 1000.0
        return 0.0


# ==============================================================================
# Validation Period Configuration
# ==============================================================================

@dataclass
class PaperValidationCriteria:
    """Configurable paper-validation period requirements."""
    minimum_paper_days: int = 7
    minimum_predictions: int = 100
    minimum_trades: int = 20
    minimum_data_coverage_pct: float = 80.0
