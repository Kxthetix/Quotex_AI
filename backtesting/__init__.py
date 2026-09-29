"""Phase 5 — Backtesting Engine public API."""

from backtesting.engine import BacktestEngine
from backtesting.events import (
    BacktestEvent,
    CandleEvent,
    EventType,
    ExecutionEvent,
    FeatureUpdateEvent,
    OrderEvent,
    PortfolioUpdateEvent,
    PositionUpdateEvent,
    SignalEvent,
)
from backtesting.execution import (
    ExecutionSimulator,
    FeeModel,
    FixedLatencyModel,
    FixedSpreadModel,
    FixedTicksSlippageModel,
    LatencyModel,
    PercentageFeeModel,
    PercentageSlippageModel,
    PercentageSpreadModel,
    SlippageModel,
    SpreadModel,
    ZeroLatencyModel,
)
from backtesting.metrics import MetricsEngine, PerformanceMetrics
from backtesting.models import (
    BacktestConfig,
    BacktestResult,
    EndOfTestAction,
    EquityPoint,
    ExitReason,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    SizingMethod,
    TradeRecord,
)
from backtesting.order_engine import OrderEngine
from backtesting.portfolio import PortfolioEngine
from backtesting.position import PositionEngine
from backtesting.replay import HistoricalReplayEngine, ReplayStats, ReplayWarning
from backtesting.reporting import BacktestReporter
from backtesting.signal_adapter import SignalAdapter
from backtesting.sizing import PositionSizer
from backtesting.walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardResult,
    WindowResult,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestEvent",
    "BacktestReporter",
    "BacktestResult",
    "CandleEvent",
    "EndOfTestAction",
    "EquityPoint",
    "EventType",
    "ExecutionEvent",
    "ExecutionSimulator",
    "ExitReason",
    "FeatureUpdateEvent",
    "FeeModel",
    "Fill",
    "FixedLatencyModel",
    "FixedSpreadModel",
    "FixedTicksSlippageModel",
    "HistoricalReplayEngine",
    "LatencyModel",
    "MetricsEngine",
    "Order",
    "OrderEngine",
    "OrderEvent",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PercentageFeeModel",
    "PercentageSlippageModel",
    "PercentageSpreadModel",
    "PerformanceMetrics",
    "PortfolioEngine",
    "PortfolioUpdateEvent",
    "Position",
    "PositionEngine",
    "PositionSide",
    "PositionSizer",
    "PositionUpdateEvent",
    "ReplayStats",
    "ReplayWarning",
    "SignalAdapter",
    "SignalEvent",
    "SizingMethod",
    "SlippageModel",
    "SpreadModel",
    "TradeRecord",
    "WalkForwardConfig",
    "WalkForwardEngine",
    "WalkForwardResult",
    "WindowResult",
    "ZeroLatencyModel",
]
