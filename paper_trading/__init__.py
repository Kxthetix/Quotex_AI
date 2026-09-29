"""Phase 7 — Paper Trading package.

Continuous paper-trading environment using live market data with simulated execution.
ABSOLUTE SAFETY INVARIANT: Real-money order execution is strictly impossible.
"""

from paper_trading.engine import PaperTradingConfig, PaperTradingEngine
from paper_trading.execution import (
    PaperExecutionConfig,
    PaperExecutionEngine,
    RealExecutionBlockedError,
)
from paper_trading.market_state import MarketStateManager
from paper_trading.model_inference import LiveModelInferenceEngine, VersionLockConfig
from paper_trading.models import (
    ComponentStatus,
    ConnectionState,
    ExitReason,
    LatencyMeasurement,
    MarketState,
    MonitoringEvent,
    MonitoringEventType,
    MonitoringSeverity,
    PaperFill,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperOrderType,
    PaperPortfolio,
    PaperPosition,
    PaperTradeRecord,
    PaperTradingSession,
    PaperValidationCriteria,
    PredictionRecord,
    PredictionStatus,
    RiskDecision,
    RiskEvaluation,
    SessionStatus,
    StaleDataState,
    SystemState,
)
from paper_trading.monitoring import (
    LatencyTracker,
    ModelAnomalyDetector,
    ModelPerformanceTracker,
    SystemHealthMonitor,
)
from paper_trading.outcome_tracker import OutcomeTracker, QualityBreakdown
from paper_trading.persistence import AsyncPersistenceQueue
from paper_trading.portfolio import PaperPortfolioManager
from paper_trading.prediction_ledger import PredictionLedger
from paper_trading.recovery import SessionRecoveryEngine
from paper_trading.reporting import PaperTradingReporter
from paper_trading.risk_engine import CircuitBreakerState, RiskConfig, RiskEngine
from paper_trading.stale_detector import StaleDataDetector

__all__ = [
    "PaperTradingConfig",
    "PaperTradingEngine",
    "PaperExecutionConfig",
    "PaperExecutionEngine",
    "RealExecutionBlockedError",
    "MarketStateManager",
    "LiveModelInferenceEngine",
    "VersionLockConfig",
    "ComponentStatus",
    "ConnectionState",
    "ExitReason",
    "LatencyMeasurement",
    "MarketState",
    "MonitoringEvent",
    "MonitoringEventType",
    "MonitoringSeverity",
    "PaperFill",
    "PaperOrder",
    "PaperOrderSide",
    "PaperOrderStatus",
    "PaperOrderType",
    "PaperPortfolio",
    "PaperPosition",
    "PaperTradeRecord",
    "PaperTradingSession",
    "PaperValidationCriteria",
    "PredictionRecord",
    "PredictionStatus",
    "RiskDecision",
    "RiskEvaluation",
    "SessionStatus",
    "StaleDataState",
    "SystemState",
    "LatencyTracker",
    "ModelAnomalyDetector",
    "ModelPerformanceTracker",
    "SystemHealthMonitor",
    "OutcomeTracker",
    "QualityBreakdown",
    "AsyncPersistenceQueue",
    "PaperPortfolioManager",
    "PredictionLedger",
    "SessionRecoveryEngine",
    "PaperTradingReporter",
    "CircuitBreakerState",
    "RiskConfig",
    "RiskEngine",
    "StaleDataDetector",
]
