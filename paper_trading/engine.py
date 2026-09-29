"""Master Paper Trading Engine for Phase 7.

Orchestrates the entire real-time paper trading lifecycle:
1. Ingests live MarketEvents or NormalizedTrades
2. Validates integrity via DataQualityEngine (breaker stops bad data)
3. Verifies freshness via StaleDataDetector (stops decisions on stale feeds)
4. Maintains live in-memory MarketState
5. Builds and aggregates live multi-timeframe Candles
6. Computes live features using historical-identical definitions (FeatureEngine)
7. Runs locked production models to generate immutable PredictionRecords
8. Monitors for model anomalies and probability collapse
9. Fuses evidence and generates BUY / SELL / NO-TRADE TradingSignals (SignalEngine)
10. Passes signals through independent Risk Engine (APPROVE / MODIFY / VETO)
11. Simulates realistic execution (PaperExecutionEngine) with spread, slippage, fees, latency
12. Updates simulated PaperPortfolio and records closed trades in PaperTradeRecord ledger
13. Appends evaluation horizons in OutcomeTracker (MFE, MAE, realized return, prediction quality)
14. Dispatches asynchronous persistence without database round-trip blocking
15. Exposes system health, latency percentiles, and auditable monitoring telemetry.

ABSOLUTE SAFETY RULE:
Execution is purely simulated. No real broker API or money order can ever be executed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.ingestion.candle_builder import CandleBuilder
from data.ingestion.market_data_interface import MarketDataInterface
from data.ingestion.models import Candle, MarketEvent, NormalizedTrade
from data.normalization.normalizer import EventNormalizer
from data.validation.engine import DataQualityEngine
from features.engine import FeatureEngine, candles_to_dataframe
from monitoring.logging import get_logger
from paper_trading.execution import PaperExecutionConfig, PaperExecutionEngine
from paper_trading.market_state import MarketStateManager
from paper_trading.model_inference import LiveModelInferenceEngine, VersionLockConfig
from paper_trading.models import (
    LatencyMeasurement,
    MonitoringEvent,
    MonitoringEventType,
    MonitoringSeverity,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperOrderType,
    PaperPortfolio,
    PaperTradingSession,
    PredictionRecord,
    RiskDecision,
    SessionStatus,
    SystemState,
)
from paper_trading.monitoring import SystemHealthMonitor
from paper_trading.outcome_tracker import OutcomeTracker
from paper_trading.persistence import AsyncPersistenceQueue
from paper_trading.portfolio import PaperPortfolioManager
from paper_trading.prediction_ledger import PredictionLedger
from paper_trading.recovery import SessionRecoveryEngine
from paper_trading.reporting import PaperTradingReporter
from paper_trading.risk_engine import RiskConfig, RiskEngine
from paper_trading.stale_detector import StaleDataDetector
from signals.engine import IntelligenceSnapshot, SignalEngine, SignalEngineConfig
from signals.models import SignalAction, TradingSignal

logger = get_logger(__name__)


@dataclass
class PaperTradingConfig:
    """Master configuration for Phase 7 paper trading."""
    session_id: str = ""
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframe: str = "1m"
    initial_capital: float = 10000.0
    evaluation_horizon_seconds: float = 900.0  # 15 minutes
    stale_threshold_seconds: float = 15.0
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    execution_config: PaperExecutionConfig = field(default_factory=PaperExecutionConfig)
    version_lock: VersionLockConfig = field(default_factory=VersionLockConfig)
    storage_dir: str = "data/paper_trading"
    auto_persist: bool = True
    broker_config: dict[str, Any] | None = None  # Tested against to verify real broker isolation


class PaperTradingEngine:
    """Enterprise coordinator for the continuous live paper-trading system."""

    def __init__(self, config: PaperTradingConfig | None = None) -> None:
        self.config = config or PaperTradingConfig()
        self.symbols = [s.upper() for s in self.config.symbols]

        # 1. Enforce Absolute Safety Invariant (Section 2, 48)
        PaperExecutionEngine.enforce_safety_invariants(self.config.broker_config)

        # 2. Session Metadata
        self.session = PaperTradingSession(
            session_id=self.config.session_id or f"pts_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            configuration_version=self.config.version_lock.configuration_version,
            model_versions=self.config.version_lock.approved_models,
            feature_version=self.config.version_lock.feature_version,
            signal_version=self.config.version_lock.signal_version,
            initial_capital=self.config.initial_capital,
            symbols=self.symbols,
        )

        # 3. Core Engine Components
        self.market_state = MarketStateManager(self.symbols)
        self.normalizer = EventNormalizer()
        self.data_quality = DataQualityEngine(stale_data_timeout_seconds=self.config.stale_threshold_seconds)
        self.stale_detector = StaleDataDetector(
            expected_interval_seconds=5.0,
            stale_threshold_seconds=self.config.stale_threshold_seconds,
            symbols=self.symbols,
        )
        self.candle_builder = CandleBuilder(timeframe=self.config.timeframe, quality_engine=self.data_quality)
        self.feature_engine = FeatureEngine()
        self.inference_engine = LiveModelInferenceEngine(version_lock=self.config.version_lock)
        self.signal_engine = SignalEngine(SignalEngineConfig(version=self.config.version_lock.signal_version))
        self.risk_engine = RiskEngine(config=self.config.risk_config)
        self.execution_engine = PaperExecutionEngine(config=self.config.execution_config)
        self.portfolio_mgr = PaperPortfolioManager(initial_capital=self.config.initial_capital)
        self.prediction_ledger = PredictionLedger()
        self.outcome_tracker = OutcomeTracker(
            prediction_ledger=self.prediction_ledger,
            default_horizon_seconds=self.config.evaluation_horizon_seconds,
        )
        self.health_monitor = SystemHealthMonitor()
        self.persistence = AsyncPersistenceQueue(storage_dir=self.config.storage_dir)
        self.recovery = SessionRecoveryEngine(checkpoint_dir=Path(self.config.storage_dir) / "checkpoints")
        self.reporter = PaperTradingReporter(reports_dir=Path(self.config.storage_dir) / "reports")

        # Operational status
        self.is_running: bool = False
        self.connection_status: str = "DISCONNECTED"
        self._candle_cache: dict[str, list[Candle]] = {s: [] for s in self.symbols}
        self.total_signals_generated: int = 0
        self.risk_veto_count: int = 0

        # Wire up candle builder callbacks
        self.candle_builder.on_candle_closed(self._handle_candle_closed)
        self.candle_builder.on_candle_update(self._handle_candle_update)

    async def start(self) -> None:
        """Deterministic startup sequence (Section 36)."""
        logger.info("executing_deterministic_startup_sequence", session_id=self.session.session_id)
        # 1. Initialize async persistence
        if self.config.auto_persist:
            await self.persistence.start()

        # 2. Mark session running
        self.session.status = SessionStatus.RUNNING
        self.is_running = True
        self.connection_status = "CONNECTED"

        event = MonitoringEvent(
            severity=MonitoringSeverity.INFO,
            component="paper_engine",
            event_type=MonitoringEventType.SESSION_START,
            message=f"Paper trading session {self.session.session_id} started successfully",
            metadata={"symbols": self.symbols, "initial_capital": self.config.initial_capital},
        )
        self.health_monitor.record_event(event)
        self.persistence.enqueue("events", event)

    async def stop(self, reason: str = "Normal shutdown") -> dict[str, Any]:
        """Graceful shutdown sequence (Section 37)."""
        logger.info("initiating_graceful_shutdown", reason=reason)
        self.is_running = False
        self.connection_status = "DISCONNECTED"
        self.session.status = SessionStatus.STOPPED
        self.session.shutdown_reason = reason
        self.session.end_time = datetime.now(UTC)

        # Flush any pending prediction evaluations using current mark prices
        current_prices = {s: self.market_state.get_latest_price(s) for s in self.symbols}
        self.outcome_tracker.check_pending_evaluations(current_prices=current_prices)

        # Save recovery checkpoint
        checkpoint_path = self.recovery.save_checkpoint(
            self.session, self.portfolio_mgr, self.execution_engine, self.prediction_ledger
        )

        # Produce end-of-session report
        report = self.reporter.generate_session_report(
            session=self.session,
            portfolio_mgr=self.portfolio_mgr,
            execution_engine=self.execution_engine,
            prediction_ledger=self.prediction_ledger,
            risk_engine=self.risk_engine,
            health_monitor=self.health_monitor,
        )

        # Stop async persistence worker
        if self.config.auto_persist:
            await self.persistence.stop()

        event = MonitoringEvent(
            severity=MonitoringSeverity.INFO,
            component="paper_engine",
            event_type=MonitoringEventType.SESSION_STOP,
            message=f"Paper trading session {self.session.session_id} stopped: {reason}",
            metadata={"checkpoint": str(checkpoint_path)},
        )
        self.health_monitor.record_event(event)

        logger.info("graceful_shutdown_complete", checkpoint=str(checkpoint_path))
        return report

    def process_raw_event(self, event: MarketEvent) -> None:
        """Process incoming raw MarketEvent across all validation and pipeline stages."""
        receive_time = datetime.now(UTC)
        m = LatencyMeasurement(event_time=event.event_time, receive_time=receive_time)

        # 1. Data Quality Validation (Section 7)
        val_result = self.data_quality.validate_raw_event(event)
        if not val_result.is_valid:
            logger.warning("event_failed_data_quality_validation", issues=[str(i) for i in val_result.issues])
            if self.data_quality.is_breaker_tripped:
                logger.error("data_quality_circuit_breaker_active_dropping_event")
                return

        # 2. Normalization
        m.normalize_time = datetime.now(UTC)
        if event.event_type in ("trade", "aggTrade"):
            trade = self.normalizer.normalize_trade(event)
            self.process_trade(trade, measurement=m)
        elif event.event_type == "kline":
            candle = self.normalizer.normalize_kline(event)
            self.process_candle(candle, measurement=m)

    def process_trade(self, trade: NormalizedTrade, measurement: LatencyMeasurement | None = None) -> None:
        """Process a validated normalized trade event in-memory."""
        m = measurement or LatencyMeasurement(event_time=trade.timestamp, receive_time=trade.receive_time)
        sym = trade.symbol.upper()
        now = datetime.now(UTC)

        # 1. Check Data Freshness & Staleness (Section 6)
        self.stale_detector.record_event(sym, trade.timestamp, now)
        is_stale = self.stale_detector.check_staleness(now)

        # 2. Update In-Memory Market State (Section 9)
        self.market_state.update_from_trade(trade)
        self.portfolio_mgr.update_market_price(sym, trade.price, now)

        # 3. Feed Candle Builder (Section 10)
        m.candle_time = datetime.now(UTC)
        self.candle_builder.process_trade(trade)

        # 4. Check for Pending Prediction Horizons (Section 22, 23)
        current_prices = {sym: trade.price}
        evaluated = self.outcome_tracker.check_pending_evaluations(current_time=now, current_prices=current_prices)
        for ep in evaluated:
            self.health_monitor.performance_tracker.record_evaluated_prediction(ep)
            self.persistence.enqueue("evaluations", ep)

        self.health_monitor.total_processed_events += 1

    def process_candle(self, candle: Candle, measurement: LatencyMeasurement | None = None) -> None:
        """Process an externally supplied or historical candle."""
        sym = candle.symbol.upper()
        now = datetime.now(UTC)
        self.market_state.update_closed_candle(candle, timeframe=candle.timeframe)
        self.portfolio_mgr.update_market_price(sym, candle.close, now)
        self.stale_detector.record_candle(sym, candle.end_time)
        self._handle_candle_closed(candle, measurement)

    def _handle_candle_update(self, active_candle: Candle) -> None:
        """Handle in-flight candle price/volume update."""
        self.market_state.update_active_candle(active_candle, timeframe=active_candle.timeframe)

    def _handle_candle_closed(self, candle: Candle, measurement: LatencyMeasurement | None = None) -> None:
        """Executed when a candle completes: Feature -> Prediction -> Signal -> Risk -> Paper Execution."""
        m = measurement or LatencyMeasurement(event_time=candle.end_time, receive_time=datetime.now(UTC))
        sym = candle.symbol.upper()
        now = datetime.now(UTC)

        # Cache candle for features and MFE/MAE
        self._candle_cache[sym].append(candle)
        if len(self._candle_cache[sym]) > 1000:
            self._candle_cache[sym] = self._candle_cache[sym][-1000:]
        self.outcome_tracker.record_candle(candle)

        # 0. Check Breakers (Data Quality, Stale Data, Risk Breaker)
        if self.data_quality.is_breaker_tripped:
            logger.warning("skipping_signal_generation_data_quality_breaker_active")
            return
        if self.stale_detector.is_stale:
            logger.warning("skipping_signal_generation_market_data_stale")
            return
        if self.health_monitor.anomaly_detector.is_abnormal:
            logger.warning("skipping_signal_generation_model_anomaly_active")
            return

        # 1. Feature Calculation (Section 11)
        m.feature_time = datetime.now(UTC)
        df_candles = candles_to_dataframe(self._candle_cache[sym])
        features_df = self.feature_engine.compute_features_dataframe(df_candles)

        # 2. Live Model Inference & Prediction Generation (Section 12, 13)
        m.prediction_time = datetime.now(UTC)
        pred = self.inference_engine.generate_prediction(
            symbol=sym,
            features_df=features_df,
            current_price=candle.close,
            regime="SIDEWAYS",
            timeframe=candle.timeframe,
            horizon_seconds=self.config.evaluation_horizon_seconds,
            timestamp=candle.end_time,
        )
        self.prediction_ledger.record_prediction(pred)
        self.persistence.enqueue("predictions", pred)

        # 3. Live Signal Engine Integration (Section 14)
        m.signal_time = datetime.now(UTC)
        snapshot = IntelligenceSnapshot(
            symbol=sym,
            timestamp=candle.end_time,
            direction_prob_up=0.55 if pred.signal_action == "BUY" else (0.45 if pred.signal_action == "SELL" else 0.50),
            expected_return=pred.expected_return,
            expected_volatility=pred.expected_volatility,
            model_version=self.config.version_lock.model_version,
            feature_version=self.config.version_lock.feature_version,
        )
        signal = self.signal_engine.generate(snapshot)
        self.total_signals_generated += 1
        self.persistence.enqueue("signals", signal)

        # 4. Risk Engine Integration (Section 15, 29)
        m.risk_time = datetime.now(UTC)
        risk_eval = self.risk_engine.evaluate(signal, self.portfolio_mgr.portfolio, current_price=candle.close)
        self.persistence.enqueue("risk_evaluations", risk_eval)

        if risk_eval.decision == RiskDecision.VETO:
            self.risk_veto_count += 1
            logger.info("signal_vetoed_by_risk_engine", signal_id=signal.signal_id, reason=risk_eval.reason)
            return

        # 5. Paper Order & Simulated Execution (Section 16, 17, 18)
        if signal.action == SignalAction.NO_TRADE or risk_eval.approved_quantity <= 0:
            return

        m.order_time = datetime.now(UTC)
        order_side = PaperOrderSide.BUY if signal.action == SignalAction.BUY else PaperOrderSide.SELL
        order = self.execution_engine.create_order(
            signal_id=signal.signal_id,
            symbol=sym,
            side=order_side,
            requested_quantity=risk_eval.approved_quantity,
            order_type=PaperOrderType.MARKET,
            requested_price=candle.close,
            creation_time=now,
        )
        self.execution_engine.submit_order(order.order_id, submitted_time=now)
        self.persistence.enqueue("orders", order)

        # Simulate fill
        fill = self.execution_engine.simulate_execution(
            order=order,
            current_price=candle.close,
            volume=candle.volume,
            timestamp=now,
        )
        m.execution_time = datetime.now(UTC)
        self.health_monitor.latency_tracker.record_measurement(m)

        if fill:
            self.persistence.enqueue("fills", fill)
            # 6. Update Paper Portfolio & Trade Ledger (Section 19, 20)
            closed_trade = self.portfolio_mgr.process_fill(
                fill=fill,
                signal_id=signal.signal_id,
                regime=signal.regime,
                model_version=self.config.version_lock.model_version,
            )
            if closed_trade:
                self.risk_engine.record_trade_pnl(closed_trade.pnl)
                self.persistence.enqueue("trades", closed_trade)
                logger.info(
                    "paper_trade_completed",
                    trade_id=closed_trade.trade_id,
                    symbol=closed_trade.symbol,
                    pnl=round(closed_trade.pnl, 2),
                    return_pct=round(closed_trade.return_pct, 4),
                )

    def get_health_status(self) -> dict[str, Any]:
        """Expose operational telemetry and health status."""
        return self.health_monitor.get_health_report(
            connection_status=self.connection_status,
            is_stale=self.stale_detector.is_stale,
            is_quality_breaker_tripped=self.data_quality.is_breaker_tripped,
            is_risk_breaker_tripped=self.risk_engine.breaker.is_tripped,
        )
