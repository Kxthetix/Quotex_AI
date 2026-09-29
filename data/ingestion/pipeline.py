"""Data Ingestion Pipeline.

Orchestrates the entire live market data flow:
ConnectionManager -> EventValidator -> Normalizer -> Async Persistence -> CandleBuilder -> Health & Latency Tracking.
"""

import asyncio
import contextlib
import inspect
import time
from collections.abc import Awaitable, Callable

from configs.config import Settings, get_settings
from data.ingestion.binance_connector import BinanceWebSocketConnector
from data.ingestion.candle_builder import CandleBuilder
from data.ingestion.connection_manager import ConnectionManager
from data.ingestion.market_data_interface import MarketDataInterface
from data.ingestion.models import Candle, CandleCorrectionRecord, MarketEvent, NormalizedTrade
from data.normalization.normalizer import EventNormalizer
from data.storage.postgres import AsyncPersistenceWorker, DatabaseManager
from data.storage.redis_store import MarketStateStore
from data.validation.engine import DataQualityEngine
from data.validation.models import DataQualityStats, ValidationIssue
from monitoring.health import LatencyTracker, SystemHealthMonitor, SystemHealthReport
from monitoring.logging import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """Master Ingestion Pipeline coordinating real-time market data components."""

    def __init__(
        self,
        connector: MarketDataInterface | None = None,
        symbols: list[str] | None = None,
        settings: Settings | None = None,
        quality_engine: DataQualityEngine | None = None,
        normalizer: EventNormalizer | None = None,
        candle_builder: CandleBuilder | None = None,
        state_store: MarketStateStore | None = None,
        persistence_worker: AsyncPersistenceWorker | None = None,
        db_manager: DatabaseManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.symbols = [s.upper() for s in (symbols or self.settings.market_symbols)]

        self.connector = connector or BinanceWebSocketConnector(
            ws_url=self.settings.ws_base_url,
            rest_url=self.settings.rest_base_url,
        )

        self.connection_manager = ConnectionManager(
            connector=self.connector,
            symbols=self.symbols,
            max_reconnect_attempts=self.settings.max_reconnect_attempts,
            initial_delay_seconds=self.settings.initial_reconnect_delay_seconds,
            max_delay_seconds=self.settings.max_reconnect_delay_seconds,
        )

        self.quality_engine = quality_engine or DataQualityEngine(
            stale_data_timeout_seconds=self.settings.stale_data_timeout_seconds
        )
        self.normalizer = normalizer or EventNormalizer(provider=self.connector.provider_name)
        self.candle_builder = candle_builder or CandleBuilder(
            timeframe="1m", quality_engine=self.quality_engine
        )

        self.state_store = state_store or MarketStateStore()
        self.db_manager = db_manager
        self.persistence_worker = persistence_worker or AsyncPersistenceWorker(
            db_manager=self.db_manager
        )

        self.latency_tracker = LatencyTracker()
        self.health_monitor = SystemHealthMonitor(latency_tracker=self.latency_tracker)

        self._is_running = False
        self._pipeline_task: asyncio.Task | None = None
        self._trade_listeners: list[Callable[[NormalizedTrade], Awaitable[None] | None]] = []

        # Wire candle builder to persistence and state store
        self.candle_builder.on_candle_closed(self._handle_candle_closed)
        self.candle_builder.on_candle_update(self._handle_candle_update)
        self.candle_builder.on_candle_corrected(self._handle_candle_corrected)

        # Wire connection lifecycle hooks
        self.connection_manager.on_disconnect(self._handle_disconnect)
        self.connection_manager.on_connect(self._handle_connect)

    def on_trade(self, callback: Callable[[NormalizedTrade], Awaitable[None] | None]) -> None:
        """Register listener for valid normalized trades."""
        self._trade_listeners.append(callback)

    def on_candle_closed(self, callback: Callable[[Candle], Awaitable[None] | None]) -> None:
        """Register listener for completed candles."""
        self.candle_builder.on_candle_closed(callback)

    def on_candle_corrected(
        self, callback: Callable[[CandleCorrectionRecord], Awaitable[None] | None]
    ) -> None:
        """Register listener for auditable candle corrections."""
        self.candle_builder.on_candle_corrected(callback)

    async def start(self) -> None:
        """Start the ingestion pipeline."""
        if self._is_running:
            return

        self._is_running = True
        logger.info(
            "starting_ingestion_pipeline",
            symbols=self.symbols,
            provider=self.connector.provider_name,
        )

        await self.state_store.connect()
        self.persistence_worker.start()

        self._pipeline_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Gracefully stop the pipeline, disconnect, and flush queues."""
        self._is_running = False
        if self._pipeline_task:
            self._pipeline_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pipeline_task

        await self.connection_manager.disconnect()
        await self.persistence_worker.stop()
        await self.state_store.close()
        logger.info("ingestion_pipeline_stopped")

    async def _run_loop(self) -> None:
        """Core streaming loop processing events with low-latency in-memory path."""
        try:
            async for raw_event in self.connection_manager.run_with_auto_reconnect():
                if not self._is_running:
                    break
                await self.process_event(raw_event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("pipeline_fatal_error", error=str(e))
            self.quality_engine.trip_breaker(f"Pipeline crashed: {e}")

    async def process_event(self, event: MarketEvent) -> None:
        """Process an individual market event through validation, normalization, and aggregation."""
        start_time = time.perf_counter()

        # 1. Raw event validation
        val_result = self.quality_engine.validate_raw_event(event)
        if not val_result.is_valid:
            logger.warning(
                "invalid_raw_market_event_dropped",
                symbol=event.symbol,
                issues=[i.message for i in val_result.issues],
            )
            for issue in val_result.issues:
                self.persistence_worker.enqueue_incident(issue)
            return

        # 2. Async persistence enqueue (non-blocking)
        self.persistence_worker.enqueue_event(event)

        # 3. Normalization
        try:
            if event.event_type in ("trade", "aggTrade") or "p" in event.payload:
                trade = self.normalizer.normalize_trade(event)
                trade_val = self.quality_engine.validate_trade(trade)
                if not trade_val.is_valid:
                    logger.warning(
                        "invalid_trade_tick_dropped",
                        symbol=trade.symbol,
                        issues=[i.message for i in trade_val.issues],
                    )
                    for issue in trade_val.issues:
                        self.persistence_worker.enqueue_incident(issue)
                    return

                # Clean persistence enqueue
                self.persistence_worker.enqueue_trade(trade)

                # Update in-memory state
                await self.state_store.set_latest_trade(trade)

                # Feed tick to candle builder
                await self.candle_builder.process_trade(trade)

                # Notify trade listeners
                for cb in self._trade_listeners:
                    try:
                        res = cb(trade)
                        if inspect.isawaitable(res):
                            await res
                    except Exception as e:
                        logger.exception("trade_listener_error", error=str(e))

            elif event.event_type == "kline" or "k" in event.payload:
                candle = self.normalizer.normalize_kline(event)
                candle_val = self.quality_engine.validate_candle(candle)
                if not candle_val.is_valid:
                    for issue in candle_val.issues:
                        self.persistence_worker.enqueue_incident(issue)
                elif candle.is_closed:
                    self.persistence_worker.enqueue_candle(candle)
                    await self.state_store.set_latest_candle(candle)

        except Exception as e:
            logger.exception("event_processing_error", error=str(e), symbol=event.symbol)

        # 4. Measure end-to-end processing latency
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.latency_tracker.record_latency_ms(elapsed_ms)

    async def _handle_candle_closed(self, candle: Candle) -> None:
        """Callback when CandleBuilder closes a 1-minute candle."""
        self.persistence_worker.enqueue_candle(candle)
        await self.state_store.set_latest_candle(candle)
        logger.info(
            "candle_closed",
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            start=candle.start_time.isoformat(),
            close=candle.close,
            volume=round(candle.volume, 4),
            trades=candle.trades_count,
        )

    async def _handle_candle_update(self, candle: Candle) -> None:
        """Callback on in-flight candle price/volume update."""
        await self.state_store.set_latest_candle(candle)

    async def _handle_candle_corrected(self, correction: CandleCorrectionRecord) -> None:
        """Callback when CandleBuilder corrects a finalized candle due to an auditable late trade."""
        self.persistence_worker.enqueue_correction(correction)
        logger.warning(
            "candle_corrected_audit_recorded",
            symbol=correction.symbol,
            timeframe=correction.timeframe,
            candle_start=correction.candle_start_time.isoformat(),
            correction_id=correction.correction_id,
            reason=correction.reason,
            original_ohlcv=[
                correction.original_open,
                correction.original_high,
                correction.original_low,
                correction.original_close,
                correction.original_volume,
            ],
            revised_ohlcv=[
                correction.revised_open,
                correction.revised_high,
                correction.revised_low,
                correction.revised_close,
                correction.revised_volume,
            ],
        )

    async def _handle_disconnect(self, reason: str) -> None:
        """Safety handler: trip circuit breaker or pause decision making on disconnect."""
        logger.warning("disconnect_safety_triggered", reason=reason)
        self.quality_engine.trip_breaker(f"Market disconnected: {reason}")

    async def _handle_connect(self) -> None:
        """Handler when market connection is established or restored."""
        logger.info("market_connection_established_or_restored")
        for symbol in self.symbols:
            self.quality_engine.record_reconnection(symbol)

        # Reset breaker if it was tripped solely due to market disconnect
        if (
            self.quality_engine.is_breaker_tripped
            and self.quality_engine.breaker_reason
            and "Market disconnected" in self.quality_engine.breaker_reason
        ):
            self.quality_engine.reset_breaker()
            logger.info("data_quality_breaker_reset_after_reconnection")

    def get_data_quality_stats(self, symbol: str | None = None) -> DataQualityStats:
        """Retrieve aggregated data quality metrics."""
        return self.quality_engine.get_stats(symbol)

    def get_recent_incidents(self) -> list[ValidationIssue]:
        """Retrieve recent data quality incidents."""
        return self.quality_engine.get_recent_incidents()

    def get_health(self) -> SystemHealthReport:
        """Generate structured health report."""
        conn_health = self.connector.get_health()
        conn_health.update(self.connection_manager.get_health())
        return self.health_monitor.generate_report(
            is_connected=self.connector.is_connected,
            is_breaker_tripped=self.quality_engine.is_breaker_tripped,
            breaker_reason=self.quality_engine.breaker_reason,
            connector_health=conn_health,
        )
