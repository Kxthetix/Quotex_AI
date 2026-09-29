"""PostgreSQL and TimescaleDB storage engine with asynchronous batch persistence and run tracking."""

import asyncio
import contextlib
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from configs.config import get_settings
from data.ingestion.models import (
    Candle,
    CandleCorrectionRecord,
    IngestionRun,
    IngestionRunStatus,
    MarketEvent,
    NormalizedTrade,
)
from data.storage.models import (
    Base,
    CandleCorrectionModel,
    CandleModel,
    DataQualityIncidentModel,
    IngestionRunModel,
    NormalizedTradeModel,
    RawMarketEventModel,
)
from data.validation.models import ValidationIssue
from monitoring.logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages PostgreSQL/TimescaleDB connection pool and schema setup."""

    def __init__(self, dsn: str | None = None) -> None:
        self.settings = get_settings()
        self.dsn = dsn or self.settings.postgres_dsn
        self.engine = create_async_engine(
            self.dsn,
            pool_size=self.settings.postgres_pool_size,
            max_overflow=5,
            echo=False,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_schema(self) -> None:
        """Create tables and schemas if they don't exist."""
        async with self.engine.begin() as conn:
            # Create schema first
            await conn.execute(
                __import__("sqlalchemy").text("CREATE SCHEMA IF NOT EXISTS market_data;")
            )
            await conn.run_sync(Base.metadata.create_all)
            # Try to convert to hypertables if TimescaleDB extension is available
            try:
                await conn.execute(
                    __import__("sqlalchemy").text(
                        "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
                    )
                )
                for table, time_col in [
                    ("market_data.raw_market_events", "receive_time"),
                    ("market_data.candles", "start_time"),
                    ("market_data.normalized_trades", "timestamp"),
                ]:
                    await conn.execute(
                        __import__("sqlalchemy").text(
                            f"SELECT create_hypertable('{table}', '{time_col}', if_not_exists => TRUE);"
                        )
                    )
            except Exception as e:
                logger.warning("timescale_hypertable_init_skipped", error=str(e))

    async def close(self) -> None:
        """Close database engine."""
        await self.engine.dispose()


class IngestionRunRepository:
    """Repository for managing IngestionRun records with in-memory fallback."""

    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        self.db_manager = db_manager
        self._memory_runs: dict[str, IngestionRun] = {}

    async def create_run(self, run: IngestionRun) -> IngestionRun:
        """Persist a new ingestion run."""
        self._memory_runs[run.run_id] = run.model_copy()

        if self.db_manager:
            try:
                async with self.db_manager.session_factory() as session:
                    model = IngestionRunModel(
                        run_id=run.run_id,
                        provider=run.provider,
                        symbol=run.symbol,
                        timeframe=run.timeframe,
                        requested_start=run.requested_start,
                        requested_end=run.requested_end,
                        actual_start=run.actual_start,
                        actual_end=run.actual_end,
                        status=run.status.value,
                        records_received=run.records_received,
                        records_accepted=run.records_accepted,
                        records_rejected=run.records_rejected,
                        duplicates=run.duplicates,
                        errors=run.errors,
                        created_at=run.created_at,
                        completion_time=run.completion_time,
                    )
                    session.add(model)
                    await session.commit()
            except Exception as e:
                logger.warning("db_create_run_failed_using_memory", error=str(e), run_id=run.run_id)

        return run

    async def update_run(self, run: IngestionRun) -> IngestionRun:
        """Update an existing ingestion run."""
        self._memory_runs[run.run_id] = run.model_copy()

        if self.db_manager:
            try:
                async with self.db_manager.session_factory() as session:
                    result = await session.execute(
                        select(IngestionRunModel).where(IngestionRunModel.run_id == run.run_id)
                    )
                    model = result.scalar_one_or_none()
                    if model:
                        model.actual_start = run.actual_start
                        model.actual_end = run.actual_end
                        model.status = run.status.value
                        model.records_received = run.records_received
                        model.records_accepted = run.records_accepted
                        model.records_rejected = run.records_rejected
                        model.duplicates = run.duplicates
                        model.errors = run.errors
                        model.completion_time = run.completion_time or datetime.now(UTC)
                        await session.commit()
            except Exception as e:
                logger.warning("db_update_run_failed_using_memory", error=str(e), run_id=run.run_id)

        return run

    async def get_run(self, run_id: str) -> IngestionRun | None:
        """Fetch an ingestion run by ID."""
        if run_id in self._memory_runs:
            return self._memory_runs[run_id]

        if self.db_manager:
            try:
                async with self.db_manager.session_factory() as session:
                    result = await session.execute(
                        select(IngestionRunModel).where(IngestionRunModel.run_id == run_id)
                    )
                    model = result.scalar_one_or_none()
                    if model:
                        return IngestionRun(
                            run_id=model.run_id,
                            provider=model.provider,
                            symbol=model.symbol,
                            timeframe=model.timeframe,
                            requested_start=model.requested_start,
                            requested_end=model.requested_end,
                            actual_start=model.actual_start,
                            actual_end=model.actual_end,
                            status=IngestionRunStatus(model.status),
                            records_received=model.records_received,
                            records_accepted=model.records_accepted,
                            records_rejected=model.records_rejected,
                            duplicates=model.duplicates,
                            errors=model.errors or [],
                            created_at=model.created_at,
                            completion_time=model.completion_time,
                        )
            except Exception as e:
                logger.warning("db_get_run_failed", error=str(e))

        return None

    async def find_completed_run(
        self, provider: str, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> IngestionRun | None:
        """Check if an identical backfill run has already been successfully executed."""
        for run in self._memory_runs.values():
            if (
                run.provider == provider
                and run.symbol == symbol.upper()
                and run.timeframe == timeframe
                and run.requested_start == start
                and run.requested_end == end
                and run.status == IngestionRunStatus.COMPLETED
            ):
                return run

        if self.db_manager:
            try:
                async with self.db_manager.session_factory() as session:
                    stmt = select(IngestionRunModel).where(
                        IngestionRunModel.provider == provider,
                        IngestionRunModel.symbol == symbol.upper(),
                        IngestionRunModel.timeframe == timeframe,
                        IngestionRunModel.requested_start == start,
                        IngestionRunModel.requested_end == end,
                        IngestionRunModel.status == IngestionRunStatus.COMPLETED.value,
                    )
                    result = await session.execute(stmt)
                    model = result.scalar_one_or_none()
                    if model:
                        return IngestionRun(
                            run_id=model.run_id,
                            provider=model.provider,
                            symbol=model.symbol,
                            timeframe=model.timeframe,
                            requested_start=model.requested_start,
                            requested_end=model.requested_end,
                            actual_start=model.actual_start,
                            actual_end=model.actual_end,
                            status=IngestionRunStatus(model.status),
                            records_received=model.records_received,
                            records_accepted=model.records_accepted,
                            records_rejected=model.records_rejected,
                            duplicates=model.duplicates,
                            errors=model.errors or [],
                            created_at=model.created_at,
                            completion_time=model.completion_time,
                        )
            except Exception as e:
                logger.warning("db_find_completed_run_failed", error=str(e))

        return None


class AsyncPersistenceWorker:
    """High-throughput asynchronous batch persistence worker for Phase 2.

    Decouples real-time in-memory event processing from database I/O latency.
    Batches raw events, candles, normalized trades, incidents, and corrections.
    """

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        batch_size: int = 100,
        flush_interval_seconds: float = 1.0,
        raw_fallback_path: Path | None = None,
    ) -> None:
        self.settings = get_settings()
        self.db_manager = db_manager
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.raw_fallback_path = raw_fallback_path or self.settings.raw_storage_path
        self.raw_fallback_path.mkdir(parents=True, exist_ok=True)

        self._event_queue: asyncio.Queue[MarketEvent] = asyncio.Queue(maxsize=50_000)
        self._candle_queue: asyncio.Queue[Candle] = asyncio.Queue(maxsize=10_000)
        self._trade_queue: asyncio.Queue[NormalizedTrade] = asyncio.Queue(maxsize=50_000)
        self._incident_queue: asyncio.Queue[ValidationIssue] = asyncio.Queue(maxsize=10_000)
        self._correction_queue: asyncio.Queue[CandleCorrectionRecord] = asyncio.Queue(maxsize=5_000)

        self._is_running = False
        self._worker_task: asyncio.Task | None = None

    def enqueue_event(self, event: MarketEvent) -> None:
        """Enqueue raw market event without blocking the live processing loop."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("event_persistence_queue_full")

    def enqueue_candle(self, candle: Candle) -> None:
        """Enqueue completed candle for persistence."""
        try:
            self._candle_queue.put_nowait(candle)
        except asyncio.QueueFull:
            logger.warning("candle_persistence_queue_full")

    def enqueue_trade(self, trade: NormalizedTrade) -> None:
        """Enqueue normalized trade tick for clean persistence."""
        try:
            self._trade_queue.put_nowait(trade)
        except asyncio.QueueFull:
            logger.warning("trade_persistence_queue_full")

    def enqueue_incident(self, incident: ValidationIssue) -> None:
        """Enqueue data quality violation incident for auditable persistence."""
        try:
            self._incident_queue.put_nowait(incident)
        except asyncio.QueueFull:
            logger.warning("incident_persistence_queue_full")

    def enqueue_correction(self, correction: CandleCorrectionRecord) -> None:
        """Enqueue candle adjustment audit record."""
        try:
            self._correction_queue.put_nowait(correction)
        except asyncio.QueueFull:
            logger.warning("correction_persistence_queue_full")

    def start(self) -> None:
        """Start background persistence worker."""
        if not self._is_running:
            self._is_running = True
            self._worker_task = asyncio.create_task(self._run_loop())
            logger.info("async_persistence_worker_started")

    async def stop(self) -> None:
        """Stop background worker and flush remaining buffers."""
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        # Final flush
        await self._flush_all()
        logger.info("async_persistence_worker_stopped")

    async def _run_loop(self) -> None:
        """Continuous batching loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self.flush_interval_seconds)
                await self._flush_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("persistence_worker_error", error=str(e))

    async def _flush_all(self) -> None:
        """Flush all queues."""
        await self._flush_events()
        await self._flush_candles()
        await self._flush_trades()
        await self._flush_incidents()
        await self._flush_corrections()

    async def _flush_events(self) -> None:
        """Batch insert raw market events."""
        events: list[MarketEvent] = []
        while not self._event_queue.empty() and len(events) < self.batch_size:
            events.append(self._event_queue.get_nowait())

        if not events:
            return

        if self.db_manager:
            try:
                async with self.db_manager.session_factory() as session:
                    models = [
                        RawMarketEventModel(
                            source=e.source,
                            symbol=e.symbol,
                            event_type=e.event_type,
                            event_time=e.event_time,
                            receive_time=e.receive_time,
                            payload=e.payload,
                        )
                        for e in events
                    ]
                    session.add_all(models)
                    await session.commit()
                return
            except Exception as e:
                logger.warning(
                    "db_persistence_failed_writing_to_fallback", error=str(e), count=len(events)
                )

        self._write_events_to_disk(events)

    async def _flush_candles(self) -> None:
        """Batch upsert closed candles."""
        candles: list[Candle] = []
        while not self._candle_queue.empty() and len(candles) < self.batch_size:
            candles.append(self._candle_queue.get_nowait())

        if not candles:
            return

        if self.db_manager:
            try:
                async with self.db_manager.session_factory() as session:
                    for c in candles:
                        stmt = (
                            insert(CandleModel)
                            .values(
                                symbol=c.symbol,
                                timeframe=c.timeframe,
                                start_time=c.start_time,
                                end_time=c.end_time,
                                open=c.open,
                                high=c.high,
                                low=c.low,
                                close=c.close,
                                volume=c.volume,
                                trades_count=c.trades_count,
                                is_closed=c.is_closed,
                            )
                            .on_conflict_do_update(
                                index_elements=["symbol", "timeframe", "start_time"],
                                set_={
                                    "high": c.high,
                                    "low": c.low,
                                    "close": c.close,
                                    "volume": c.volume,
                                    "trades_count": c.trades_count,
                                    "is_closed": c.is_closed,
                                },
                            )
                        )
                        await session.execute(stmt)
                    await session.commit()
                return
            except Exception as e:
                logger.warning(
                    "candle_db_persistence_failed_writing_to_fallback",
                    error=str(e),
                    count=len(candles),
                )

        self._write_candles_to_disk(candles)

    async def _flush_trades(self) -> None:
        """Batch insert normalized trade ticks."""
        trades: list[NormalizedTrade] = []
        while not self._trade_queue.empty() and len(trades) < self.batch_size:
            trades.append(self._trade_queue.get_nowait())

        if not trades:
            return

        if self.db_manager:
            try:
                async with self.db_manager.session_factory() as session:
                    models = [
                        NormalizedTradeModel(
                            trade_id=t.trade_id,
                            source=t.source,
                            symbol=t.symbol,
                            price=t.price,
                            quantity=t.quantity,
                            side=t.side,
                            timestamp=t.timestamp,
                            receive_time=t.receive_time,
                        )
                        for t in trades
                    ]
                    session.add_all(models)
                    await session.commit()
                return
            except Exception as e:
                logger.warning("trade_db_persistence_failed", error=str(e))

        # Local fallback disk write
        cleaned_path = self.settings.cleaned_storage_path
        today_str = datetime.now(UTC).strftime("%Y-%m-%d")
        file_path = cleaned_path / f"trades_{today_str}.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            for t in trades:
                f.write(t.model_dump_json() + "\n")

    async def _flush_incidents(self) -> None:
        """Batch insert data quality violation incidents."""
        incidents: list[ValidationIssue] = []
        while not self._incident_queue.empty() and len(incidents) < self.batch_size:
            incidents.append(self._incident_queue.get_nowait())

        if not incidents:
            return

        if self.db_manager:
            try:
                async with self.db_manager.session_factory() as session:
                    models = [
                        DataQualityIncidentModel(
                            check_name=i.check_name,
                            category=i.category.value,
                            severity=i.severity.value,
                            symbol=i.symbol,
                            message=i.message,
                            details=i.details,
                            recorded_at=i.timestamp,
                        )
                        for i in incidents
                    ]
                    session.add_all(models)
                    await session.commit()
            except Exception as e:
                logger.warning("incident_db_persistence_failed", error=str(e))

    async def _flush_corrections(self) -> None:
        """Batch insert candle adjustments."""
        corrections: list[CandleCorrectionRecord] = []
        while not self._correction_queue.empty() and len(corrections) < self.batch_size:
            corrections.append(self._correction_queue.get_nowait())

        if not corrections:
            return

        if self.db_manager:
            try:
                async with self.db_manager.session_factory() as session:
                    models = [
                        CandleCorrectionModel(
                            correction_id=c.correction_id,
                            symbol=c.symbol,
                            timeframe=c.timeframe,
                            start_time=c.start_time,
                            original_open=c.original_open,
                            original_high=c.original_high,
                            original_low=c.original_low,
                            original_close=c.original_close,
                            original_volume=c.original_volume,
                            original_trades=c.original_trades,
                            revised_high=c.revised_high,
                            revised_low=c.revised_low,
                            revised_close=c.revised_close,
                            revised_volume=c.revised_volume,
                            revised_trades=c.revised_trades,
                            late_trade_id=c.late_trade_id,
                            late_trade_price=c.late_trade_price,
                            late_trade_quantity=c.late_trade_quantity,
                            late_trade_timestamp=c.late_trade_timestamp,
                            detected_at=c.detected_at,
                        )
                        for c in corrections
                    ]
                    session.add_all(models)
                    await session.commit()
            except Exception as e:
                logger.warning("correction_db_persistence_failed", error=str(e))

    def _write_events_to_disk(self, events: list[MarketEvent]) -> None:
        """Write raw events to immutable date-partitioned JSONL files."""
        today_str = datetime.now(UTC).strftime("%Y-%m-%d")
        file_path = self.raw_fallback_path / f"raw_events_{today_str}.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            for e in events:
                f.write(e.model_dump_json() + "\n")

    def _write_candles_to_disk(self, candles: list[Candle]) -> None:
        """Write candles to date-partitioned JSONL files."""
        cleaned_path = self.settings.cleaned_storage_path
        cleaned_path.mkdir(parents=True, exist_ok=True)
        today_str = datetime.now(UTC).strftime("%Y-%m-%d")
        file_path = cleaned_path / f"candles_{today_str}.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            for c in candles:
                f.write(c.model_dump_json() + "\n")
