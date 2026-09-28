"""PostgreSQL and TimescaleDB storage engine with asynchronous batch persistence."""

import asyncio
import contextlib
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from configs.config import get_settings
from data.ingestion.models import Candle, MarketEvent
from data.storage.models import Base, CandleModel, RawMarketEventModel
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
                await conn.execute(
                    __import__("sqlalchemy").text(
                        "SELECT create_hypertable('market_data.raw_market_events', 'receive_time', if_not_exists => TRUE);"
                    )
                )
                await conn.execute(
                    __import__("sqlalchemy").text(
                        "SELECT create_hypertable('market_data.candles', 'start_time', if_not_exists => TRUE);"
                    )
                )
            except Exception as e:
                logger.warning("timescale_hypertable_init_skipped", error=str(e))

    async def close(self) -> None:
        """Close database engine."""
        await self.engine.dispose()


class AsyncPersistenceWorker:
    """High-throughput asynchronous batch persistence worker.

    Decouples real-time in-memory event processing from database I/O latency.
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
        self._is_running = False
        self._worker_task: asyncio.Task | None = None

    def enqueue_event(self, event: MarketEvent) -> None:
        """Enqueue raw market event without blocking the live processing loop."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("event_persistence_queue_full_dropping_or_flushing")

    def enqueue_candle(self, candle: Candle) -> None:
        """Enqueue completed candle for persistence."""
        try:
            self._candle_queue.put_nowait(candle)
        except asyncio.QueueFull:
            logger.warning("candle_persistence_queue_full")

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
        """Flush both event and candle queues."""
        await self._flush_events()
        await self._flush_candles()

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

        # Local immutable fallback persistence (append-only JSONL)
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
