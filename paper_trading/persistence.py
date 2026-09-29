"""Asynchronous Persistence Engine for Phase 7 Paper Trading.

Ensures that live market data processing remains strictly non-blocking and in-memory.
Events, candles, predictions, signals, paper orders, fills, trades, outcomes,
and health telemetry are queued in memory and persisted asynchronously.

CRITICAL INVARIANT:
Database or storage I/O latency or failure NEVER corrupts, stalls, or blocks
the in-memory live execution loop.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monitoring.logging import get_logger

logger = get_logger(__name__)


class AsyncPersistenceQueue:
    """Non-blocking asynchronous queue and worker for persisting live paper-trading records."""

    def __init__(
        self,
        storage_dir: Path | str = "data/paper_trading",
        batch_size: int = 50,
        flush_interval_seconds: float = 1.0,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds

        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._is_running: bool = False
        self._worker_task: asyncio.Task[None] | None = None
        self._total_persisted: int = 0
        self._persistence_errors: int = 0

    def enqueue(self, category: str, item: Any) -> None:
        """Enqueue an item for asynchronous persistence without awaiting DB."""
        payload: dict[str, Any]
        if hasattr(item, "to_dict"):
            payload = item.to_dict()
        elif is_dataclass(item):
            payload = asdict(item)
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            payload = {"data": str(item)}

        payload["_category"] = category
        payload["_enqueued_at"] = datetime.now(UTC).isoformat()

        # Non-blocking put
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("async_persistence_queue_full_dropping_item", category=category)

    async def start(self) -> None:
        """Start the background persistence worker."""
        if not self._is_running:
            self._is_running = True
            self._worker_task = asyncio.create_task(self._persistence_worker())
            logger.info("async_persistence_worker_started", storage_dir=str(self.storage_dir))

    async def stop(self) -> None:
        """Gracefully stop worker and flush all remaining queued records."""
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self.flush()
        logger.info("async_persistence_worker_stopped", total_persisted=self._total_persisted)

    async def _persistence_worker(self) -> None:
        """Continuous background task draining the queue and writing batches to disk."""
        batch: list[dict[str, Any]] = []
        while self._is_running:
            try:
                # Wait for next item or timeout to flush partial batch
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=self.flush_interval_seconds)
                    batch.append(item)
                    self._queue.task_done()
                except asyncio.TimeoutError:
                    pass

                if len(batch) >= self.batch_size or (batch and not self._is_running):
                    await self._write_batch(batch)
                    batch = []

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._persistence_errors += 1
                logger.error("persistence_worker_exception", error=str(e))
                await asyncio.sleep(0.5)

        if batch:
            await self._write_batch(batch)

    async def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        """Write a batch of items segregated by category to JSON Lines files."""
        if not batch:
            return

        categorized: dict[str, list[dict[str, Any]]] = {}
        for item in batch:
            cat = item.get("_category", "general")
            categorized.setdefault(cat, []).append(item)

        today = datetime.now(UTC).strftime("%Y%m%d")
        for cat, items in categorized.items():
            out_file = self.storage_dir / f"{cat}_{today}.jsonl"
            try:
                # Synchronous file write running in thread to avoid event loop stalling
                await asyncio.to_thread(self._append_to_file, out_file, items)
                self._total_persisted += len(items)
            except Exception as e:
                self._persistence_errors += 1
                logger.error("failed_to_persist_batch", category=cat, file=str(out_file), error=str(e))

    @staticmethod
    def _append_to_file(path: Path, items: list[dict[str, Any]]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, default=str) + "\n")

    async def flush(self) -> None:
        """Drain all remaining items in queue synchronously and persist."""
        batch: list[dict[str, Any]] = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._write_batch(batch)
