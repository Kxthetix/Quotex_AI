"""Unit tests for Storage, Ingestion Run Repository, and Persistence Worker (Test Group I).

Tests compliance with Phase 2 Sections 6, 21, and 22:
- IngestionRunRepository lifecycle (create, update, get, idempotency lookup)
- AsyncPersistenceWorker batch queues (events, candles, trades, incidents, corrections)
- Raw immutable persistence fallback to disk
"""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from data.ingestion.models import (
    Candle,
    CandleCorrectionRecord,
    IngestionRun,
    IngestionRunStatus,
    MarketEvent,
    NormalizedTrade,
)
from data.storage.postgres import AsyncPersistenceWorker, IngestionRunRepository
from data.validation.models import ValidationCategory, ValidationIssue, ValidationSeverity


@pytest.mark.asyncio
async def test_ingestion_run_repository_lifecycle():
    """Verify IngestionRun creation, updating, and idempotency query."""
    repo = IngestionRunRepository(db_manager=None)  # memory-backed
    now = datetime.now(UTC)
    start = now - timedelta(hours=1)
    end = now

    run = IngestionRun(
        provider="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        requested_start=start,
        requested_end=end,
        status=IngestionRunStatus.IN_PROGRESS,
    )
    await repo.create_run(run)

    # Verify retrieval by ID
    retrieved = await repo.get_run(run.run_id)
    assert retrieved is not None
    assert retrieved.symbol == "BTCUSDT"
    assert retrieved.status == IngestionRunStatus.IN_PROGRESS

    # In-progress run is NOT matched by find_completed_run
    found_in_progress = await repo.find_completed_run("binance", "BTCUSDT", "1m", start, end)
    assert found_in_progress is None

    # Update to COMPLETED
    run.status = IngestionRunStatus.COMPLETED
    run.records_received = 60
    run.records_accepted = 60
    run.completion_time = datetime.now(UTC)
    await repo.update_run(run)

    # Now find_completed_run should match
    completed_match = await repo.find_completed_run("binance", "BTCUSDT", "1m", start, end)
    assert completed_match is not None
    assert completed_match.run_id == run.run_id
    assert completed_match.records_accepted == 60


@pytest.mark.asyncio
async def test_async_persistence_worker_queues_and_flush(tmp_path: Path):
    """Verify AsyncPersistenceWorker processes and flushes all item types."""
    worker = AsyncPersistenceWorker(
        db_manager=None,
        raw_fallback_path=tmp_path / "raw",
        flush_interval_seconds=0.1,
    )

    now = datetime.now(UTC)

    # 1. Enqueue raw event
    event = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=now,
        receive_time=now,
        payload={"p": "50000.0"},
    )
    worker.enqueue_event(event)

    # 2. Enqueue candle
    candle = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=50000.0,
        high=50100.0,
        low=49950.0,
        close=50050.0,
        volume=10.0,
        trades_count=50,
        start_time=now,
        end_time=now + timedelta(minutes=1),
        is_closed=True,
    )
    worker.enqueue_candle(candle)

    # 3. Enqueue trade
    trade = NormalizedTrade(
        trade_id="t1",
        source="binance",
        symbol="BTCUSDT",
        price=50000.0,
        quantity=1.0,
        timestamp=now,
    )
    worker.enqueue_trade(trade)

    # 4. Enqueue incident
    incident = ValidationIssue(
        check_name="test_check",
        category=ValidationCategory.INVALID_PRICE,
        severity=ValidationSeverity.ERROR,
        message="Test issue",
        symbol="BTCUSDT",
    )
    worker.enqueue_incident(incident)

    # 5. Enqueue correction
    correction = CandleCorrectionRecord(
        symbol="BTCUSDT",
        timeframe="1m",
        start_time=now,
        original_open=50000.0,
        original_high=50050.0,
        original_low=49950.0,
        original_close=50020.0,
        original_volume=1.0,
        original_trades=10,
        revised_high=50100.0,
        revised_low=49950.0,
        revised_close=50080.0,
        revised_volume=2.0,
        revised_trades=11,
        late_trade_id="late_1",
        late_trade_price=50100.0,
        late_trade_quantity=1.0,
        late_trade_timestamp=now + timedelta(seconds=30),
    )
    worker.enqueue_correction(correction)

    worker.start()
    # Wait for flush loop to execute
    await asyncio.sleep(0.3)
    await worker.stop()

    # Verify fallback files were created on disk
    raw_files = list((tmp_path / "raw").glob("raw_events_*.jsonl"))
    assert len(raw_files) >= 1
