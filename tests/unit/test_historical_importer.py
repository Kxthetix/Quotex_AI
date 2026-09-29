"""Unit tests for Historical Data Importer and Idempotency (Test Group E).

Tests compliance with Phase 2 Sections 16, 17, 18, 22:
- Historical import flow (raw jsonl -> validate -> parquet -> IngestionRun)
- Idempotency & duplicate detection (repeated imports skip re-work)
- Date range selection
- Partial failures & validation error recording in IngestionRun
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from configs.config import Settings
from data.ingestion.historical_importer import HistoricalDataImporter
from data.ingestion.market_data_interface import HistoricalMarketDataProvider
from data.ingestion.models import Candle, IngestionRunStatus


class MockHistoricalProvider(HistoricalMarketDataProvider):
    """Mock historical provider returning mock candles."""

    def __init__(self, candles: list[Candle]) -> None:
        self.candles = candles

    @property
    def provider_name(self) -> str:
        return "binance"

    async def fetch_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        filtered = self.candles
        if start:
            filtered = [c for c in filtered if c.start_time >= start]
        if end:
            filtered = [c for c in filtered if c.end_time <= end]
        return filtered[:limit]


@pytest.mark.asyncio
async def test_historical_import_end_to_end(tmp_path: Path):
    """Verify standard historical import flow with raw storage and IngestionRun record."""
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    c1 = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=50000.0,
        high=50100.0,
        low=49950.0,
        close=50050.0,
        volume=10.0,
        trades_count=150,
        start_time=base,
        end_time=base + timedelta(minutes=1),
        is_closed=True,
    )
    c2 = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=50050.0,
        high=50200.0,
        low=50000.0,
        close=50180.0,
        volume=15.0,
        trades_count=200,
        start_time=base + timedelta(minutes=1),
        end_time=base + timedelta(minutes=2),
        is_closed=True,
    )

    provider = MockHistoricalProvider([c1, c2])
    test_settings = Settings(
        raw_storage_path=tmp_path / "raw",
        parquet_storage_path=tmp_path / "parquet",
    )

    importer = HistoricalDataImporter(provider=provider, settings=test_settings)

    run = await importer.import_candles(
        symbol="BTCUSDT",
        timeframe="1m",
        start=base,
        end=base + timedelta(minutes=2),
        limit=10,
    )

    # Verify IngestionRun ledger
    assert run.run_id is not None
    assert run.provider == "binance"
    assert run.symbol == "BTCUSDT"
    assert run.timeframe == "1m"
    assert run.status == IngestionRunStatus.COMPLETED
    assert run.records_received == 2
    assert run.records_accepted == 2
    assert run.records_rejected == 0
    assert run.duplicates == 0

    # Verify raw historical jsonl dump created
    raw_files = list((tmp_path / "raw" / "historical").glob("*.jsonl"))
    assert len(raw_files) == 1

    # Verify clean Parquet file created
    parquet_files = list((tmp_path / "parquet" / "BTCUSDT").glob("*.parquet"))
    assert len(parquet_files) == 1


@pytest.mark.asyncio
async def test_historical_import_idempotency(tmp_path: Path):
    """Verify that importing the exact same date range twice does not duplicate work."""
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    c1 = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=50000.0,
        high=50100.0,
        low=49950.0,
        close=50050.0,
        volume=10.0,
        trades_count=150,
        start_time=base,
        end_time=base + timedelta(minutes=1),
        is_closed=True,
    )

    provider = MockHistoricalProvider([c1])
    test_settings = Settings(
        raw_storage_path=tmp_path / "raw",
        parquet_storage_path=tmp_path / "parquet",
    )

    importer = HistoricalDataImporter(provider=provider, settings=test_settings)

    # First import
    run1 = await importer.import_candles(
        symbol="BTCUSDT",
        timeframe="1m",
        start=base,
        end=base + timedelta(minutes=1),
    )
    assert run1.status == IngestionRunStatus.COMPLETED

    # Second identical import (idempotency check)
    run2 = await importer.import_candles(
        symbol="BTCUSDT",
        timeframe="1m",
        start=base,
        end=base + timedelta(minutes=1),
        force_reimport=False,
    )

    # Should return the existing run without creating a new one
    assert run2.run_id == run1.run_id

    # Verify only 1 raw dump and 1 parquet exists (no duplication)
    raw_files = list((tmp_path / "raw" / "historical").glob("*.jsonl"))
    assert len(raw_files) == 1


@pytest.mark.asyncio
async def test_historical_import_with_partial_failure(tmp_path: Path):
    """Verify invalid candles in download stream result in PARTIAL status and error logging."""
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)
    valid_c = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=50000.0,
        high=50100.0,
        low=49950.0,
        close=50050.0,
        volume=10.0,
        trades_count=150,
        start_time=base,
        end_time=base + timedelta(minutes=1),
        is_closed=True,
    )
    # Invalid candle: OHLC inconsistency (high is less than open)
    invalid_c = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=50050.0,
        high=49000.0,  # high < open
        low=48000.0,
        close=48500.0,
        volume=5.0,
        trades_count=200,
        start_time=base + timedelta(minutes=1),
        end_time=base + timedelta(minutes=2),
        is_closed=True,
    )

    provider = MockHistoricalProvider([valid_c, invalid_c])
    test_settings = Settings(
        raw_storage_path=tmp_path / "raw",
        parquet_storage_path=tmp_path / "parquet",
    )

    importer = HistoricalDataImporter(provider=provider, settings=test_settings)
    run = await importer.import_candles(
        symbol="BTCUSDT",
        timeframe="1m",
        start=base,
        end=base + timedelta(minutes=2),
    )

    assert run.status == IngestionRunStatus.PARTIAL
    assert run.records_received == 2
    assert run.records_accepted == 1
    assert run.records_rejected == 1
    assert len(run.errors) == 1
