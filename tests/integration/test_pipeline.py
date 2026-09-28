"""Integration test for the full end-to-end Ingestion Pipeline."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from configs.config import Settings
from data.ingestion.market_data_interface import MarketDataInterface
from data.ingestion.models import Candle, MarketEvent, NormalizedTrade
from data.ingestion.pipeline import IngestionPipeline
from data.storage.postgres import AsyncPersistenceWorker


class StreamingMockConnector(MarketDataInterface):
    """Mock connector that streams predefined events and terminates."""

    def __init__(self, events: list[MarketEvent]) -> None:
        self.events = events
        self._connected = False

    @property
    def provider_name(self) -> str:
        return "binance"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def subscribe_trades(self, symbols: list[str]) -> None:
        pass

    async def subscribe_candles(self, symbols: list[str], timeframe: str = "1m") -> None:
        pass

    async def stream_events(self) -> AsyncIterator[MarketEvent]:
        for e in self.events:
            if not self._connected:
                break
            yield e

    async def fetch_historical_candles(
        self, symbol: str, timeframe: str = "1m", start=None, end=None, limit=500
    ):
        return []

    def get_health(self) -> dict[str, Any]:
        return {"streaming_mock": True}


@pytest.mark.asyncio
async def test_full_pipeline_event_flow(tmp_path: Path):
    # Setup test events across 2 minutes for BTCUSDT
    base_time = datetime(2026, 1, 1, 10, 0, 10, tzinfo=UTC)

    # 3 ticks in minute 10:00
    e1 = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=base_time,
        receive_time=base_time,
        payload={
            "e": "trade",
            "E": int(base_time.timestamp() * 1000),
            "s": "BTCUSDT",
            "t": 101,
            "p": "65000.0",
            "q": "1.0",
            "m": False,
        },
    )
    e2 = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=base_time + timedelta(seconds=20),
        receive_time=base_time + timedelta(seconds=20),
        payload={
            "e": "trade",
            "E": int((base_time + timedelta(seconds=20)).timestamp() * 1000),
            "s": "BTCUSDT",
            "t": 102,
            "p": "65100.0",
            "q": "2.0",
            "m": False,
        },
    )
    # 1 tick in minute 10:01 (triggers closure of 10:00 candle)
    e3 = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=base_time + timedelta(seconds=60),
        receive_time=base_time + timedelta(seconds=60),
        payload={
            "e": "trade",
            "E": int((base_time + timedelta(seconds=60)).timestamp() * 1000),
            "s": "BTCUSDT",
            "t": 103,
            "p": "65050.0",
            "q": "0.5",
            "m": True,
        },
    )

    connector = StreamingMockConnector([e1, e2, e3])

    test_settings = Settings(
        raw_storage_path=tmp_path / "raw",
        cleaned_storage_path=tmp_path / "cleaned",
        parquet_storage_path=tmp_path / "parquet",
    )

    persistence_worker = AsyncPersistenceWorker(
        db_manager=None,
        raw_fallback_path=tmp_path / "raw",
        flush_interval_seconds=0.1,
    )

    pipeline = IngestionPipeline(
        connector=connector,
        symbols=["BTCUSDT"],
        settings=test_settings,
        persistence_worker=persistence_worker,
    )

    closed_candles: list[Candle] = []
    received_trades: list[NormalizedTrade] = []

    pipeline.on_candle_closed(lambda c: closed_candles.append(c))
    pipeline.on_trade(lambda t: received_trades.append(t))

    # Process events directly through pipeline
    await pipeline.state_store.connect()
    persistence_worker.start()

    await pipeline.process_event(e1)
    await pipeline.process_event(e2)
    await pipeline.process_event(e3)

    # Allow worker to flush
    await asyncio.sleep(0.2)

    # 1. Verify trades were normalized & received
    assert len(received_trades) == 3
    assert received_trades[0].price == 65000.0
    assert received_trades[0].side == "buy"
    assert received_trades[1].price == 65100.0
    assert received_trades[2].side == "sell"

    # 2. Verify minute rollover closed candle
    assert len(closed_candles) == 1
    candle = closed_candles[0]
    assert candle.symbol == "BTCUSDT"
    assert candle.open == 65000.0
    assert candle.high == 65100.0
    assert candle.low == 65000.0
    assert candle.close == 65100.0
    assert candle.volume == 3.0
    assert candle.trades_count == 2
    assert candle.is_closed is True

    # 3. Verify in-memory state updated
    latest_trade = await pipeline.state_store.get_latest_trade("BTCUSDT")
    assert latest_trade is not None
    assert latest_trade.price == 65050.0

    # 4. Verify telemetry & health while live
    health = pipeline.get_health()
    assert health.data_quality_healthy is True
    assert health.breaker_tripped is False
    assert health.latency.sample_count == 3
    assert health.latency.p50_ms >= 0.0

    # Stop pipeline
    await pipeline.stop()

    # 5. Verify local fallback file persistence
    raw_files = list((tmp_path / "raw").glob("*.jsonl"))
    assert len(raw_files) > 0
