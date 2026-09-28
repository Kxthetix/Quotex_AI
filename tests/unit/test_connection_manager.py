"""Unit tests for ConnectionManager and reconnect handling."""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import pytest

from data.ingestion.connection_manager import ConnectionManager, ConnectionStatus
from data.ingestion.market_data_interface import MarketDataInterface
from data.ingestion.models import MarketEvent


class MockConnector(MarketDataInterface):
    """Mock connector to simulate disconnections and reconnections deterministically."""

    def __init__(self, fail_first_connect: bool = False, fail_n_times: int = 0) -> None:
        self._provider = "mock"
        self._is_connected = False
        self.connect_count = 0
        self.fail_n_times = fail_n_times
        self.subscribed_trades: list[str] = []
        self.subscribed_candles: list[str] = []

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def connect(self) -> None:
        self.connect_count += 1
        if self.connect_count <= self.fail_n_times:
            raise ConnectionError(f"Simulated network outage (attempt {self.connect_count})")
        self._is_connected = True

    async def disconnect(self) -> None:
        self._is_connected = False

    async def subscribe_trades(self, symbols: list[str]) -> None:
        self.subscribed_trades.extend(symbols)

    async def subscribe_candles(self, symbols: list[str], timeframe: str = "1m") -> None:
        self.subscribed_candles.extend(symbols)

    async def stream_events(self) -> AsyncIterator[MarketEvent]:
        # Empty stream for unit test
        if False:
            yield MarketEvent(
                source="mock",
                symbol="BTC",
                event_type="trade",
                event_time=datetime.now(),
                payload={},
            )

    async def fetch_historical_candles(
        self, symbol: str, timeframe: str = "1m", start=None, end=None, limit=500
    ):
        return []

    def get_health(self) -> dict[str, Any]:
        return {"mock": True}


@pytest.mark.asyncio
async def test_connection_manager_connect_and_subscribe():
    connector = MockConnector()
    manager = ConnectionManager(
        connector=connector,
        symbols=["BTCUSDT", "ETHUSDT"],
    )

    connect_called = []
    manager.on_connect(lambda: connect_called.append(True))

    await manager.connect_and_subscribe()

    assert manager.status == ConnectionStatus.CONNECTED
    assert connector.is_connected
    assert "BTCUSDT" in connector.subscribed_trades
    assert "ETHUSDT" in connector.subscribed_trades
    assert len(connect_called) == 1


@pytest.mark.asyncio
async def test_connection_manager_disconnect_callback():
    connector = MockConnector()
    manager = ConnectionManager(connector=connector, symbols=["BTCUSDT"])
    await manager.connect_and_subscribe()

    disconnect_reasons = []
    manager.on_disconnect(lambda reason: disconnect_reasons.append(reason))

    await manager.disconnect()

    assert manager.status == ConnectionStatus.DISCONNECTED
    assert not connector.is_connected
    assert len(disconnect_reasons) == 1
    assert "Manual disconnect" in disconnect_reasons[0]


@pytest.mark.asyncio
async def test_connection_manager_exponential_backoff_retry():
    # Fails 2 times before succeeding on 3rd attempt
    connector = MockConnector(fail_n_times=2)
    manager = ConnectionManager(
        connector=connector,
        symbols=["BTCUSDT"],
        max_reconnect_attempts=5,
        initial_delay_seconds=0.01,  # Short delay for fast tests
        max_delay_seconds=0.05,
    )

    manager._is_running = True
    success = await manager._handle_reconnect()

    assert success is True
    assert connector.connect_count == 3
    assert manager.status == ConnectionStatus.CONNECTED
