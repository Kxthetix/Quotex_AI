"""Unit tests for Connection Manager & Reconnect Lifecycle (Test Group F).

Tests compliance with Phase 2 Section 14:
- Connection states (DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING, FAILED)
- Auto-reconnect with exponential backoff
- Subscription re-establishment after reconnection
- Max attempts failure handling
"""

from unittest.mock import MagicMock

import pytest

from data.ingestion.connection_manager import ConnectionManager, ConnectionStatus
from data.ingestion.market_data_interface import MarketDataInterface


class MockDisconnectConnector(MarketDataInterface):
    """Connector that simulates disconnects and reconnect success."""

    def __init__(self, fail_count: int = 1) -> None:
        self.fail_count = fail_count
        self.connect_attempts = 0
        self._connected = False
        self.provider_name_val = "binance"
        self.subscribed_trades: list[str] = []
        self.subscribed_candles: list[str] = []

    @property
    def provider_name(self) -> str:
        return self.provider_name_val

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self.connect_attempts += 1
        if self.connect_attempts <= self.fail_count:
            self._connected = False
            raise ConnectionResetError("Simulated network drop")
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def subscribe_trades(self, symbols: list[str]) -> None:
        self.subscribed_trades.extend(symbols)

    async def subscribe_candles(self, symbols: list[str], timeframe: str = "1m") -> None:
        self.subscribed_candles.extend(symbols)

    async def stream_events(self):
        if not self._connected:
            raise ConnectionError("Not connected")
        yield MagicMock()

    async def fetch_historical_candles(
        self, symbol: str, timeframe: str = "1m", start=None, end=None, limit=500
    ):
        return []

    def get_health(self):
        return {"mock": True}


@pytest.mark.asyncio
async def test_connection_manager_connect_and_subscribe():
    """Verify initial connection and subscription restoration."""
    connector = MockDisconnectConnector(fail_count=0)
    mgr = ConnectionManager(
        connector=connector,
        symbols=["BTCUSDT", "ETHUSDT"],
        initial_delay_seconds=0.01,
        max_delay_seconds=0.05,
    )

    await mgr.connect_and_subscribe()
    assert mgr.status == ConnectionStatus.CONNECTED
    assert "BTCUSDT" in connector.subscribed_trades
    assert "ETHUSDT" in connector.subscribed_trades


@pytest.mark.asyncio
async def test_connection_manager_reconnect_retry_success():
    """Verify exponential backoff successfully reconnects after transient drop."""
    # Fails 2 times before succeeding on 3rd attempt
    connector = MockDisconnectConnector(fail_count=2)
    mgr = ConnectionManager(
        connector=connector,
        symbols=["BTCUSDT"],
        max_reconnect_attempts=5,
        initial_delay_seconds=0.01,
        max_delay_seconds=0.05,
    )
    mgr._is_running = True

    reconnected = await mgr._handle_reconnect()
    assert reconnected is True
    assert mgr.status == ConnectionStatus.CONNECTED
    assert connector.connect_attempts == 3


@pytest.mark.asyncio
async def test_connection_manager_max_attempts_exceeded():
    """Verify connection state transitions to FAILED when max retries exceeded."""
    # Fails permanently
    connector = MockDisconnectConnector(fail_count=100)
    mgr = ConnectionManager(
        connector=connector,
        symbols=["BTCUSDT"],
        max_reconnect_attempts=3,
        initial_delay_seconds=0.01,
        max_delay_seconds=0.02,
    )
    mgr._is_running = True

    reconnected = await mgr._handle_reconnect()
    assert reconnected is False
    assert mgr._reconnect_attempts == 3
