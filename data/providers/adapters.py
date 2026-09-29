"""Market Data Provider Adapters (Section 5).

Provides production adapters and a controllable SyntheticProviderAdapter
supporting realistic trades, candles, and order-book snapshots.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

from data.assets.registry import AssetRegistry
from data.ingestion.models import Candle, MarketEvent, NormalizedTrade
from data.providers.base import MarketDataProvider


class SyntheticProviderAdapter(MarketDataProvider):
    """Controllable test and simulation adapter supporting failure injection and order-book streams."""

    def __init__(
        self,
        name: str = "mock_provider",
        base_price: float = 50000.0,
        spread_bps: float = 2.0,
    ) -> None:
        self._name = name
        self._base_price = base_price
        self._spread_bps = spread_bps
        self._connected = False
        self._subscribed_symbols: set[str] = set()
        self._event_queue: asyncio.Queue[MarketEvent] = asyncio.Queue()
        self._message_count = 0
        self._injected_failure: str | None = None

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        return self._connected and self._injected_failure != "DISCONNECTED"

    def inject_failure(self, failure_type: str | None) -> None:
        """Inject failure mode: 'DISCONNECTED', 'CORRUPT_DATA', 'STALE'."""
        self._injected_failure = failure_type

    async def connect(self) -> None:
        if self._injected_failure == "CONNECTION_REFUSED":
            raise ConnectionRefusedError(f"Connection refused by {self._name}")
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def subscribe_trades(self, symbols: list[str]) -> None:
        for s in symbols:
            self._subscribed_symbols.add(s.upper())

    async def subscribe_candles(self, symbols: list[str], timeframe: str = "1m") -> None:
        for s in symbols:
            self._subscribed_symbols.add(s.upper())

    async def subscribe_orderbook(self, symbols: list[str], depth: int = 20) -> None:
        for s in symbols:
            self._subscribed_symbols.add(s.upper())

    def emit_event(self, event: MarketEvent) -> None:
        """Manually push an event into the stream."""
        self._message_count += 1
        self._event_queue.put_nowait(event)

    async def stream_events(self) -> AsyncIterator[MarketEvent]:
        while self.is_connected:
            try:
                # Wait for next event or timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                if not self.is_connected:
                    break
                continue

    async def fetch_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        now = datetime.now(UTC)
        start_t = start or (now - timedelta(minutes=limit))
        candles = []
        curr = start_t
        p = self._base_price
        for i in range(min(limit, 100)):
            c = Candle(
                symbol=symbol,
                timeframe=timeframe,
                start_time=curr,
                end_time=curr + timedelta(minutes=1),
                open=p,
                high=p + 10.0,
                low=p - 10.0,
                close=p + 5.0,
                volume=10.0 + i,
            )
            candles.append(c)
            curr += timedelta(minutes=1)
            p += 5.0
        return candles

    async def fetch_orderbook_snapshot(self, symbol: str, depth: int = 20) -> dict[str, Any]:
        half_spread = self._base_price * (self._spread_bps / 10000.0) / 2.0
        best_bid = self._base_price - half_spread
        best_ask = self._base_price + half_spread

        bids = [[best_bid - (i * 0.5), 1.0 + (i * 0.2)] for i in range(depth)]
        asks = [[best_ask + (i * 0.5), 1.0 + (i * 0.2)] for i in range(depth)]

        return {
            "symbol": symbol,
            "provider": self._name,
            "timestamp": datetime.now(UTC).isoformat(),
            "sequence_number": 100,
            "bids": bids,
            "asks": asks,
        }

    def get_health(self) -> dict[str, Any]:
        return {
            "provider": self._name,
            "connected": self.is_connected,
            "messages_emitted": self._message_count,
            "subscribed_symbols": list(self._subscribed_symbols),
        }
