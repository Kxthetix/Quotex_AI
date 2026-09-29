"""Unified Market Data Provider Interface (Section 5, 10).

Defines the canonical contract for all exchange and data feed adapters,
including live trades, candlesticks, and order-book depth streaming.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from data.ingestion.models import Candle, MarketEvent, NormalizedTrade


class MarketDataProvider(ABC):
    """Canonical provider contract across crypto, equities, and futures venues."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier (e.g., 'binance', 'coinbase', 'alpaca', 'mock')."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Current connectivity status of the data feed."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection and initialize session with provider."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully terminate connection."""
        pass

    @abstractmethod
    async def subscribe_trades(self, symbols: list[str]) -> None:
        """Subscribe to real-time trade execution feeds."""
        pass

    @abstractmethod
    async def subscribe_candles(self, symbols: list[str], timeframe: str = "1m") -> None:
        """Subscribe to real-time candlestick feeds."""
        pass

    @abstractmethod
    async def subscribe_orderbook(self, symbols: list[str], depth: int = 20) -> None:
        """Subscribe to Level 2 order-book updates and depth snapshots."""
        pass

    @abstractmethod
    async def stream_events(self) -> AsyncIterator[MarketEvent]:
        """Asynchronously yield canonical MarketEvents."""
        yield  # type: ignore[misc]

    @abstractmethod
    async def fetch_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """Retrieve historical candles via REST API."""
        pass

    @abstractmethod
    async def fetch_orderbook_snapshot(self, symbol: str, depth: int = 20) -> dict[str, Any]:
        """Fetch latest Level 2 order-book snapshot."""
        pass

    @abstractmethod
    def get_health(self) -> dict[str, Any]:
        """Return provider connection diagnostics, latency, and message rates."""
        pass
