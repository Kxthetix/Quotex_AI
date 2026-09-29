"""Provider-neutral Market Data Interface hierarchy.

Defines the contract for streaming live market events and retrieving historical
candles across any exchange or data provider (crypto, equities, futures).
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from data.ingestion.models import Candle, MarketEvent, NormalizedTrade


class WebSocketMarketDataProvider(ABC):
    """Abstract interface for streaming live real-time market data over WebSockets."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the data provider (e.g. 'binance', 'alpaca', 'mock')."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if WebSocket connection is currently active and healthy."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the market data provider."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from the market data provider."""
        pass

    @abstractmethod
    async def subscribe_trades(self, symbols: list[str]) -> None:
        """Subscribe to real-time individual trades for the specified symbols."""
        pass

    @abstractmethod
    async def subscribe_candles(self, symbols: list[str], timeframe: str = "1m") -> None:
        """Subscribe to real-time candlestick streams for the specified symbols and timeframe."""
        pass

    @abstractmethod
    async def stream_events(self) -> AsyncIterator[MarketEvent]:
        """Stream raw incoming MarketEvents asynchronously from the provider."""
        yield  # type: ignore[misc]

    @abstractmethod
    def get_health(self) -> dict[str, Any]:
        """Return operational connection diagnostics and telemetry."""
        pass


class HistoricalMarketDataProvider(ABC):
    """Abstract interface for fetching historical market data via REST/HTTP or files."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the historical data provider."""
        pass

    @abstractmethod
    async def fetch_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """Fetch historical candlesticks via HTTP/REST for backfilling and reconciliation."""
        pass

    async def fetch_historical_trades(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[NormalizedTrade]:
        """Fetch historical trade ticks where supported by provider."""
        return []


class MarketDataInterface(WebSocketMarketDataProvider, HistoricalMarketDataProvider, ABC):
    """Unified provider interface supporting both live streaming and historical downloads."""

    pass
