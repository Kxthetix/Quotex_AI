"""Connection Manager and Reconnection Engine.

Manages market data connection lifecycles, exponential backoff with jitter,
automatic re-subscription, and safety notifications upon disconnections.
"""

import asyncio
import inspect
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from data.ingestion.market_data_interface import MarketDataInterface
from data.ingestion.models import MarketEvent
from monitoring.logging import get_logger

logger = get_logger(__name__)


class ConnectionStatus(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


class ConnectionManager:
    """Enterprise connection manager with resilient auto-reconnect and circuit notifications."""

    def __init__(
        self,
        connector: MarketDataInterface,
        symbols: list[str],
        max_reconnect_attempts: int = 10,
        initial_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
        backoff_multiplier: float = 2.0,
    ) -> None:
        self.connector = connector
        self.symbols = [s.upper() for s in symbols]
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay_seconds = initial_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.backoff_multiplier = backoff_multiplier

        self.status: ConnectionStatus = ConnectionStatus.DISCONNECTED
        self._reconnect_attempts: int = 0
        self._total_disconnections: int = 0
        self._last_connected_at: datetime | None = None
        self._last_disconnected_at: datetime | None = None
        self._is_running: bool = False

        # Callbacks for safety integration
        self._disconnect_callbacks: list[Callable[[str], Awaitable[None] | None]] = []
        self._connect_callbacks: list[Callable[[], Awaitable[None] | None]] = []

    def on_disconnect(self, callback: Callable[[str], Awaitable[None] | None]) -> None:
        """Register callback invoked whenever disconnection occurs (halts live decision-making)."""
        self._disconnect_callbacks.append(callback)

    def on_connect(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        """Register callback invoked whenever connection is established."""
        self._connect_callbacks.append(callback)

    async def connect_and_subscribe(self) -> None:
        """Connect to provider and subscribe to symbol streams."""
        self.status = ConnectionStatus.CONNECTING
        logger.info(
            "connecting_market_data", provider=self.connector.provider_name, symbols=self.symbols
        )
        try:
            await self.connector.connect()
            await self.connector.subscribe_trades(self.symbols)
            await self.connector.subscribe_candles(self.symbols, timeframe="1m")

            self.status = ConnectionStatus.CONNECTED
            self._reconnect_attempts = 0
            self._last_connected_at = datetime.now(UTC)
            logger.info(
                "market_data_connected_and_subscribed", provider=self.connector.provider_name
            )
            await self._notify_connect()
        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error("connection_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Explicitly disconnect."""
        self._is_running = False
        self.status = ConnectionStatus.DISCONNECTED
        self._last_disconnected_at = datetime.now(UTC)
        await self.connector.disconnect()
        await self._notify_disconnect("Manual disconnect requested")

    async def run_with_auto_reconnect(self) -> AsyncIterator[MarketEvent]:
        """Async generator yielding MarketEvents with automatic exponential-backoff reconnection."""
        self._is_running = True

        while self._is_running:
            try:
                if not self.connector.is_connected:
                    await self.connect_and_subscribe()

                async for event in self.connector.stream_events():
                    if not self._is_running:
                        break
                    yield event

            except asyncio.CancelledError:
                logger.info("stream_cancelled_by_caller")
                self._is_running = False
                break
            except Exception as e:
                self._total_disconnections += 1
                self.status = ConnectionStatus.RECONNECTING
                self._last_disconnected_at = datetime.now(UTC)
                reason = f"Stream interrupted: {e}"
                logger.warning(
                    "market_stream_disconnected", error=str(e), attempts=self._reconnect_attempts
                )

                # Safety Rule: halt live decision-making upon disconnect
                await self._notify_disconnect(reason)

                # Attempt reconnect with exponential backoff
                reconnected = await self._handle_reconnect()
                if not reconnected:
                    self.status = ConnectionStatus.FAILED
                    self._is_running = False
                    logger.critical("max_reconnect_attempts_exceeded_halting")
                    raise ConnectionError("Failed to reconnect after maximum retry attempts") from e

    async def _handle_reconnect(self) -> bool:
        """Execute exponential backoff with jitter retry loop."""
        while self._reconnect_attempts < self.max_reconnect_attempts and self._is_running:
            self._reconnect_attempts += 1
            # Exponential backoff calculation
            base_delay = min(
                self.max_delay_seconds,
                self.initial_delay_seconds
                * (self.backoff_multiplier ** (self._reconnect_attempts - 1)),
            )
            # Add full jitter (0 to 100% of base_delay)
            jitter = random.uniform(0, 0.5 * base_delay)
            sleep_duration = base_delay + jitter

            logger.info(
                "attempting_reconnect",
                attempt=self._reconnect_attempts,
                max_attempts=self.max_reconnect_attempts,
                wait_seconds=round(sleep_duration, 2),
            )
            await asyncio.sleep(sleep_duration)

            try:
                await self.connect_and_subscribe()
                logger.info("reconnection_successful", attempt=self._reconnect_attempts)
                return True
            except Exception as e:
                logger.warning(
                    "reconnection_attempt_failed", attempt=self._reconnect_attempts, error=str(e)
                )

        return False

    async def _notify_disconnect(self, reason: str) -> None:
        """Notify all callbacks that connection was severed."""
        for cb in self._disconnect_callbacks:
            try:
                res = cb(reason)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                logger.exception("disconnect_callback_error", error=str(e))

    async def _notify_connect(self) -> None:
        """Notify all callbacks that connection is restored."""
        for cb in self._connect_callbacks:
            try:
                res = cb()
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                logger.exception("connect_callback_error", error=str(e))

    def get_health(self) -> dict[str, Any]:
        """Return connection diagnostic metrics."""
        return {
            "status": self.status.value,
            "is_connected": self.connector.is_connected,
            "reconnect_attempts": self._reconnect_attempts,
            "total_disconnections": self._total_disconnections,
            "last_connected_at": self._last_connected_at.isoformat()
            if self._last_connected_at
            else None,
            "last_disconnected_at": self._last_disconnected_at.isoformat()
            if self._last_disconnected_at
            else None,
            "symbols": self.symbols,
        }
