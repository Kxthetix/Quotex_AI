"""1-Minute Candlestick Builder.

Aggregates incoming normalized trade ticks into clean 1-minute OHLCV candles in memory.
Emits completed closed candles to downstream listeners (persistence, market state, feature engine).
"""

import inspect
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

from data.ingestion.models import Candle, NormalizedTrade
from data.validation.engine import DataQualityEngine
from monitoring.logging import get_logger

logger = get_logger(__name__)


def floor_to_minute(dt: datetime) -> datetime:
    """Floor a datetime to the beginning of its minute (00 seconds, 0 microseconds)."""
    return dt.replace(second=0, microsecond=0)


class CandleBuilder:
    """In-memory 1-minute candle aggregator with event notification."""

    def __init__(
        self,
        timeframe: str = "1m",
        quality_engine: DataQualityEngine | None = None,
    ) -> None:
        self.timeframe = timeframe
        self.quality_engine = quality_engine or DataQualityEngine()

        # In-flight candles: symbol -> Candle
        self._active_candles: dict[str, Candle] = {}

        # Callbacks for closed candles and live candle updates
        self._closed_candle_callbacks: list[Callable[[Candle], Awaitable[None] | None]] = []
        self._update_candle_callbacks: list[Callable[[Candle], Awaitable[None] | None]] = []

    def on_candle_closed(self, callback: Callable[[Candle], Awaitable[None] | None]) -> None:
        """Register a callback invoked whenever a candle is completed and closed."""
        self._closed_candle_callbacks.append(callback)

    def on_candle_update(self, callback: Callable[[Candle], Awaitable[None] | None]) -> None:
        """Register a callback invoked on in-flight candle price/volume updates."""
        self._update_candle_callbacks.append(callback)

    def get_active_candle(self, symbol: str) -> Candle | None:
        """Return the current in-flight candle for a symbol, if any."""
        return self._active_candles.get(symbol.upper())

    async def process_trade(self, trade: NormalizedTrade) -> Candle | None:
        """Process a trade tick. Closes and returns the completed candle if a minute rolled over."""
        symbol = trade.symbol.upper()
        trade_minute = floor_to_minute(trade.timestamp)
        candle_end = trade_minute + timedelta(minutes=1)

        active = self._active_candles.get(symbol)
        closed_candle: Candle | None = None

        if active is None:
            # First tick for this symbol: start new candle
            self._active_candles[symbol] = Candle(
                symbol=symbol,
                timeframe=self.timeframe,
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                volume=trade.quantity,
                start_time=trade_minute,
                end_time=candle_end,
                trades_count=1,
                is_closed=False,
            )
            await self._notify_update(self._active_candles[symbol])
            return None

        # Check if trade belongs to current active candle
        if trade_minute == active.start_time:
            # Update active candle in memory
            active.high = max(active.high, trade.price)
            active.low = min(active.low, trade.price)
            active.close = trade.price
            active.volume += trade.quantity
            active.trades_count += 1
            await self._notify_update(active)
            return None

        elif trade_minute > active.start_time:
            # Minute boundary crossed! Close previous candle and open new one
            active.is_closed = True
            closed_candle = active

            # Validate the completed candle
            validation = self.quality_engine.validate_candle(closed_candle)
            if not validation.is_valid:
                logger.error(
                    "candle_validation_failed",
                    symbol=symbol,
                    issues=[i.message for i in validation.issues],
                )

            # Emit closed candle
            await self._notify_closed(closed_candle)

            # Start new candle for the new trade minute
            self._active_candles[symbol] = Candle(
                symbol=symbol,
                timeframe=self.timeframe,
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                volume=trade.quantity,
                start_time=trade_minute,
                end_time=candle_end,
                trades_count=1,
                is_closed=False,
            )
            await self._notify_update(self._active_candles[symbol])
            return closed_candle

        else:
            # Trade arrived with timestamp older than current active candle
            logger.warning(
                "late_trade_tick_ignored",
                symbol=symbol,
                trade_ts=trade.timestamp.isoformat(),
                active_start=active.start_time.isoformat(),
            )
            return None

    async def flush_all_active(self) -> list[Candle]:
        """Flush and close all in-flight candles (used during shutdown or interval sweeps)."""
        closed_list: list[Candle] = []
        for symbol, active in list(self._active_candles.items()):
            active.is_closed = True
            validation = self.quality_engine.validate_candle(active)
            if validation.is_valid:
                await self._notify_closed(active)
                closed_list.append(active)
            else:
                logger.error("invalid_candle_on_flush", symbol=symbol)
        self._active_candles.clear()
        return closed_list

    async def _notify_closed(self, candle: Candle) -> None:
        """Dispatch closed candle to all registered listeners."""
        for cb in self._closed_candle_callbacks:
            try:
                res = cb(candle)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                logger.exception("candle_closed_callback_error", error=str(e))

    async def _notify_update(self, candle: Candle) -> None:
        """Dispatch live candle update to all registered listeners."""
        for cb in self._update_candle_callbacks:
            try:
                res = cb(candle)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                logger.exception("candle_update_callback_error", error=str(e))
