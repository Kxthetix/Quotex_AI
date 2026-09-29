"""1-Minute Candlestick Builder with auditable late-event correction policy.

Aggregates incoming normalized trade ticks into clean 1-minute OHLCV candles in memory.
Handles minute boundary transitions, multi-symbol concurrency, and out-of-order/late arrivals
via an auditable correction mechanism that prevents silent data mutation.
"""

import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from data.ingestion.models import Candle, CandleCorrectionRecord, NormalizedTrade
from data.validation.engine import DataQualityEngine
from monitoring.logging import get_logger

logger = get_logger(__name__)


def floor_to_minute(dt: datetime) -> datetime:
    """Floor a datetime to the beginning of its minute (00 seconds, 0 microseconds)."""
    return dt.replace(second=0, microsecond=0)


class CandleBuilder:
    """In-memory 1-minute candle aggregator with event notification and auditable corrections."""

    def __init__(
        self,
        timeframe: str = "1m",
        quality_engine: DataQualityEngine | None = None,
        late_tolerance_minutes: int = 5,
    ) -> None:
        self.timeframe = timeframe
        self.quality_engine = quality_engine or DataQualityEngine()
        self.late_tolerance_minutes = late_tolerance_minutes

        # In-flight candles: symbol -> Candle
        self._active_candles: dict[str, Candle] = {}

        # Auditable cache of recent closed candles for late corrections: key (symbol:start_iso) -> Candle
        self._recent_closed_candles: dict[str, Candle] = {}
        self._recent_closed_keys: deque[str] = deque(maxlen=500)

        # Callbacks
        self._closed_candle_callbacks: list[Callable[[Candle], Awaitable[None] | None]] = []
        self._update_candle_callbacks: list[Callable[[Candle], Awaitable[None] | None]] = []
        self._correction_callbacks: list[
            Callable[[CandleCorrectionRecord, Candle], Awaitable[None] | None]
        ] = []

    def on_candle_closed(self, callback: Callable[[Candle], Awaitable[None] | None]) -> None:
        """Register a callback invoked whenever a candle is completed and closed."""
        self._closed_candle_callbacks.append(callback)

    def on_candle_update(self, callback: Callable[[Candle], Awaitable[None] | None]) -> None:
        """Register a callback invoked on in-flight candle price/volume updates."""
        self._update_candle_callbacks.append(callback)

    def on_candle_corrected(
        self, callback: Callable[[CandleCorrectionRecord, Candle], Awaitable[None] | None]
    ) -> None:
        """Register a callback invoked when a late event audits and adjusts a closed candle."""
        self._correction_callbacks.append(callback)

    def get_active_candle(self, symbol: str) -> Candle | None:
        """Return the current in-flight candle for a symbol, if any."""
        return self._active_candles.get(symbol.upper())

    def get_recent_closed_candle(self, symbol: str, start_time: datetime) -> Candle | None:
        """Return a recently closed candle by symbol and start time."""
        key = f"{symbol.upper()}:{start_time.isoformat()}"
        return self._recent_closed_candles.get(key)

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
            # Minute boundary crossed! Finalize previous candle
            active.is_closed = True
            closed_candle = active

            # Cache closed candle for late correction audits
            closed_key = f"{symbol}:{closed_candle.start_time.isoformat()}"
            self._recent_closed_candles[closed_key] = closed_candle.model_copy()
            self._recent_closed_keys.append(closed_key)

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
            # Late / out-of-order trade arrived with timestamp older than current active candle
            await self._handle_late_trade(trade, active)
            return None

    async def _handle_late_trade(self, trade: NormalizedTrade, active: Candle) -> None:
        """Auditable late-event handling policy: adjust recent closed candle with audit record."""
        symbol = trade.symbol.upper()
        trade_minute = floor_to_minute(trade.timestamp)
        key = f"{symbol}:{trade_minute.isoformat()}"

        now = datetime.now(UTC)
        elapsed_seconds = (active.start_time - trade_minute).total_seconds()
        max_tolerance = self.late_tolerance_minutes * 60

        if key in self._recent_closed_candles and elapsed_seconds <= max_tolerance:
            existing = self._recent_closed_candles[key]

            # Create an auditable record of what changed
            correction = CandleCorrectionRecord(
                symbol=symbol,
                timeframe=self.timeframe,
                start_time=existing.start_time,
                original_open=existing.open,
                original_high=existing.high,
                original_low=existing.low,
                original_close=existing.close,
                original_volume=existing.volume,
                original_trades=existing.trades_count,
                revised_high=max(existing.high, trade.price),
                revised_low=min(existing.low, trade.price),
                revised_close=existing.close,  # Close remains the tick closest to end_time
                revised_volume=existing.volume + trade.quantity,
                revised_trades=existing.trades_count + 1,
                late_trade_id=trade.trade_id,
                late_trade_price=trade.price,
                late_trade_quantity=trade.quantity,
                late_trade_timestamp=trade.timestamp,
                detected_at=now,
            )

            # Update cached closed candle
            existing.high = correction.revised_high
            existing.low = correction.revised_low
            existing.volume = correction.revised_volume
            existing.trades_count = correction.revised_trades

            logger.warning(
                "audited_candle_correction",
                symbol=symbol,
                start_time=existing.start_time.isoformat(),
                trade_id=trade.trade_id,
                price=trade.price,
                qty=trade.quantity,
            )

            # Dispatch correction to listeners (e.g. database update, persistence)
            await self._notify_corrected(correction, existing)

        else:
            logger.warning(
                "late_trade_outside_tolerance_discarded",
                symbol=symbol,
                trade_ts=trade.timestamp.isoformat(),
                active_start=active.start_time.isoformat(),
                elapsed_seconds=elapsed_seconds,
            )

    async def flush_all_active(self) -> list[Candle]:
        """Flush and close all in-flight candles (used during shutdown or interval sweeps)."""
        closed_list: list[Candle] = []
        for symbol, active in list(self._active_candles.items()):
            active.is_closed = True
            validation = self.quality_engine.validate_candle(active)
            if validation.is_valid:
                closed_key = f"{symbol}:{active.start_time.isoformat()}"
                self._recent_closed_candles[closed_key] = active.model_copy()
                self._recent_closed_keys.append(closed_key)
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

    async def _notify_corrected(self, correction: CandleCorrectionRecord, candle: Candle) -> None:
        """Dispatch candle correction to registered listeners."""
        for cb in self._correction_callbacks:
            try:
                try:
                    sig = inspect.signature(cb)
                    param_count = len(
                        [
                            p
                            for p in sig.parameters.values()
                            if p.default == inspect.Parameter.empty
                            and p.kind
                            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                        ]
                    )
                except (ValueError, TypeError):
                    param_count = 2

                res = cb(correction) if param_count == 1 else cb(correction, candle)

                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                logger.exception("candle_correction_callback_error", error=str(e))
