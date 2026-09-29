"""Historical Replay Engine for Phase 5.

Replays market data strictly chronologically, validating temporal monotonic ordering,
detecting data gaps and quality defects, and emitting CandleEvent instances.
Guarantees deterministic replay: given the same dataset and configuration, identical
sequences of events are produced.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from backtesting.events import CandleEvent
from data.ingestion.models import Candle

logger = logging.getLogger(__name__)


@dataclass
class ReplayWarning:
    """Diagnostic record for anomalies detected during historical replay."""

    timestamp: datetime
    symbol: str
    warning_type: str
    message: str


@dataclass
class ReplayStats:
    """Summary statistics for historical data replay execution."""

    total_candles: int = 0
    symbols_seen: set[str] = field(default_factory=set)
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    gaps_detected: int = 0
    invalid_candles: int = 0
    warnings: list[ReplayWarning] = field(default_factory=list)


class HistoricalReplayEngine:
    """Chronological event streamer enforcing temporal ordering and data hygiene."""

    TIMEFRAME_DELTAS: dict[str, timedelta] = {
        "1m": timedelta(minutes=1),
        "3m": timedelta(minutes=3),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "2h": timedelta(hours=2),
        "4h": timedelta(hours=4),
        "6h": timedelta(hours=6),
        "8h": timedelta(hours=8),
        "12h": timedelta(hours=12),
        "1d": timedelta(days=1),
    }

    def __init__(
        self,
        candles: list[Candle],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbols: list[str] | None = None,
        strict_monotony: bool = True,
        on_data_error: str = "skip_with_warning",  # "raise" | "skip_with_warning"
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = set(symbols) if symbols else None
        self.strict_monotony = strict_monotony
        self.on_data_error = on_data_error

        # Filter and prepare chronological stream
        self._raw_candles = candles
        self.stats = ReplayStats()
        self._prepared_candles: list[Candle] = []
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Filter, validate, and sort historical candles chronologically."""
        self.stats = ReplayStats()
        filtered: list[Candle] = []

        for c in self._raw_candles:
            # Check symbol filter
            if self.symbols and c.symbol not in self.symbols:
                continue

            # Ensure UTC comparison
            c_time = c.end_time if c.end_time.tzinfo else c.end_time.replace(tzinfo=UTC)

            if self.start_date:
                s_date = (
                    self.start_date
                    if self.start_date.tzinfo
                    else self.start_date.replace(tzinfo=UTC)
                )
                if c_time < s_date:
                    continue

            if self.end_date:
                e_date = (
                    self.end_date if self.end_date.tzinfo else self.end_date.replace(tzinfo=UTC)
                )
                if c_time > e_date:
                    continue

            # Sanity check candle physics
            if c.low > c.high or c.open <= 0 or c.close <= 0 or c.volume < 0:
                self.stats.invalid_candles += 1
                warn = ReplayWarning(
                    timestamp=c.end_time,
                    symbol=c.symbol,
                    warning_type="CORRUPT_CANDLE",
                    message=f"Invalid candle geometry: O={c.open}, H={c.high}, L={c.low}, C={c.close}, V={c.volume}",
                )
                self.stats.warnings.append(warn)
                if self.on_data_error == "raise":
                    raise ValueError(f"Corrupt candle rejected in strict mode: {warn.message}")
                continue

            filtered.append(c)

        # Sort strictly chronologically by end_time, then symbol
        filtered.sort(key=lambda x: (x.end_time, x.symbol))
        self._prepared_candles = filtered

    def stream_events(self) -> Iterator[CandleEvent]:
        """Stream historical candle events strictly in chronological order.

        Yields:
            CandleEvent at each completed candle timestamp.
        """
        last_seen_per_symbol: dict[str, datetime] = {}

        for candle in self._prepared_candles:
            sym = candle.symbol
            ts = candle.end_time

            # Check temporal monotony per symbol
            if sym in last_seen_per_symbol:
                prev_ts = last_seen_per_symbol[sym]
                if ts <= prev_ts:
                    msg = f"Non-monotonic timestamp detected for {sym}: previous={prev_ts}, current={ts}"
                    if self.strict_monotony:
                        raise ValueError(f"Temporal monotonicity violation: {msg}")
                    else:
                        self.stats.warnings.append(
                            ReplayWarning(
                                timestamp=ts, symbol=sym, warning_type="OUT_OF_ORDER", message=msg
                            )
                        )
                        continue

                # Check for unexpected gap
                tf_delta = self.TIMEFRAME_DELTAS.get(candle.timeframe, timedelta(minutes=1))
                time_diff = candle.start_time - prev_ts
                if time_diff > tf_delta * 0.1:  # Expected contiguous start == prev end
                    self.stats.gaps_detected += 1
                    self.stats.warnings.append(
                        ReplayWarning(
                            timestamp=candle.start_time,
                            symbol=sym,
                            warning_type="DATA_GAP",
                            message=f"Missing data gap between {prev_ts} and {candle.start_time}",
                        )
                    )

            last_seen_per_symbol[sym] = ts

            # Update statistics
            self.stats.total_candles += 1
            self.stats.symbols_seen.add(sym)
            if self.stats.first_timestamp is None:
                self.stats.first_timestamp = ts
            self.stats.last_timestamp = ts

            # Candle completes at end_time; only then is it observable
            yield CandleEvent(timestamp=ts, candle=candle)

    def __len__(self) -> int:
        return len(self._prepared_candles)
