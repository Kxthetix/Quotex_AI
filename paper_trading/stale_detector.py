"""Stale Data Detection Engine for Phase 7 Paper Trading.

Monitors incoming event and candle arrival frequencies.
If data arrival interval exceeds the configured threshold:
1. Halts new signal generation
2. Halts new paper orders
3. Marks system state as DEGRADED
4. Emits structured monitoring event
5. Resumes only after valid fresh data is received.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from monitoring.logging import get_logger
from paper_trading.models import (
    MonitoringEvent,
    MonitoringEventType,
    MonitoringSeverity,
    StaleDataState,
    SystemState,
)

logger = get_logger(__name__)


class StaleDataDetector:
    """Detects missing or lagging market data across symbols and timeframes."""

    def __init__(
        self,
        expected_interval_seconds: float = 5.0,
        stale_threshold_seconds: float = 15.0,
        symbols: list[str] | None = None,
    ) -> None:
        self.expected_interval_seconds = expected_interval_seconds
        self.stale_threshold_seconds = stale_threshold_seconds
        self.symbols = [s.upper() for s in (symbols or [])]

        # Per-symbol timestamp tracking
        self._last_event_time: dict[str, datetime] = {}
        self._last_receive_time: dict[str, datetime] = {}
        self._last_candle_time: dict[str, datetime] = {}

        # Global freshness state
        self._is_stale: bool = False
        self._stale_reason: str = ""
        self._events: list[MonitoringEvent] = []

    @property
    def is_stale(self) -> bool:
        """Returns True if any active symbol's market data is stale."""
        return self._is_stale

    @property
    def stale_reason(self) -> str:
        return self._stale_reason

    def record_event(self, symbol: str, event_time: datetime, receive_time: datetime | None = None) -> None:
        """Record an incoming market event timestamp."""
        sym = symbol.upper()
        if sym not in self.symbols:
            self.symbols.append(sym)

        self._last_event_time[sym] = event_time if event_time.tzinfo else event_time.replace(tzinfo=UTC)
        rcv = receive_time or datetime.now(UTC)
        self._last_receive_time[sym] = rcv if rcv.tzinfo else rcv.replace(tzinfo=UTC)

        # If system was stale, check if receiving this event resolves staleness
        if self._is_stale:
            self.check_staleness()

    def record_candle(self, symbol: str, candle_time: datetime) -> None:
        """Record a closed or updated candle timestamp."""
        sym = symbol.upper()
        c_time = candle_time if candle_time.tzinfo else candle_time.replace(tzinfo=UTC)
        self._last_candle_time[sym] = c_time

    def check_staleness(self, current_time: datetime | None = None) -> bool:
        """Evaluate if data has exceeded the stale threshold.

        Returns True if stale, False if healthy.
        """
        now = current_time or datetime.now(UTC)
        if not now.tzinfo:
            now = now.replace(tzinfo=UTC)

        if not self.symbols:
            # If no symbols configured yet, cannot be evaluated as stale
            return False

        stale_symbols: list[str] = []
        for sym in self.symbols:
            last_rcv = self._last_receive_time.get(sym)
            if last_rcv is None:
                # No data received at all for this symbol
                stale_symbols.append(f"{sym} (no data received)")
                continue

            age_seconds = (now - last_rcv).total_seconds()
            if age_seconds > self.stale_threshold_seconds:
                stale_symbols.append(f"{sym} (age: {age_seconds:.1f}s > {self.stale_threshold_seconds}s)")

        was_stale = self._is_stale

        if stale_symbols:
            self._is_stale = True
            self._stale_reason = f"Data stale for symbols: {', '.join(stale_symbols)}"
            if not was_stale:
                event = MonitoringEvent(
                    severity=MonitoringSeverity.CRITICAL,
                    component="stale_data_detector",
                    event_type=MonitoringEventType.STALE_DATA,
                    message=f"Market data stale: {self._stale_reason}",
                    metadata={"stale_symbols": stale_symbols, "threshold_seconds": self.stale_threshold_seconds},
                )
                self._events.append(event)
                logger.warning("market_data_became_stale", reason=self._stale_reason)
            return True
        else:
            self._is_stale = False
            self._stale_reason = ""
            if was_stale:
                event = MonitoringEvent(
                    severity=MonitoringSeverity.INFO,
                    component="stale_data_detector",
                    event_type=MonitoringEventType.CONNECTION,
                    message="Market data freshness restored for all symbols",
                    metadata={"symbols": self.symbols},
                )
                self._events.append(event)
                logger.info("market_data_freshness_restored")
            return False

    def get_state(self, symbol: str | None = None) -> StaleDataState:
        """Retrieve structured freshness state for a symbol or aggregate."""
        sym = symbol.upper() if symbol else (self.symbols[0] if self.symbols else "")
        return StaleDataState(
            is_stale=self._is_stale,
            last_event_time=self._last_event_time.get(sym),
            last_receive_time=self._last_receive_time.get(sym),
            last_candle_time=self._last_candle_time.get(sym),
            expected_interval_seconds=self.expected_interval_seconds,
            stale_threshold_seconds=self.stale_threshold_seconds,
            reason=self._stale_reason,
        )

    def get_events(self) -> list[MonitoringEvent]:
        return list(self._events)
