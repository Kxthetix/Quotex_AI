"""Market Data Disconnection and Gap Recovery Engine (Sections 10, 11, 12).

Manages:
- Socket disconnection protocol (halting live trading decisions immediately)
- Missing market interval detection across disconnections
- Historical backfill request and verification of candle continuity
- Blocking signal generation if intervals cannot be safely reconstructed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

from data.ingestion.models import Candle
from monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketGapInfo:
    """Detected market data gap interval across a disconnection."""
    symbol: str
    gap_start: datetime
    gap_end: datetime
    missing_bars_estimate: int
    is_recovered: bool = False
    details: str = ""


class MarketDataGapRecovery:
    """Detects, backfills, and validates continuity of market data after disconnections."""

    def __init__(
        self,
        timeframe_minutes: int = 1,
        max_allowable_gap_seconds: float = 65.0,
        backfill_fn: Callable[[str, datetime, datetime], list[Candle]] | None = None,
    ) -> None:
        self.timeframe_minutes = timeframe_minutes
        self.max_allowable_gap_seconds = max_allowable_gap_seconds
        self.backfill_fn = backfill_fn

        self._last_seen_candle_time: dict[str, datetime] = {}
        self._blocked_symbols: set[str] = set()
        self._detected_gaps: list[MarketGapInfo] = []

    def record_candle(self, candle: Candle) -> None:
        """Record the timestamp of a received closed candle."""
        sym = candle.symbol.upper()
        t = candle.end_time if candle.end_time.tzinfo else candle.end_time.replace(tzinfo=UTC)
        self._last_seen_candle_time[sym] = t

    def is_symbol_blocked(self, symbol: str) -> bool:
        """Returns True if the symbol is blocked due to unrecovered gaps."""
        return symbol.upper() in self._blocked_symbols

    def handle_reconnection(self, symbol: str, current_time: datetime) -> MarketGapInfo | None:
        """Assess whether a market data gap occurred during the disconnection."""
        clean_sym = symbol.upper()
        now = current_time if current_time.tzinfo else current_time.replace(tzinfo=UTC)
        last_t = self._last_seen_candle_time.get(clean_sym)

        if not last_t:
            # First time seeing symbol
            self._last_seen_candle_time[clean_sym] = now
            return None

        gap_duration = (now - last_t).total_seconds()

        # If gap exceeded allowable threshold (e.g. > 1 candle length)
        if gap_duration > self.max_allowable_gap_seconds:
            missing_bars = int(gap_duration // (self.timeframe_minutes * 60))
            gap_info = MarketGapInfo(
                symbol=clean_sym,
                gap_start=last_t,
                gap_end=now,
                missing_bars_estimate=missing_bars,
                details=f"Gap of {gap_duration:.1f}s detected (~{missing_bars} missing bars).",
            )
            self._detected_gaps.append(gap_info)
            logger.warning("market_data_gap_detected", symbol=clean_sym, duration_s=gap_duration, missing_bars=missing_bars)

            # Attempt historical backfill if supported
            if self.backfill_fn:
                try:
                    backfilled = self.backfill_fn(clean_sym, last_t, now)
                    if backfilled and len(backfilled) >= missing_bars:
                        gap_info.is_recovered = True
                        gap_info.details += " Successfully backfilled and continuity restored."
                        self._blocked_symbols.discard(clean_sym)
                        logger.info("market_gap_backfilled_successfully", symbol=clean_sym, count=len(backfilled))
                        return gap_info
                except Exception as e:
                    logger.error("market_gap_backfill_failed", symbol=clean_sym, error=str(e))

            # If backfill not available or failed: BLOCK symbol decisions
            self._blocked_symbols.add(clean_sym)
            gap_info.is_recovered = False
            gap_info.details += " Backfill unavailable or incomplete. Symbol blocked from signal generation."
            logger.error("symbol_blocked_due_to_unrecovered_gap", symbol=clean_sym)
            return gap_info

        # Normal small jitter, no gap
        self._last_seen_candle_time[clean_sym] = now
        self._blocked_symbols.discard(clean_sym)
        return None

    def unblock_symbol(self, symbol: str, reason: str = "Clean candle alignment achieved") -> None:
        """Manually or automatically unblock symbol once a fresh closed candle is confirmed."""
        clean_sym = symbol.upper()
        self._blocked_symbols.discard(clean_sym)
        logger.info("symbol_unblocked_for_trading", symbol=clean_sym, reason=reason)

    def get_detected_gaps(self) -> list[MarketGapInfo]:
        return list(self._detected_gaps)
