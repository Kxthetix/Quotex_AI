"""Missing Data Detector and Sequence Gap Classifier.

Analyzes candlestick sequences to identify gaps, classifying them between genuine
market session closures, network outages, provider data dropouts, and low-volume no-trade periods.
"""

from datetime import datetime, timedelta

from data.ingestion.models import Candle, CandleGapReason, CandleGapRecord
from monitoring.logging import get_logger

logger = get_logger(__name__)


class MissingDataDetector:
    """Detects and classifies sequence gaps in candlestick time-series."""

    def __init__(
        self,
        timeframe: str = "1m",
        expected_interval: timedelta = timedelta(minutes=1),
        is_24_7_market: bool = True,  # True for Crypto, False for Equities/Forex
    ) -> None:
        self.timeframe = timeframe
        self.expected_interval = expected_interval
        self.is_24_7_market = is_24_7_market

    def detect_gaps(
        self,
        candles: list[Candle],
        network_outage_windows: list[tuple[datetime, datetime]] | None = None,
    ) -> list[CandleGapRecord]:
        """Detect gaps in a sequence of candles and classify the underlying cause."""
        if len(candles) < 2:
            return []

        # Sort chronologically
        sorted_candles = sorted(candles, key=lambda c: c.start_time)
        gaps: list[CandleGapRecord] = []
        outage_windows = network_outage_windows or []

        for i in range(len(sorted_candles) - 1):
            prev_candle = sorted_candles[i]
            curr_candle = sorted_candles[i + 1]

            # Expectation: curr_candle.start_time == prev_candle.end_time
            if curr_candle.start_time > prev_candle.end_time:
                gap_start = prev_candle.end_time
                gap_end = curr_candle.start_time
                gap_duration = gap_end - gap_start
                missing_count = int(
                    gap_duration.total_seconds() // self.expected_interval.total_seconds()
                )

                if missing_count <= 0:
                    continue

                # Classify the gap
                reason, is_failure = self._classify_gap(
                    gap_start, gap_end, missing_count, outage_windows
                )

                gap_record = CandleGapRecord(
                    symbol=prev_candle.symbol,
                    timeframe=self.timeframe,
                    gap_start=gap_start,
                    gap_end=gap_end,
                    missing_candles_count=missing_count,
                    reason=reason,
                    is_failure=is_failure,
                )
                gaps.append(gap_record)

                logger.info(
                    "candle_gap_detected",
                    symbol=prev_candle.symbol,
                    start=gap_start.isoformat(),
                    end=gap_end.isoformat(),
                    missing=missing_count,
                    reason=reason.value,
                    is_failure=is_failure,
                )

        return gaps

    def _classify_gap(
        self,
        gap_start: datetime,
        gap_end: datetime,
        missing_count: int,
        outage_windows: list[tuple[datetime, datetime]],
    ) -> tuple[CandleGapReason, bool]:
        """Classify reason for the gap and whether it constitutes a data failure."""
        # 1. Check if overlapping with known network outage window
        for out_start, out_end in outage_windows:
            if not (gap_end < out_start or gap_start > out_end):
                return CandleGapReason.NETWORK_OUTAGE, True

        # 2. Traditional market weekend / closure check (if not 24/7)
        if not self.is_24_7_market:
            # Check if interval intersects Saturday (5) or Sunday (6)
            current = gap_start
            while current <= gap_end:
                if current.weekday() in (5, 6):
                    return CandleGapReason.MARKET_SESSION_CLOSED, False
                current += timedelta(hours=6)
            if gap_end.weekday() in (5, 6):
                return CandleGapReason.MARKET_SESSION_CLOSED, False

        # 3. Low-liquidity isolated no-trade period (1-2 minutes gap in crypto or illiquid pair)
        if missing_count <= 2:
            return CandleGapReason.NO_TRADE_PERIOD, False

        # 4. Sustained dropout during active market hours -> Provider dropout
        return CandleGapReason.PROVIDER_DATA_DROPOUT, True
