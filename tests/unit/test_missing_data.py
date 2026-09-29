"""Unit tests for Missing Data & Gap Detection (Test Group H).

Tests compliance with Phase 2 Section 12:
- Identify missing candle intervals
- Distinguish between provider dropout and weekend/market session closure
- Record structured CandleGapRecord metrics
"""

from datetime import UTC, datetime, timedelta

from data.ingestion.models import Candle, CandleGapReason
from data.validation.missing_data import MissingDataDetector


def _make_candle(symbol: str, start: datetime) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe="1m",
        open=50000.0,
        high=50050.0,
        low=49980.0,
        close=50020.0,
        volume=1.0,
        trades_count=10,
        start_time=start,
        end_time=start + timedelta(minutes=1),
        is_closed=True,
    )


def test_missing_data_contiguous_no_gaps():
    """Verify contiguous 1-minute sequence yields zero detected gaps."""
    detector = MissingDataDetector(is_24_7_market=True)
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)

    candles = [_make_candle("BTCUSDT", base + timedelta(minutes=i)) for i in range(5)]
    gaps = detector.detect_gaps(candles)
    assert len(gaps) == 0


def test_missing_data_detect_crypto_dropout():
    """Verify gap in 24/7 crypto series is classified as PROVIDER_DATA_DROPOUT."""
    detector = MissingDataDetector(is_24_7_market=True)
    base = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)

    # Candle 0 at 10:00, Candle 1 at 10:05 (gap of 4 minutes)
    c0 = _make_candle("BTCUSDT", base)
    c1 = _make_candle("BTCUSDT", base + timedelta(minutes=5))

    gaps = detector.detect_gaps([c0, c1])
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.symbol == "BTCUSDT"
    assert gap.timeframe == "1m"
    assert gap.gap_start == base + timedelta(minutes=1)
    assert gap.gap_end == base + timedelta(minutes=5)
    assert gap.missing_candles_count == 4
    assert gap.reason == CandleGapReason.PROVIDER_DATA_DROPOUT


def test_missing_data_detect_weekend_closure():
    """Verify gap spanning a Saturday/Sunday in a traditional market is classified as MARKET_SESSION_CLOSED."""
    detector = MissingDataDetector(is_24_7_market=False)

    # Friday 2026-01-02 21:00 UTC to Monday 2026-01-05 09:00 UTC
    friday_close = datetime(2026, 1, 2, 21, 0, 0, tzinfo=UTC)
    monday_open = datetime(2026, 1, 5, 9, 0, 0, tzinfo=UTC)

    c_friday = _make_candle("SPY", friday_close)
    c_monday = _make_candle("SPY", monday_open)

    gaps = detector.detect_gaps([c_friday, c_monday])
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.reason == CandleGapReason.MARKET_SESSION_CLOSED
