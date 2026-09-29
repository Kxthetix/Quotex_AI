"""Unit tests for Market Data Gap Recovery (Sections 10, 11, 12, 13)."""

from datetime import UTC, datetime, timedelta

from data.ingestion.models import Candle
from production.recovery.gap_recovery import MarketDataGapRecovery


def test_gap_recovery_no_gap():
    recovery = MarketDataGapRecovery(timeframe_minutes=1, max_allowable_gap_seconds=65.0)

    t0 = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)
    c1 = Candle(
        symbol="BTC/USDT",
        timeframe="1m",
        start_time=t0 - timedelta(minutes=1),
        end_time=t0,
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50050.0,
        volume=10.0,
    )
    recovery.record_candle(c1)

    # Next tick 30 seconds later (no gap)
    t1 = t0 + timedelta(seconds=30)
    gap_info = recovery.handle_reconnection("BTC/USDT", t1)
    assert gap_info is None
    assert recovery.is_symbol_blocked("BTC/USDT") is False


def test_gap_recovery_detects_gap_and_backfills():
    t0 = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)

    def mock_backfiller(symbol, start_t, end_t):
        return [
            Candle(
                symbol=symbol,
                timeframe="1m",
                start_time=start_t,
                end_time=start_t + timedelta(minutes=1),
                open=50050.0,
                high=50200.0,
                low=50000.0,
                close=50150.0,
                volume=15.0,
            ),
            Candle(
                symbol=symbol,
                timeframe="1m",
                start_time=start_t + timedelta(minutes=1),
                end_time=end_t,
                open=50150.0,
                high=50300.0,
                low=50100.0,
                close=50250.0,
                volume=20.0,
            ),
        ]

    recovery = MarketDataGapRecovery(
        timeframe_minutes=1,
        max_allowable_gap_seconds=65.0,
        backfill_fn=mock_backfiller,
    )

    c0 = Candle(
        symbol="BTC/USDT",
        timeframe="1m",
        start_time=t0 - timedelta(minutes=1),
        end_time=t0,
        open=50000.0,
        high=50100.0,
        low=49900.0,
        close=50050.0,
        volume=10.0,
    )
    recovery.record_candle(c0)

    # Reconnect after 2.5 minutes (150 seconds -> ~2 missing bars)
    t_reconnect = t0 + timedelta(seconds=150)
    gap_info = recovery.handle_reconnection("BTC/USDT", t_reconnect)

    assert gap_info is not None
    assert gap_info.is_recovered is True
    assert gap_info.missing_bars_estimate == 2
    assert recovery.is_symbol_blocked("BTC/USDT") is False


def test_gap_recovery_blocks_symbol_if_backfill_unavailable():
    t0 = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)
    recovery = MarketDataGapRecovery(timeframe_minutes=1, max_allowable_gap_seconds=65.0, backfill_fn=None)

    c0 = Candle(
        symbol="ETH/USDT",
        timeframe="1m",
        start_time=t0 - timedelta(minutes=1),
        end_time=t0,
        open=3000.0,
        high=3020.0,
        low=2990.0,
        close=3010.0,
        volume=50.0,
    )
    recovery.record_candle(c0)

    t_reconnect = t0 + timedelta(minutes=5)  # 300s gap
    gap_info = recovery.handle_reconnection("ETH/USDT", t_reconnect)

    assert gap_info is not None
    assert gap_info.is_recovered is False
    assert recovery.is_symbol_blocked("ETH/USDT") is True
