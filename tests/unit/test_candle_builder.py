"""Unit tests for the 1-minute CandleBuilder."""

from datetime import UTC, datetime

import pytest

from data.ingestion.candle_builder import CandleBuilder
from data.ingestion.models import NormalizedTrade


@pytest.mark.asyncio
async def test_candle_builder_single_minute_aggregation():
    builder = CandleBuilder(timeframe="1m")

    # Three trades in the same minute: 12:00:10, 12:00:25, 12:00:50
    t1 = NormalizedTrade(
        trade_id="1",
        source="binance",
        symbol="BTCUSDT",
        price=100.0,
        quantity=1.0,
        timestamp=datetime(2026, 1, 1, 12, 0, 10, tzinfo=UTC),
    )
    t2 = NormalizedTrade(
        trade_id="2",
        source="binance",
        symbol="BTCUSDT",
        price=105.0,
        quantity=2.0,
        timestamp=datetime(2026, 1, 1, 12, 0, 25, tzinfo=UTC),
    )
    t3 = NormalizedTrade(
        trade_id="3",
        source="binance",
        symbol="BTCUSDT",
        price=98.0,
        quantity=1.5,
        timestamp=datetime(2026, 1, 1, 12, 0, 50, tzinfo=UTC),
    )

    closed1 = await builder.process_trade(t1)
    assert closed1 is None

    closed2 = await builder.process_trade(t2)
    assert closed2 is None

    closed3 = await builder.process_trade(t3)
    assert closed3 is None

    active = builder.get_active_candle("BTCUSDT")
    assert active is not None
    assert active.open == 100.0
    assert active.high == 105.0
    assert active.low == 98.0
    assert active.close == 98.0
    assert active.volume == 4.5
    assert active.trades_count == 3
    assert not active.is_closed


@pytest.mark.asyncio
async def test_candle_builder_minute_rollover():
    builder = CandleBuilder(timeframe="1m")

    closed_candles = []
    builder.on_candle_closed(lambda c: closed_candles.append(c))

    # Minute 1: 12:00:15
    t1 = NormalizedTrade(
        trade_id="1",
        source="binance",
        symbol="BTCUSDT",
        price=100.0,
        quantity=1.0,
        timestamp=datetime(2026, 1, 1, 12, 0, 15, tzinfo=UTC),
    )
    # Minute 1: 12:00:45
    t2 = NormalizedTrade(
        trade_id="2",
        source="binance",
        symbol="BTCUSDT",
        price=110.0,
        quantity=2.0,
        timestamp=datetime(2026, 1, 1, 12, 0, 45, tzinfo=UTC),
    )
    # Minute 2: 12:01:05 (triggers rollover)
    t3 = NormalizedTrade(
        trade_id="3",
        source="binance",
        symbol="BTCUSDT",
        price=112.0,
        quantity=0.5,
        timestamp=datetime(2026, 1, 1, 12, 1, 5, tzinfo=UTC),
    )

    await builder.process_trade(t1)
    await builder.process_trade(t2)
    closed = await builder.process_trade(t3)

    assert closed is not None
    assert closed.is_closed
    assert closed.start_time == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert closed.end_time == datetime(2026, 1, 1, 12, 1, 0, tzinfo=UTC)
    assert closed.open == 100.0
    assert closed.high == 110.0
    assert closed.low == 100.0
    assert closed.close == 110.0
    assert closed.volume == 3.0
    assert closed.trades_count == 2

    assert len(closed_candles) == 1
    assert closed_candles[0] == closed

    # Check new active candle for minute 12:01
    active = builder.get_active_candle("BTCUSDT")
    assert active is not None
    assert active.start_time == datetime(2026, 1, 1, 12, 1, 0, tzinfo=UTC)
    assert active.open == 112.0
    assert active.close == 112.0
    assert active.volume == 0.5
    assert active.trades_count == 1


@pytest.mark.asyncio
async def test_candle_builder_flush():
    builder = CandleBuilder(timeframe="1m")
    t1 = NormalizedTrade(
        trade_id="1",
        source="binance",
        symbol="ETHUSDT",
        price=3000.0,
        quantity=5.0,
        timestamp=datetime(2026, 1, 1, 12, 0, 10, tzinfo=UTC),
    )
    await builder.process_trade(t1)

    flushed = await builder.flush_all_active()
    assert len(flushed) == 1
    assert flushed[0].symbol == "ETHUSDT"
    assert flushed[0].is_closed
    assert builder.get_active_candle("ETHUSDT") is None
