"""Unit tests for Advanced Candle Construction & Auditable Correction Policy (Test Group D).

Tests compliance with Phase 2 Sections 10 & 11:
- Single & multiple events in a minute
- Price direction (increasing, decreasing, extremes)
- Volume accumulation
- Minute boundary transitions & empty minute gaps
- Out-of-order events within the active minute
- Late event auditable correction policy (produces CandleCorrectionRecord without silent mutation)
- Concurrent multi-symbol candle aggregation
"""

from datetime import UTC, datetime, timedelta

import pytest

from data.ingestion.candle_builder import CandleBuilder
from data.ingestion.models import CandleCorrectionRecord, NormalizedTrade


@pytest.mark.asyncio
async def test_candle_increasing_and_decreasing_prices():
    """Verify high and low accurately capture price extremes."""
    builder = CandleBuilder(timeframe="1m")
    base = datetime(2026, 1, 1, 14, 0, 0, tzinfo=UTC)

    # Sequence: 100 -> 120 -> 90 -> 110
    prices = [100.0, 120.0, 90.0, 110.0]
    for idx, p in enumerate(prices):
        trade = NormalizedTrade(
            trade_id=f"t_{idx}",
            source="binance",
            symbol="BTCUSDT",
            price=p,
            quantity=1.0,
            timestamp=base + timedelta(seconds=idx * 10),
        )
        await builder.process_trade(trade)

    active = builder.get_active_candle("BTCUSDT")
    assert active is not None
    assert active.open == 100.0
    assert active.high == 120.0
    assert active.low == 90.0
    assert active.close == 110.0
    assert active.volume == 4.0
    assert active.trades_count == 4


@pytest.mark.asyncio
async def test_candle_out_of_order_within_active_minute():
    """Verify ticks arriving out-of-order within the same active minute properly update high/low."""
    builder = CandleBuilder(timeframe="1m")
    base = datetime(2026, 1, 1, 14, 0, 0, tzinfo=UTC)

    # First event at :30 (price 100)
    t1 = NormalizedTrade(
        trade_id="t1",
        source="binance",
        symbol="BTCUSDT",
        price=100.0,
        quantity=1.0,
        timestamp=base + timedelta(seconds=30),
    )
    # Second event arriving later but with timestamp at :10 (price 80 - lower low)
    t2_earlier = NormalizedTrade(
        trade_id="t2",
        source="binance",
        symbol="BTCUSDT",
        price=80.0,
        quantity=2.0,
        timestamp=base + timedelta(seconds=10),
    )
    # Third event at :45 (price 150 - higher high)
    t3 = NormalizedTrade(
        trade_id="t3",
        source="binance",
        symbol="BTCUSDT",
        price=150.0,
        quantity=1.5,
        timestamp=base + timedelta(seconds=45),
    )

    await builder.process_trade(t1)
    await builder.process_trade(t2_earlier)
    await builder.process_trade(t3)

    active = builder.get_active_candle("BTCUSDT")
    assert active is not None
    assert active.high == 150.0
    assert active.low == 80.0
    assert active.volume == 4.5
    assert active.trades_count == 3


@pytest.mark.asyncio
async def test_candle_auditable_late_event_correction():
    """Verify that late events arriving after a minute has closed trigger CandleCorrectionRecord."""
    builder = CandleBuilder(timeframe="1m", late_tolerance_minutes=5)
    base = datetime(2026, 1, 1, 15, 0, 0, tzinfo=UTC)

    corrections: list[CandleCorrectionRecord] = []
    builder.on_candle_corrected(lambda c: corrections.append(c))

    # Minute 15:00 ticks
    t1 = NormalizedTrade(
        trade_id="t1",
        source="binance",
        symbol="BTCUSDT",
        price=100.0,
        quantity=1.0,
        timestamp=base + timedelta(seconds=10),
    )
    t2 = NormalizedTrade(
        trade_id="t2",
        source="binance",
        symbol="BTCUSDT",
        price=105.0,
        quantity=1.0,
        timestamp=base + timedelta(seconds=40),
    )
    # Minute 15:01 tick (closes 15:00 candle)
    t3 = NormalizedTrade(
        trade_id="t3",
        source="binance",
        symbol="BTCUSDT",
        price=108.0,
        quantity=1.0,
        timestamp=base + timedelta(minutes=1, seconds=5),
    )

    await builder.process_trade(t1)
    await builder.process_trade(t2)
    closed_1500 = await builder.process_trade(t3)

    assert closed_1500 is not None
    assert closed_1500.open == 100.0
    assert closed_1500.high == 105.0
    assert closed_1500.low == 100.0
    assert closed_1500.close == 105.0
    assert closed_1500.volume == 2.0

    # Late tick belonging to closed minute 15:00 (timestamp 15:00:55, price 125.0, volume 3.0)
    # Arrives when active minute is already 15:01
    late_tick = NormalizedTrade(
        trade_id="t_late",
        source="binance",
        symbol="BTCUSDT",
        price=125.0,
        quantity=3.0,
        timestamp=base + timedelta(seconds=55),
    )

    # Process late trade
    res = await builder.process_trade(late_tick)
    assert res is None  # Does not close the active 15:01 candle

    # Check auditable correction was dispatched
    assert len(corrections) == 1
    corr = corrections[0]
    assert corr.symbol == "BTCUSDT"
    assert corr.original_high == 105.0
    assert corr.revised_high == 125.0
    assert corr.original_volume == 2.0
    assert corr.revised_volume == 5.0
    assert corr.late_trade_id == "t_late"
    assert corr.late_trade_price == 125.0

    # Ensure active 15:01 candle was NOT polluted by late 15:00 tick
    active = builder.get_active_candle("BTCUSDT")
    assert active is not None
    assert active.start_time == base + timedelta(minutes=1)
    assert active.open == 108.0
    assert active.trades_count == 1


@pytest.mark.asyncio
async def test_candle_multiple_concurrent_symbols():
    """Verify independent aggregation for multiple symbols simultaneously."""
    builder = CandleBuilder(timeframe="1m")
    base = datetime(2026, 1, 1, 16, 0, 0, tzinfo=UTC)

    # BTC trade
    btc_trade = NormalizedTrade(
        trade_id="btc_1",
        source="binance",
        symbol="BTCUSDT",
        price=65000.0,
        quantity=0.5,
        timestamp=base + timedelta(seconds=15),
    )
    # ETH trade
    eth_trade = NormalizedTrade(
        trade_id="eth_1",
        source="binance",
        symbol="ETHUSDT",
        price=3500.0,
        quantity=5.0,
        timestamp=base + timedelta(seconds=20),
    )

    await builder.process_trade(btc_trade)
    await builder.process_trade(eth_trade)

    btc_active = builder.get_active_candle("BTCUSDT")
    eth_active = builder.get_active_candle("ETHUSDT")

    assert btc_active is not None
    assert btc_active.symbol == "BTCUSDT"
    assert btc_active.close == 65000.0

    assert eth_active is not None
    assert eth_active.symbol == "ETHUSDT"
    assert eth_active.close == 3500.0


@pytest.mark.asyncio
async def test_candle_empty_minute_gap():
    """Verify that multi-minute gaps cleanly roll over without intermediate synthetic candles."""
    builder = CandleBuilder(timeframe="1m")
    base = datetime(2026, 1, 1, 17, 0, 0, tzinfo=UTC)

    # Trade in minute 17:00
    t1 = NormalizedTrade(
        trade_id="t1",
        source="binance",
        symbol="BTCUSDT",
        price=100.0,
        quantity=1.0,
        timestamp=base + timedelta(seconds=10),
    )
    # Trade 5 minutes later at 17:05 (gap from 17:01 to 17:04)
    t2 = NormalizedTrade(
        trade_id="t2",
        source="binance",
        symbol="BTCUSDT",
        price=105.0,
        quantity=2.0,
        timestamp=base + timedelta(minutes=5, seconds=10),
    )

    await builder.process_trade(t1)
    closed = await builder.process_trade(t2)

    assert closed is not None
    assert closed.start_time == base  # 17:00 candle is closed

    active = builder.get_active_candle("BTCUSDT")
    assert active is not None
    assert active.start_time == base + timedelta(minutes=5)  # 17:05 active candle
    assert active.open == 105.0
