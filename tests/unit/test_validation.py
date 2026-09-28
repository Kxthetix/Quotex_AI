"""Unit tests for the Data Quality Engine."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from data.ingestion.models import Candle, MarketEvent, NormalizedTrade
from data.validation.engine import DataQualityEngine


def test_validate_raw_event_timestamp_drift():
    engine = DataQualityEngine(max_clock_drift_seconds=5.0)

    # Event in normal time range
    normal_event = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=datetime.now(UTC),
        payload={"test": 123},
    )
    res = engine.validate_raw_event(normal_event)
    assert res.is_valid

    # Event far in future
    future_event = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=datetime.now(UTC) + timedelta(minutes=10),
        payload={"test": 123},
    )
    res_future = engine.validate_raw_event(future_event)
    assert not res_future.is_valid
    assert any(i.check_name == "timestamp_future_drift" for i in res_future.issues)


def test_validate_duplicate_trade():
    engine = DataQualityEngine()
    now = datetime.now(UTC)

    trade1 = NormalizedTrade(
        trade_id="trade_1001",
        source="binance",
        symbol="BTCUSDT",
        price=65000.0,
        quantity=0.5,
        side="buy",
        timestamp=now,
    )
    trade2_duplicate = NormalizedTrade(
        trade_id="trade_1001",
        source="binance",
        symbol="BTCUSDT",
        price=65000.0,
        quantity=0.5,
        side="buy",
        timestamp=now,
    )

    res1 = engine.validate_trade(trade1)
    assert res1.is_valid

    res2 = engine.validate_trade(trade2_duplicate)
    assert any(i.check_name == "duplicate_trade" for i in res2.issues)


def test_validate_invalid_trade_values():
    engine = DataQualityEngine()
    now = datetime.now(UTC)

    # 1. Pydantic validation rejects negative price at model level
    with pytest.raises(ValidationError):
        NormalizedTrade(
            trade_id="t1",
            source="binance",
            symbol="BTCUSDT",
            price=-10.0,
            quantity=1.0,
            side="buy",
            timestamp=now,
        )

    # 2. Engine check catches non-positive price if constructed bypassing pydantic
    bad_price = NormalizedTrade.model_construct(
        trade_id="t1",
        source="binance",
        symbol="BTCUSDT",
        price=-10.0,
        quantity=1.0,
        side="buy",
        timestamp=now,
        receive_time=now,
    )
    assert not engine.validate_trade(bad_price).is_valid

    # Engine check catches non-positive quantity
    bad_qty = NormalizedTrade.model_construct(
        trade_id="t2",
        source="binance",
        symbol="BTCUSDT",
        price=50000.0,
        quantity=0.0,
        side="buy",
        timestamp=now,
        receive_time=now,
    )
    assert not engine.validate_trade(bad_qty).is_valid


def test_validate_price_outlier_jump():
    engine = DataQualityEngine(max_price_change_pct=15.0)
    now = datetime.now(UTC)

    # Initial valid trade sets baseline
    t1 = NormalizedTrade(
        trade_id="t1",
        source="binance",
        symbol="BTCUSDT",
        price=50000.0,
        quantity=1.0,
        timestamp=now,
    )
    assert engine.validate_trade(t1).is_valid

    # Extreme jump (+30%)
    t2_spike = NormalizedTrade(
        trade_id="t2",
        source="binance",
        symbol="BTCUSDT",
        price=65000.0,
        quantity=1.0,
        timestamp=now,
    )
    res_spike = engine.validate_trade(t2_spike)
    assert not res_spike.is_valid
    assert any(i.check_name == "price_outlier_jump" for i in res_spike.issues)


def test_validate_candle_ohlc_consistency():
    engine = DataQualityEngine()
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    end = datetime(2026, 1, 1, 12, 1, 0, tzinfo=UTC)

    # Valid candle
    valid_candle = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        volume=10.0,
        start_time=start,
        end_time=end,
    )
    assert engine.validate_candle(valid_candle).is_valid

    # Invalid: High < Open
    bad_high = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=100.0,
        high=90.0,
        low=85.0,
        close=88.0,
        volume=10.0,
        start_time=start,
        end_time=end,
    )
    res_high = engine.validate_candle(bad_high)
    assert not res_high.is_valid
    assert any("High (90.0) is less than Open (100.0)" in i.message for i in res_high.issues)

    # Invalid: Low > Close
    bad_low = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=100.0,
        high=110.0,
        low=102.0,
        close=98.0,
        volume=10.0,
        start_time=start,
        end_time=end,
    )
    res_low = engine.validate_candle(bad_low)
    assert not res_low.is_valid

    # Invalid: Negative volume
    with pytest.raises(ValidationError):
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=-1.0,
            start_time=start,
            end_time=end,
        )

    bad_vol = Candle.model_construct(
        symbol="BTCUSDT",
        timeframe="1m",
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        volume=-1.0,
        start_time=start,
        end_time=end,
    )
    assert not engine.validate_candle(bad_vol).is_valid


def test_missing_candle_gap_detection():
    engine = DataQualityEngine()
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 12, 1, 0, tzinfo=UTC)
    # Gap: skip 12:01 -> 12:02
    t3 = datetime(2026, 1, 1, 12, 2, 0, tzinfo=UTC)
    t4 = datetime(2026, 1, 1, 12, 3, 0, tzinfo=UTC)

    c1 = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=100,
        high=105,
        low=95,
        close=102,
        volume=1,
        start_time=t0,
        end_time=t1,
    )
    c2_gap = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open=102,
        high=108,
        low=101,
        close=106,
        volume=1,
        start_time=t3,
        end_time=t4,
    )

    assert engine.validate_candle(c1).is_valid
    res2 = engine.validate_candle(c2_gap)
    # Has warning for gap
    assert any(i.check_name == "missing_candle_gap" for i in res2.issues)


def test_circuit_breaker_trips_on_consecutive_errors():
    engine = DataQualityEngine(max_consecutive_errors=3)
    now = datetime.now(UTC)

    assert not engine.is_breaker_tripped

    # Inject 3 consecutive invalid trades via model_construct
    for i in range(3):
        bad_trade = NormalizedTrade.model_construct(
            trade_id=f"bad_{i}",
            source="binance",
            symbol="BTCUSDT",
            price=-1.0,
            quantity=1.0,
            side="buy",
            timestamp=now,
            receive_time=now,
        )
        engine.validate_trade(bad_trade)

    assert engine.is_breaker_tripped
    assert "Exceeded max consecutive data quality errors" in (engine.breaker_reason or "")

    # Reset
    engine.reset_breaker()
    assert not engine.is_breaker_tripped
