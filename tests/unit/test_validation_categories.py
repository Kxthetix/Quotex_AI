"""Unit tests for Data Quality Validation Categories (Test Group B).

Verifies the 8 required validation categories:
INVALID_TIMESTAMP, MISSING_FIELD, INVALID_PRICE, INVALID_VOLUME,
UNKNOWN_EVENT_TYPE, MALFORMED_PAYLOAD, DUPLICATE_EVENT, OUT_OF_ORDER_EVENT,
plus DataQualityStats tracking.
"""

from datetime import UTC, datetime, timedelta

from data.ingestion.models import MarketEvent, NormalizedTrade
from data.validation.engine import DataQualityEngine
from data.validation.models import ValidationCategory


def test_validation_malformed_payload():
    """Verify empty or non-dictionary payload triggers MALFORMED_PAYLOAD."""
    engine = DataQualityEngine()
    event = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=datetime.now(UTC),
        receive_time=datetime.now(UTC),
        payload={},  # Empty payload
    )
    result = engine.validate_raw_event(event)
    assert not result.is_valid
    assert any(i.category == ValidationCategory.MALFORMED_PAYLOAD for i in result.issues)


def test_validation_unknown_event_type():
    """Verify unrecognized event type triggers UNKNOWN_EVENT_TYPE."""
    engine = DataQualityEngine()
    event = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="weird_unknown_action",
        event_time=datetime.now(UTC),
        receive_time=datetime.now(UTC),
        payload={"foo": "bar"},
    )
    result = engine.validate_raw_event(event)
    assert any(i.category == ValidationCategory.UNKNOWN_EVENT_TYPE for i in result.issues)


def test_validation_invalid_timestamp_future_drift():
    """Verify event time in far future triggers INVALID_TIMESTAMP."""
    engine = DataQualityEngine(max_clock_drift_seconds=5.0)
    future_time = datetime.now(UTC) + timedelta(minutes=10)
    event = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=future_time,
        receive_time=datetime.now(UTC),
        payload={"p": "50000.0"},
    )
    result = engine.validate_raw_event(event)
    assert not result.is_valid
    assert any(i.category == ValidationCategory.INVALID_TIMESTAMP for i in result.issues)


def test_validation_invalid_price():
    """Verify non-positive price triggers INVALID_PRICE."""
    engine = DataQualityEngine()
    now = datetime.now(UTC)

    # Zero price in raw event
    event_zero = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=now,
        receive_time=now,
        payload={"p": "0.0", "q": "1.0", "t": "1001"},
    )
    res_zero = engine.validate_raw_event(event_zero)
    assert not res_zero.is_valid
    assert any(i.category == ValidationCategory.INVALID_PRICE for i in res_zero.issues)

    # Negative price in raw event
    event_neg = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=now,
        receive_time=now,
        payload={"p": "-100.0", "q": "1.0", "t": "1002"},
    )
    res_neg = engine.validate_raw_event(event_neg)
    assert not res_neg.is_valid
    assert any(i.category == ValidationCategory.INVALID_PRICE for i in res_neg.issues)


def test_validation_invalid_volume():
    """Verify negative volume triggers INVALID_VOLUME."""
    engine = DataQualityEngine()
    now = datetime.now(UTC)
    event = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=now,
        receive_time=now,
        payload={"p": "50000.0", "q": "-0.5", "t": "1003"},
    )
    res = engine.validate_raw_event(event)
    assert not res.is_valid
    assert any(i.category == ValidationCategory.INVALID_VOLUME for i in res.issues)


def test_validation_missing_field():
    """Verify missing required trade_id in normalized trade or missing price in raw event triggers MISSING_FIELD."""
    engine = DataQualityEngine()
    trade = NormalizedTrade(
        trade_id="",  # empty required field
        source="binance",
        symbol="BTCUSDT",
        price=50000.0,
        quantity=1.0,
        timestamp=datetime.now(UTC),
    )
    res = engine.validate_trade(trade)
    assert not res.is_valid
    assert any(i.category == ValidationCategory.MISSING_FIELD for i in res.issues)

    # Also check missing field in raw event (missing symbol)
    raw_missing = MarketEvent(
        source="binance",
        symbol="",  # Missing symbol
        event_type="trade",
        event_time=datetime.now(UTC),
        receive_time=datetime.now(UTC),
        payload={"q": "1.0"},
    )
    raw_res = engine.validate_raw_event(raw_missing)
    assert not raw_res.is_valid
    assert any(i.category == ValidationCategory.MISSING_FIELD for i in raw_res.issues)


def test_validation_duplicate_event():
    """Verify duplicate trade_id triggers DUPLICATE_EVENT."""
    engine = DataQualityEngine()
    now = datetime.now(UTC)
    trade1 = NormalizedTrade(
        trade_id="dup_101",
        source="binance",
        symbol="BTCUSDT",
        price=50000.0,
        quantity=1.0,
        timestamp=now,
    )
    trade2 = NormalizedTrade(
        trade_id="dup_101",
        source="binance",
        symbol="BTCUSDT",
        price=50000.0,
        quantity=1.0,
        timestamp=now,
    )

    res1 = engine.validate_trade(trade1)
    assert res1.is_valid

    res2 = engine.validate_trade(trade2)
    assert not res2.is_valid
    assert any(i.category == ValidationCategory.DUPLICATE_EVENT for i in res2.issues)


def test_validation_out_of_order_event():
    """Verify trade with timestamp earlier than prior processed tick triggers OUT_OF_ORDER_EVENT."""
    engine = DataQualityEngine()
    now = datetime.now(UTC)

    trade1 = NormalizedTrade(
        trade_id="t_seq_1",
        source="binance",
        symbol="BTCUSDT",
        price=50000.0,
        quantity=1.0,
        timestamp=now,
    )
    trade2_older = NormalizedTrade(
        trade_id="t_seq_2",
        source="binance",
        symbol="BTCUSDT",
        price=50010.0,
        quantity=1.0,
        timestamp=now - timedelta(seconds=15),
    )

    res1 = engine.validate_trade(trade1)
    assert res1.is_valid

    res2 = engine.validate_trade(trade2_older)
    assert any(i.category == ValidationCategory.OUT_OF_ORDER_EVENT for i in res2.issues)


def test_data_quality_stats_accumulation():
    """Verify DataQualityStats correctly tracks counts across categories and symbols."""
    engine = DataQualityEngine()
    now = datetime.now(UTC)

    # Valid trade
    engine.validate_trade(
        NormalizedTrade(
            trade_id="t1",
            source="binance",
            symbol="BTCUSDT",
            price=50000.0,
            quantity=1.0,
            timestamp=now,
        )
    )

    # Invalid event (negative price in payload)
    engine.validate_raw_event(
        MarketEvent(
            source="binance",
            symbol="BTCUSDT",
            event_type="trade",
            event_time=now,
            receive_time=now,
            payload={"p": "-10.0", "q": "1.0", "t": "t2"},
        )
    )

    # Invalid event for another symbol (ETHUSDT)
    engine.validate_raw_event(
        MarketEvent(
            source="binance",
            symbol="ETHUSDT",
            event_type="trade",
            event_time=now,
            receive_time=now,
            payload={"p": "-5.0", "q": "1.0", "t": "t3"},
        )
    )

    btc_stats = engine.get_stats("BTCUSDT")
    assert btc_stats.symbol == "BTCUSDT"
    assert btc_stats.valid_events == 1
    assert btc_stats.invalid_events == 1
    assert btc_stats.category_counts[ValidationCategory.INVALID_PRICE.value] == 1

    eth_stats = engine.get_stats("ETHUSDT")
    assert eth_stats.symbol == "ETHUSDT"
    assert eth_stats.invalid_events == 1

    # Global aggregate
    global_stats = engine.get_stats()
    assert global_stats.valid_events == 1
    assert global_stats.invalid_events == 2
    assert global_stats.category_counts[ValidationCategory.INVALID_PRICE.value] == 2
