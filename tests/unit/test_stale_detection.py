"""Unit tests for Stale Data Detection and Health Tracking (Test Group G).

Tests compliance with Phase 2 Sections 13, 15, and 20:
- Tracking last event time, last receive time, last candle
- Stale data timeout detection
- Circuit breaker tripping on stale feed or excessive consecutive errors
- Feed recovery and breaker reset
"""

from datetime import UTC, datetime, timedelta

from data.ingestion.models import MarketEvent, NormalizedTrade
from data.validation.engine import DataQualityEngine
from data.validation.models import ValidationCategory


def test_stale_data_detection():
    """Verify that an inactive feed exceeding timeout trips the circuit breaker."""
    engine = DataQualityEngine(stale_data_timeout_seconds=2.0)
    now = datetime.now(UTC)

    # Initial trade establishes receive time
    trade = NormalizedTrade(
        trade_id="t1",
        source="binance",
        symbol="BTCUSDT",
        price=50000.0,
        quantity=1.0,
        timestamp=now - timedelta(seconds=5),  # 5 seconds ago
    )
    engine.validate_trade(trade)

    # Initially, last receive was just set to 'now' inside validate_trade
    # Let's artificially set _last_receive_time back by 10 seconds
    engine._last_receive_time["BTCUSDT"] = now - timedelta(seconds=10)

    # Check staleness
    is_stale = engine.check_feed_staleness("BTCUSDT")
    assert is_stale is True
    assert engine.is_breaker_tripped is True
    assert "Market data stale" in engine.breaker_reason

    # Verify incident was logged
    incidents = engine.get_recent_incidents()
    assert any(i.category == ValidationCategory.STALE_DATA for i in incidents)

    # Verify stats incremented
    stats = engine.get_stats("BTCUSDT")
    assert stats.stale_data_incidents >= 1


def test_feed_recovery_after_staleness():
    """Verify that fresh event arrival updates receive time and clears staleness."""
    engine = DataQualityEngine(stale_data_timeout_seconds=5.0)
    now = datetime.now(UTC)

    # Simulate stale state
    engine._last_receive_time["BTCUSDT"] = now - timedelta(seconds=20)
    assert engine.check_feed_staleness("BTCUSDT") is True

    # Reset breaker manually
    engine.reset_breaker()
    assert engine.is_breaker_tripped is False

    # Fresh event arrives
    fresh_trade = NormalizedTrade(
        trade_id="t_fresh",
        source="binance",
        symbol="BTCUSDT",
        price=50100.0,
        quantity=1.0,
        timestamp=now,
    )
    engine.validate_trade(fresh_trade)

    # Staleness check should now be healthy
    assert engine.check_feed_staleness("BTCUSDT") is False


def test_consecutive_errors_breaker_trip():
    """Verify breaker trips when consecutive validation errors exceed limit."""
    engine = DataQualityEngine(max_consecutive_errors=3)
    now = datetime.now(UTC)

    for i in range(3):
        # Invalid event (negative price in payload)
        bad_event = MarketEvent(
            source="binance",
            symbol="BTCUSDT",
            event_type="trade",
            event_time=now,
            receive_time=now,
            payload={"p": "-10.0", "q": "1.0", "t": f"bad_{i}"},
        )
        engine.validate_raw_event(bad_event)

    assert engine.is_breaker_tripped is True
    assert "consecutive" in engine.breaker_reason.lower()
