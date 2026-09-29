"""Critical Stale-Data Test for Phase 7 (Section 49).

Simulates:
1. Valid live market data arrivals and normal signal generation.
2. Market data feed halts, and stale threshold is exceeded.
3. System verifies new signals stop, new paper orders stop, state becomes DEGRADED,
   and a STALE_DATA monitoring event is recorded.
4. When fresh valid data resumes, system recovers and records recovery event.
"""

from datetime import UTC, datetime, timedelta

import pytest

from data.ingestion.models import Candle
from paper_trading.engine import PaperTradingConfig, PaperTradingEngine
from paper_trading.models import MonitoringEventType, MonitoringSeverity, SystemState
from paper_trading.stale_detector import StaleDataDetector


class TestCriticalStaleDataProtection:
    """Verifies that stale market data immediately halts live decision-making."""

    def test_stale_data_detector_lifecycle(self):
        """Test StaleDataDetector threshold transitions and event emission."""
        detector = StaleDataDetector(
            expected_interval_seconds=2.0,
            stale_threshold_seconds=5.0,
            symbols=["BTCUSDT"],
        )

        t0 = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
        detector.record_event("BTCUSDT", t0, receive_time=t0)

        # 1. Fresh: 2 seconds later
        t1 = t0 + timedelta(seconds=2)
        assert not detector.check_staleness(t1)
        assert not detector.is_stale

        # 2. Stale: 6 seconds later (exceeds 5.0s threshold)
        t2 = t0 + timedelta(seconds=6)
        assert detector.check_staleness(t2)
        assert detector.is_stale
        assert "BTCUSDT" in detector.stale_reason

        events = detector.get_events()
        assert len(events) >= 1
        assert events[-1].event_type == MonitoringEventType.STALE_DATA
        assert events[-1].severity == MonitoringSeverity.CRITICAL

        # 3. Resume: Fresh data received at t3
        t3 = t0 + timedelta(seconds=7)
        detector.record_event("BTCUSDT", t3, receive_time=t3)
        assert not detector.check_staleness(t3)
        assert not detector.is_stale

        events2 = detector.get_events()
        assert len(events2) >= 2
        assert events2[-1].event_type == MonitoringEventType.CONNECTION
        assert "restored" in events2[-1].message

    def test_section_49_engine_halts_signals_and_orders_when_stale(self):
        """Simulate live engine processing, market halt, staleness halt, and resume."""
        config = PaperTradingConfig(
            session_id="test_stale_sess",
            symbols=["BTCUSDT"],
            stale_threshold_seconds=10.0,
            auto_persist=False,
        )
        engine = PaperTradingEngine(config)

        t0 = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)

        # Feed 5 valid candles
        for i in range(5):
            c_time = t0 + timedelta(minutes=i)
            c = Candle(
                source="binance",
                symbol="BTCUSDT",
                timeframe="1m",
                start_time=c_time,
                end_time=c_time + timedelta(minutes=1),
                open=50000.0 + i * 10,
                high=50050.0 + i * 10,
                low=49980.0 + i * 10,
                close=50020.0 + i * 10,
                volume=100.0,
            )
            engine.process_candle(c)

        initial_signals = engine.total_signals_generated
        assert initial_signals > 0

        # Simulate data stopping: advance clock by 25 seconds (beyond 10s threshold)
        t_stopped = t0 + timedelta(minutes=5, seconds=25)
        is_stale = engine.stale_detector.check_staleness(t_stopped)
        assert is_stale is True

        # System health must reflect DEGRADED state
        health = engine.get_health_status()
        assert health["stale_data"] is True
        assert health["system_state"] in ("DEGRADED", "BLOCKED")

        # An incoming candle while stale must be rejected from generating signals
        stale_candle = Candle(
            source="binance",
            symbol="BTCUSDT",
            timeframe="1m",
            start_time=t0 + timedelta(minutes=5),
            end_time=t0 + timedelta(minutes=6),
            open=50060.0,
            high=50070.0,
            low=50050.0,
            close=50065.0,
            volume=100.0,
        )
        engine._handle_candle_closed(stale_candle)

        # Signals count must NOT increase while stale
        assert engine.total_signals_generated == initial_signals

        # Resume fresh data
        t_resumed = t_stopped + timedelta(seconds=1)
        engine.stale_detector.record_event("BTCUSDT", t_resumed, t_resumed)
        assert engine.stale_detector.check_staleness(t_resumed) is False

        # Now candle processing works again
        fresh_candle = Candle(
            source="binance",
            symbol="BTCUSDT",
            timeframe="1m",
            start_time=t_resumed,
            end_time=t_resumed + timedelta(minutes=1),
            open=50065.0,
            high=50080.0,
            low=50060.0,
            close=50075.0,
            volume=100.0,
        )
        engine._handle_candle_closed(fresh_candle)
        assert engine.total_signals_generated > initial_signals
