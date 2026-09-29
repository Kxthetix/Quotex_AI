"""End-to-End Pipeline and API Integration Test for Phase 7 Paper Trading."""

from datetime import UTC, datetime, timedelta

import pytest

from api.main import (
    get_paper_health,
    get_paper_portfolio,
    get_paper_predictions,
    get_paper_report,
    get_paper_signals,
    get_paper_status,
    set_paper_engine,
)
from data.ingestion.models import Candle, NormalizedTrade
from paper_trading.engine import PaperTradingConfig, PaperTradingEngine
from paper_trading.models import PredictionStatus, SessionStatus


class TestPaperTradingPipelineE2E:
    """Full lifecycle testing of Phase 7 Paper Trading system."""

    @pytest.mark.asyncio
    async def test_full_pipeline_run_and_reporting(self):
        config = PaperTradingConfig(
            session_id="test_e2e_session",
            symbols=["BTCUSDT"],
            timeframe="1m",
            initial_capital=10000.0,
            evaluation_horizon_seconds=60.0,  # 1 min horizon for fast test
            auto_persist=False,
        )
        engine = PaperTradingEngine(config)

        # 1. Startup
        await engine.start()
        assert engine.is_running is True
        assert engine.session.status == SessionStatus.RUNNING

        t0 = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)

        # 2. Feed series of 15 closed candles (sufficient to calculate features)
        for i in range(15):
            c_start = t0 + timedelta(minutes=i)
            c_end = c_start + timedelta(minutes=1)
            # Create an uptrending price series
            price = 50000.0 + i * 50
            candle = Candle(
                source="binance",
                symbol="BTCUSDT",
                timeframe="1m",
                start_time=c_start,
                end_time=c_end,
                open=price - 10,
                high=price + 20,
                low=price - 15,
                close=price,
                volume=100.0 + i * 5,
            )
            engine.process_candle(candle)

        # 3. Verify predictions and signals generated
        preds = engine.prediction_ledger.get_all_predictions()
        assert len(preds) > 0
        assert engine.total_signals_generated > 0

        # Verify market state in memory
        mkt_state = engine.market_state.get_state("BTCUSDT")
        assert mkt_state.latest_price > 50000.0
        assert mkt_state.symbol == "BTCUSDT"

        # 4. Advance time past the evaluation horizon (1 min) and trigger evaluation
        t_eval = t0 + timedelta(minutes=16)
        evaluated = engine.outcome_tracker.check_pending_evaluations(
            current_time=t_eval,
            current_prices={"BTCUSDT": 51000.0},
        )
        assert len(evaluated) > 0
        for ep in evaluated:
            assert ep.status == PredictionStatus.EVALUATED
            assert ep.realized_return is not None
            assert ep.actual_outcome in ("CORRECT", "INCORRECT", "NEUTRAL")

        # 5. Check Latency Tracking
        health = engine.get_health_status()
        assert health["total_events_processed"] >= 0
        assert "latency_percentiles" in health

        # 6. Stop and Graceful Shutdown
        report = await engine.stop(reason="E2E test complete")
        assert engine.is_running is False
        assert engine.session.status == SessionStatus.STOPPED
        assert "portfolio_performance" in report
        assert "prediction_statistics" in report
        assert "execution_statistics" in report

    @pytest.mark.asyncio
    async def test_fastapi_endpoints_integration(self):
        """Test API endpoints in api/main.py for paper trading."""
        config = PaperTradingConfig(
            session_id="test_api_session",
            symbols=["BTCUSDT"],
            auto_persist=False,
        )
        engine = PaperTradingEngine(config)
        set_paper_engine(engine)

        # GET /api/v1/paper/status
        status_data = await get_paper_status()
        assert "session" in status_data
        assert "market_states" in status_data
        assert status_data["session"]["session_id"] == "test_api_session"

        # GET /api/v1/paper/portfolio
        port_data = await get_paper_portfolio()
        assert port_data["cash"] == 10000.0
        assert port_data["equity"] == 10000.0

        # GET /api/v1/paper/signals
        sig_data = await get_paper_signals()
        assert "total_signals_generated" in sig_data

        # GET /api/v1/paper/predictions
        pred_data = await get_paper_predictions()
        assert "total_count" in pred_data
        assert "pending_count" in pred_data

        # GET /api/v1/paper/health
        health_data = await get_paper_health()
        assert "system_state" in health_data

        # GET /api/v1/paper/report
        report_data = await get_paper_report()
        assert "session_metadata" in report_data
