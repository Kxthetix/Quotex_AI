"""Critical Restart and Recovery Test for Phase 7 (Section 52).

Verifies:
1. Runs a paper session creating predictions, orders, fills, and positions.
2. Stops the system, saving checkpoint snapshot.
3. Restarts the system from the checkpoint.
4. Verifies:
   - Paper portfolio cash, equity, and positions restored
   - Trade ledger restored without duplicates
   - Prediction ledger restored with pending evaluations intact
   - Model versions unchanged
   - Duplicate event/trade protection is enforced.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from paper_trading.execution import PaperExecutionEngine
from paper_trading.models import (
    ExitReason,
    PaperFill,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperPosition,
    PaperTradeRecord,
    PaperTradingSession,
    PredictionRecord,
    PredictionStatus,
    SessionStatus,
)
from paper_trading.portfolio import PaperPortfolioManager
from paper_trading.prediction_ledger import PredictionLedger
from paper_trading.recovery import SessionRecoveryEngine


class TestCriticalRestartRecovery:
    """Verifies that restart recovery is completely lossless and idempotent."""

    def test_section_52_restart_checkpoint_and_restoration(self, tmp_path: Path):
        recovery = SessionRecoveryEngine(checkpoint_dir=tmp_path)

        # 1. Setup active session
        session = PaperTradingSession(
            session_id="session_restart_sec52",
            configuration_version="v1.0.0",
            model_versions={"direction": "v1.0.0"},
            feature_version="v1.0.0",
            signal_version="v1.0.0",
            initial_capital=10000.0,
            symbols=["BTCUSDT"],
        )

        portfolio_mgr = PaperPortfolioManager(initial_capital=10000.0)
        execution_engine = PaperExecutionEngine()
        prediction_ledger = PredictionLedger()

        # Create active position: Buy 0.1 BTC at 50,000
        order = execution_engine.create_order(
            signal_id="sig_test_1",
            symbol="BTCUSDT",
            side=PaperOrderSide.BUY,
            requested_quantity=0.1,
            requested_price=50000.0,
        )
        fill = execution_engine.simulate_execution(order, current_price=50000.0)
        assert fill is not None
        portfolio_mgr.process_fill(fill, signal_id="sig_test_1")

        # Create completed trade: Sell 0.05 BTC at 52,000 (partial close)
        order2 = execution_engine.create_order(
            signal_id="sig_test_2",
            symbol="BTCUSDT",
            side=PaperOrderSide.SELL,
            requested_quantity=0.05,
            requested_price=52000.0,
        )
        fill2 = execution_engine.simulate_execution(order2, current_price=52000.0)
        assert fill2 is not None
        trade = portfolio_mgr.process_fill(fill2, signal_id="sig_test_2", exit_reason=ExitReason.TAKE_PROFIT)
        assert trade is not None

        # Add 1 evaluated prediction and 1 pending prediction
        p1 = PredictionRecord(
            prediction_id="pred_eval_1",
            symbol="BTCUSDT",
            signal_action="BUY",
            confidence=0.8,
            status=PredictionStatus.EVALUATED,
            realized_return=0.04,
        )
        p2 = PredictionRecord(
            prediction_id="pred_pending_2",
            symbol="BTCUSDT",
            signal_action="BUY",
            confidence=0.75,
            status=PredictionStatus.PENDING,
        )
        prediction_ledger.record_prediction(p1)
        prediction_ledger.record_prediction(p2)

        # Record state values before shutdown
        saved_cash = portfolio_mgr.portfolio.cash
        saved_positions = dict(portfolio_mgr.portfolio.positions)
        saved_trade_count = len(portfolio_mgr.get_trade_ledger())
        saved_pred_count = len(prediction_ledger.get_all_predictions())

        # 2. Stop system & Save Checkpoint
        checkpoint_file = recovery.save_checkpoint(
            session=session,
            portfolio_mgr=portfolio_mgr,
            execution_engine=execution_engine,
            prediction_ledger=prediction_ledger,
        )
        assert checkpoint_file.exists()

        # 3. Simulate Restart into clean new instances
        new_portfolio_mgr = PaperPortfolioManager(initial_capital=10000.0)
        new_execution_engine = PaperExecutionEngine()
        new_prediction_ledger = PredictionLedger()

        restored_session = recovery.restore_session(
            checkpoint_file=checkpoint_file,
            portfolio_mgr=new_portfolio_mgr,
            execution_engine=new_execution_engine,
            prediction_ledger=new_prediction_ledger,
        )

        # 4. Invariant verifications
        # Session state recovered
        assert restored_session.session_id == session.session_id
        assert restored_session.model_versions == session.model_versions
        assert restored_session.feature_version == session.feature_version

        # Portfolio cash and positions recovered
        assert pytest.approx(new_portfolio_mgr.portfolio.cash, rel=1e-4) == saved_cash
        assert len(new_portfolio_mgr.portfolio.positions) == len(saved_positions)
        btc_pos = new_portfolio_mgr.portfolio.positions["BTCUSDT"]
        assert pytest.approx(btc_pos.quantity, rel=1e-4) == 0.05
        assert btc_pos.side == "LONG"

        # Trade ledger recovered without duplicates
        restored_trades = new_portfolio_mgr.get_trade_ledger()
        assert len(restored_trades) == saved_trade_count
        assert restored_trades[0].trade_id == trade.trade_id

        # Prediction ledger recovered with pending evaluations restored
        restored_preds = new_prediction_ledger.get_all_predictions()
        assert len(restored_preds) == saved_pred_count
        pending_preds = new_prediction_ledger.get_pending_predictions()
        assert len(pending_preds) == 1
        assert pending_preds[0].prediction_id == "pred_pending_2"

        # Idempotence: re-restoring the same checkpoint does NOT duplicate trades or predictions
        recovery.restore_session(
            checkpoint_file=checkpoint_file,
            portfolio_mgr=new_portfolio_mgr,
            execution_engine=new_execution_engine,
            prediction_ledger=new_prediction_ledger,
        )
        assert len(new_portfolio_mgr.get_trade_ledger()) == saved_trade_count
        assert len(new_prediction_ledger.get_all_predictions()) == saved_pred_count
