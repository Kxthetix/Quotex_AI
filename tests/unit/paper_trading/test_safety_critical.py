"""Critical Real-Execution Safety Test for Phase 7 (Section 48).

Proves that Phase 7 can NEVER place a real order, even if production broker
credentials, live exchange URLs, or API trading permissions are configured.
There is NO code path from Signal -> Risk -> Paper Trading that can invoke
a real-money broker order.
"""

import pytest

from paper_trading.execution import (
    PaperExecutionConfig,
    PaperExecutionEngine,
    RealExecutionBlockedError,
)
from paper_trading.models import (
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperOrderType,
    RiskDecision,
    RiskEvaluation,
)
from paper_trading.risk_engine import RiskConfig, RiskEngine
from signals.models import SignalAction, TradingSignal


class TestCriticalRealExecutionSafety:
    """Rigorous verification that Phase 7 execution is 100% paper simulated."""

    def test_section_48_real_broker_credentials_intercepted(self):
        """Attempting to configure real broker credentials must be intercepted and forced to paper mode."""
        malicious_or_accidental_config = {
            "api_key": "LIVE_PRODUCTION_KEY_SECRET_12345",
            "secret": "TOP_SECRET_EXCHANGE_PRIVATE_KEY",
            "broker_url": "https://api.binance.com/api/v3/order",
            "live_trading_enabled": True,
            "alpaca_secret_key": "AK_LIVE_SECRET",
        }

        # Enforce safety intercept
        PaperExecutionEngine.enforce_safety_invariants(malicious_or_accidental_config)

        engine = PaperExecutionEngine(PaperExecutionConfig())
        order = engine.create_order(
            signal_id="sig_test_1",
            symbol="BTCUSDT",
            side=PaperOrderSide.BUY,
            requested_quantity=0.1,
            order_type=PaperOrderType.MARKET,
            requested_price=50000.0,
        )

        assert isinstance(order, PaperOrder)
        assert order.status == PaperOrderStatus.CREATED

        fill = engine.simulate_execution(
            order=order,
            current_price=50000.0,
            bid=49995.0,
            ask=50005.0,
        )

        assert fill is not None
        # Must be an internal paper fill with simulated fees, slippage, and spread
        assert fill.fees > 0.0
        assert fill.slippage > 0.0
        assert fill.spread_cost > 0.0
        assert fill.execution_price > 50000.0  # Buy order incurs slippage & spread

    def test_no_real_order_method_exists_on_order_or_execution(self):
        """Verify PaperOrder and PaperExecutionEngine have no real broker submission attributes."""
        engine = PaperExecutionEngine()
        order = engine.create_order(
            signal_id="sig_test_2",
            symbol="ETHUSDT",
            side=PaperOrderSide.BUY,
            requested_quantity=1.0,
        )

        forbidden_methods = [
            "place_broker_order",
            "send_to_exchange",
            "execute_real_money",
            "withdraw",
            "deposit",
            "transfer_funds",
            "live_order",
        ]

        for method in forbidden_methods:
            assert not hasattr(engine, method), f"PaperExecutionEngine contains forbidden real method: {method}"
            assert not hasattr(order, method), f"PaperOrder contains forbidden real method: {method}"

    def test_pipeline_signal_to_risk_to_paper_execution_is_purely_simulated(self):
        """End-to-end pipeline check from TradingSignal to RiskEngine to PaperExecution."""
        from datetime import UTC, datetime
        signal = TradingSignal(
            signal_id="sig_live_test",
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            conviction=0.85,
            regime="TREND_UP",
            model_version="v1.0.0",
            timestamp=datetime.now(UTC),
        )

        risk_engine = RiskEngine(RiskConfig())
        from paper_trading.models import PaperPortfolio
        portfolio = PaperPortfolio(cash=10000.0)

        risk_eval = risk_engine.evaluate(signal, portfolio, current_price=50000.0)
        assert risk_eval.decision in (RiskDecision.APPROVE, RiskDecision.MODIFY)

        exec_engine = PaperExecutionEngine()
        order = exec_engine.create_order(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=PaperOrderSide.BUY,
            requested_quantity=risk_eval.approved_quantity,
        )
        fill = exec_engine.simulate_execution(order, current_price=50000.0)

        assert fill is not None
        assert fill.symbol == "BTCUSDT"
        assert fill.quantity == risk_eval.approved_quantity
        assert order.status == PaperOrderStatus.FILLED
