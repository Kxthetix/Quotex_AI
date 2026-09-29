"""Unit tests for the Independent Risk Engine (Phase 7 Section 15, 29)."""

from datetime import UTC, datetime

import pytest

from paper_trading.models import PaperPortfolio, PaperPosition, RiskDecision
from paper_trading.risk_engine import CircuitBreakerState, RiskConfig, RiskEngine
from signals.models import SignalAction, TradingSignal


class TestRiskEngine:
    """Tests the independent risk evaluation layer."""

    def test_approve_signal_within_limits(self):
        config = RiskConfig(
            max_position_pct=0.10,
            max_total_exposure_pct=0.50,
            min_signal_conviction=0.10,
        )
        engine = RiskEngine(config)
        portfolio = PaperPortfolio(cash=10000.0, peak_equity=10000.0)

        signal = TradingSignal(
            signal_id="sig_test_approve",
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            conviction=0.75,
            timestamp=datetime.now(UTC),
        )

        eval_res = engine.evaluate(signal, portfolio, current_price=50000.0)
        assert eval_res.decision == RiskDecision.APPROVE
        # Max position value = 10000 * 0.10 = 1000 -> 1000 / 50000 = 0.02 BTC
        assert pytest.approx(eval_res.approved_quantity, rel=1e-3) == 0.02

    def test_veto_signal_low_conviction(self):
        config = RiskConfig(min_signal_conviction=0.30)
        engine = RiskEngine(config)
        portfolio = PaperPortfolio(cash=10000.0)

        signal = TradingSignal(
            signal_id="sig_low_conv",
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            conviction=0.15,
            timestamp=datetime.now(UTC),
        )

        eval_res = engine.evaluate(signal, portfolio, current_price=50000.0)
        assert eval_res.decision == RiskDecision.VETO
        assert "below minimum" in eval_res.reason

    def test_veto_signal_circuit_breaker_active(self):
        engine = RiskEngine()
        portfolio = PaperPortfolio(cash=10000.0)
        engine.breaker.trip("Manual emergency trip")

        signal = TradingSignal(
            signal_id="sig_veto_cb",
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            conviction=0.90,
            timestamp=datetime.now(UTC),
        )

        eval_res = engine.evaluate(signal, portfolio, current_price=50000.0)
        assert eval_res.decision == RiskDecision.VETO
        assert "Circuit breaker active" in eval_res.reason

    def test_modify_size_when_exposure_limit_near(self):
        config = RiskConfig(
            max_position_pct=0.20,
            max_total_exposure_pct=0.25,
            min_signal_conviction=0.10,
        )
        engine = RiskEngine(config)
        portfolio = PaperPortfolio(cash=8000.0, peak_equity=10000.0)
        # Existing position has $2,000 exposure out of $2,500 max allowed
        portfolio.positions["ETHUSDT"] = PaperPosition(
            symbol="ETHUSDT",
            side="LONG",
            quantity=1.0,
            current_price=2000.0,
            avg_entry_price=2000.0,
        )

        signal = TradingSignal(
            signal_id="sig_modify",
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            conviction=0.80,
            timestamp=datetime.now(UTC),
        )

        # Remaining capacity is $500, but max requested position would be $2000
        # At $50,000, $500 capacity is 0.01 BTC instead of 0.04 BTC
        eval_res = engine.evaluate(signal, portfolio, current_price=50000.0)
        assert eval_res.decision == RiskDecision.MODIFY
        assert pytest.approx(eval_res.approved_quantity, rel=1e-3) == 0.01

    def test_drawdown_circuit_breaker_trips(self):
        config = RiskConfig(max_drawdown_pct=0.10)
        engine = RiskEngine(config)
        # Peak equity 10,000, current cash 8,500 -> 15% drawdown
        portfolio = PaperPortfolio(cash=8500.0, peak_equity=10000.0)

        signal = TradingSignal(
            signal_id="sig_dd",
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            conviction=0.80,
            timestamp=datetime.now(UTC),
        )

        eval_res = engine.evaluate(signal, portfolio, current_price=50000.0)
        assert eval_res.decision == RiskDecision.VETO
        assert engine.breaker.is_tripped is True
        assert "Drawdown" in engine.breaker.reason

    def test_risk_reducing_order_is_approved_even_under_drawdown(self):
        config = RiskConfig(max_drawdown_pct=0.10)
        engine = RiskEngine(config)
        portfolio = PaperPortfolio(cash=8500.0, peak_equity=10000.0)
        # Open LONG BTC position
        portfolio.positions["BTCUSDT"] = PaperPosition(
            symbol="BTCUSDT",
            side="LONG",
            quantity=0.1,
            avg_entry_price=50000.0,
            current_price=45000.0,
        )

        # SELL signal to close or reduce the losing LONG position
        signal = TradingSignal(
            signal_id="sig_close_pos",
            symbol="BTCUSDT",
            action=SignalAction.SELL,
            conviction=0.50,
            timestamp=datetime.now(UTC),
        )

        eval_res = engine.evaluate(signal, portfolio, current_price=45000.0)
        assert eval_res.decision == RiskDecision.APPROVE
        assert "Risk-reducing order approved" in eval_res.reason
