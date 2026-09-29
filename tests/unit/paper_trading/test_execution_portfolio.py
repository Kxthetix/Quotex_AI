"""Unit tests for PaperExecutionEngine and PaperPortfolioManager (Phase 7 Sections 16-20)."""

from datetime import UTC, datetime

import pytest

from paper_trading.execution import PaperExecutionConfig, PaperExecutionEngine
from paper_trading.models import (
    ExitReason,
    PaperFill,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperOrderType,
)
from paper_trading.portfolio import PaperPortfolioManager


class TestPaperExecutionAndPortfolio:
    """Verifies simulated fills, realistic frictions, and portfolio accounting."""

    def test_market_order_fill_with_slippage_and_fees(self):
        config = PaperExecutionConfig(
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
            default_spread_pct=0.0004,
            slippage_base_pct=0.0005,
        )
        engine = PaperExecutionEngine(config)

        order = engine.create_order(
            signal_id="sig_test_mkt",
            symbol="BTCUSDT",
            side=PaperOrderSide.BUY,
            requested_quantity=1.0,
            requested_price=50000.0,
        )

        fill = engine.simulate_execution(
            order=order,
            current_price=50000.0,
            volume=500.0,
            volatility=0.02,
        )

        assert fill is not None
        assert order.status == PaperOrderStatus.FILLED
        assert fill.quantity == 1.0
        # Buy price must be higher than current_price due to spread and slippage
        assert fill.execution_price > 50000.0
        assert fill.fees > 0.0
        assert fill.slippage > 0.0
        assert fill.spread_cost > 0.0
        assert fill.simulated_latency_ms > 0.0

    def test_limit_order_feasibility(self):
        engine = PaperExecutionEngine()

        # Buy limit below market: limit = 49,000, market = 50,000 -> Cannot fill
        order_unfilled = engine.create_order(
            signal_id="sig_limit_unfilled",
            symbol="BTCUSDT",
            side=PaperOrderSide.BUY,
            requested_quantity=0.5,
            order_type=PaperOrderType.LIMIT,
            limit_price=49000.0,
        )
        fill1 = engine.simulate_execution(order_unfilled, current_price=50000.0)
        assert fill1 is None
        assert order_unfilled.status == PaperOrderStatus.CREATED

        # Buy limit above market: limit = 50,100, market = 50,000 -> Fills!
        order_filled = engine.create_order(
            signal_id="sig_limit_filled",
            symbol="BTCUSDT",
            side=PaperOrderSide.BUY,
            requested_quantity=0.5,
            order_type=PaperOrderType.LIMIT,
            limit_price=50100.0,
        )
        fill2 = engine.simulate_execution(order_filled, current_price=50000.0)
        assert fill2 is not None
        assert order_filled.status == PaperOrderStatus.FILLED

    def test_partial_fill_liquidity_cap(self):
        config = PaperExecutionConfig(
            allow_partial_fills=True,
            max_bar_volume_pct=0.10,  # Max 10% of bar volume
        )
        engine = PaperExecutionEngine(config)

        # Request 5.0 BTC on a candle with only 10.0 BTC volume -> max fill = 1.0 BTC
        order = engine.create_order(
            signal_id="sig_partial",
            symbol="BTCUSDT",
            side=PaperOrderSide.BUY,
            requested_quantity=5.0,
            requested_price=50000.0,
        )

        fill = engine.simulate_execution(order, current_price=50000.0, volume=10.0)
        assert fill is not None
        assert fill.is_partial is True
        assert pytest.approx(fill.quantity, rel=1e-3) == 1.0
        assert order.status == PaperOrderStatus.PARTIALLY_FILLED
        assert pytest.approx(order.filled_quantity, rel=1e-3) == 1.0

    def test_portfolio_long_open_and_close_accounting(self):
        mgr = PaperPortfolioManager(initial_capital=10000.0)

        # 1. Open LONG: Buy 0.1 BTC at 50,000 (Cost = $5,000 + $5 fee)
        fill_buy = PaperFill(
            fill_id="f1",
            symbol="BTCUSDT",
            side=PaperOrderSide.BUY,
            quantity=0.1,
            execution_price=50000.0,
            fees=5.0,
            slippage=2.0,
        )
        trade1 = mgr.process_fill(fill_buy, signal_id="sig_1")
        assert trade1 is None  # Opening trade doesn't generate closed trade

        pos = mgr.get_position("BTCUSDT")
        assert pos.side == "LONG"
        assert pytest.approx(pos.quantity, rel=1e-3) == 0.1
        assert pytest.approx(mgr.portfolio.cash, rel=1e-3) == 10000.0 - (5000.0 + 5.0)

        # Mark price rises to 55,000
        mgr.update_market_price("BTCUSDT", 55000.0)
        # Unrealized PnL = 0.1 * (55000 - 50000) = $500
        assert pytest.approx(pos.unrealized_pnl, rel=1e-3) == 500.0
        # Total equity = cash ($4,995) + market value ($5,500) = $10,495
        assert pytest.approx(mgr.portfolio.total_equity, rel=1e-3) == 10495.0

        # 2. Close LONG: Sell 0.1 BTC at 55,000 (Proceeds = $5,500 - $5.50 fee)
        fill_sell = PaperFill(
            fill_id="f2",
            symbol="BTCUSDT",
            side=PaperOrderSide.SELL,
            quantity=0.1,
            execution_price=55000.0,
            fees=5.50,
            slippage=2.0,
        )
        closed_trade = mgr.process_fill(fill_sell, signal_id="sig_2", exit_reason=ExitReason.TAKE_PROFIT)

        assert closed_trade is not None
        assert closed_trade.symbol == "BTCUSDT"
        assert closed_trade.side == "LONG"
        # Gross PnL = $500, Total Fees = $10.50 -> Net PnL = $489.50
        assert pytest.approx(closed_trade.pnl, rel=1e-2) == 489.50
        assert pos.side == "FLAT"
        assert pos.quantity == 0.0

        # Cash balance is now starting $10,000 + Net PnL $489.50 = $10,489.50
        assert pytest.approx(mgr.portfolio.cash, rel=1e-2) == 10489.50
        assert pytest.approx(mgr.portfolio.total_equity, rel=1e-2) == 10489.50
        assert len(mgr.get_trade_ledger()) == 1
