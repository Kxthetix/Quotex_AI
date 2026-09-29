"""Unit tests for Position Engine, Portfolio Accounting, and Position Sizing (Phase 5).

Covers:
- Position lifecycle (Open, Increase, Reduce, Close)
- Weighted-average entry price accounting
- Trade record generation with fee/slippage/spread attribution
- Portfolio cash and equity balance invariants
- Gross and net exposure tracking
- Position sizing algorithms and pre-trade limit enforcement
"""

from datetime import UTC, datetime, timedelta

from backtesting.models import (
    BacktestConfig,
    ExitReason,
    Fill,
    OrderSide,
    PositionSide,
    SizingMethod,
)
from backtesting.portfolio import PortfolioEngine
from backtesting.position import PositionEngine
from backtesting.sizing import PositionSizer


class TestPositionEngine:
    def test_open_and_close_long_position(self):
        pe = PositionEngine(symbol="BTC/USDT")
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # 1. Open Long 1.0 @ 100.0
        fill_entry = Fill(
            order_id="o1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            timestamp=t0,
            price=100.0,
            quantity=1.0,
            fee=0.05,
            slippage_cost=0.03,
            spread_cost=0.02,
        )
        trade = pe.apply_fill(fill_entry)
        assert trade is None
        assert pe.position.side == PositionSide.LONG
        assert pe.position.quantity == 1.0
        assert pe.position.avg_entry_price == 100.0

        # Update mark price to 110.0
        pe.update_market_price(110.0, timestamp=t0 + timedelta(minutes=1))
        assert pe.position.unrealized_pnl == 10.0
        assert pe.position.market_value == 110.0

        # 2. Close Long 1.0 @ 110.0
        fill_exit = Fill(
            order_id="o2",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            timestamp=t0 + timedelta(minutes=5),
            price=110.0,
            quantity=1.0,
            fee=0.055,
            slippage_cost=0.03,
            spread_cost=0.02,
        )
        trade = pe.apply_fill(fill_exit, exit_reason=ExitReason.SIGNAL)
        assert trade is not None
        assert trade.quantity == 1.0
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert round(trade.gross_pnl, 4) == 10.0000
        # Total fees = entry fee (0.05) + exit fee (0.055) = 0.105
        assert round(trade.fees, 4) == 0.1050
        assert round(trade.net_pnl, 4) == 9.8950
        assert trade.holding_period_seconds == 300.0
        assert trade.exit_reason == ExitReason.SIGNAL

        # Position should now be flat
        assert pe.position.side == PositionSide.FLAT
        assert pe.position.quantity == 0.0
        assert pe.position.avg_entry_price == 0.0

    def test_average_entry_price_on_scale_in(self):
        pe = PositionEngine(symbol="BTC/USDT")
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Buy 1.0 @ 100
        pe.apply_fill(
            Fill(
                order_id="o1",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                timestamp=t0,
                price=100.0,
                quantity=1.0,
            )
        )
        # Buy another 1.0 @ 110
        pe.apply_fill(
            Fill(
                order_id="o2",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                timestamp=t0 + timedelta(minutes=1),
                price=110.0,
                quantity=1.0,
            )
        )

        assert pe.position.quantity == 2.0
        # Average price = (100*1 + 110*1) / 2 = 105.0
        assert round(pe.position.avg_entry_price, 4) == 105.0000


class TestPortfolioEngine:
    def test_cash_and_equity_conservation(self):
        initial_cap = 10000.0
        pf = PortfolioEngine(initial_capital=initial_cap)
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

        # Buy 1 BTC @ 100 with 0.10 fee
        fill_buy = Fill(
            order_id="o1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            timestamp=t0,
            price=100.0,
            quantity=1.0,
            fee=0.10,
        )
        pf.process_fill(fill_buy)
        assert round(pf.cash, 4) == round(initial_cap - 100.10, 4)
        assert round(pf.cumulative_fees, 4) == 0.10

        # Valuation at mark price = 100.0
        pt = pf.update_valuation({"BTC/USDT": 100.0}, t0)
        assert round(pt.market_value, 4) == 100.0
        # Equity = Cash + Market Value = (10000 - 100.10) + 100.0 = 9999.90
        assert round(pt.equity, 4) == 9999.90
        assert round(pt.gross_exposure, 4) == 100.0
        assert round(pt.net_exposure, 4) == 100.0

        # Mark price rises to 150.0
        pt2 = pf.update_valuation({"BTC/USDT": 150.0}, t0 + timedelta(minutes=1))
        assert round(pt2.market_value, 4) == 150.0
        assert round(pt2.equity, 4) == round(9999.90 + 50.0, 4)
        assert round(pt2.unrealized_pnl, 4) == 50.0

        # Sell 1 BTC @ 150 with 0.15 fee
        fill_sell = Fill(
            order_id="o2",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            timestamp=t0 + timedelta(minutes=2),
            price=150.0,
            quantity=1.0,
            fee=0.15,
        )
        pf.process_fill(fill_sell)
        pt3 = pf.update_valuation({"BTC/USDT": 150.0}, t0 + timedelta(minutes=2))
        assert round(pt3.market_value, 4) == 0.0
        assert round(pt3.gross_exposure, 4) == 0.0
        # Realized net P&L = (150 - 100) - (0.10 + 0.15) = 49.75
        assert round(pf.cumulative_realized_pnl, 4) == 49.75
        assert round(pt3.equity, 4) == round(initial_cap + 49.75, 4)
        assert round(pf.cash, 4) == round(initial_cap + 49.75, 4)


class TestPositionSizing:
    def test_percent_capital_sizing(self):
        cfg = BacktestConfig(
            sizing_method=SizingMethod.PERCENT_CAPITAL,
            sizing_value=0.10,  # 10% of equity
            max_position_size=10.0,
            max_notional=50000.0,
        )
        sizer = PositionSizer(cfg)
        # Equity = $10,000, 10% = $1,000, Price = $100 -> 10.0 units
        qty = sizer.calculate_order_quantity(
            current_equity=10000.0,
            current_cash=10000.0,
            current_price=100.0,
            current_position_qty=0.0,
        )
        assert round(qty, 4) == 10.0

    def test_enforce_max_position_limit(self):
        cfg = BacktestConfig(
            sizing_method=SizingMethod.PERCENT_CAPITAL,
            sizing_value=0.50,  # 50% = $5,000 / $100 = 50 units
            max_position_size=5.0,  # Limit is 5 units!
        )
        sizer = PositionSizer(cfg)
        qty = sizer.calculate_order_quantity(
            current_equity=10000.0,
            current_cash=10000.0,
            current_price=100.0,
            current_position_qty=0.0,
        )
        assert qty == 5.0

    def test_enforce_max_notional_limit(self):
        cfg = BacktestConfig(
            sizing_method=SizingMethod.FIXED_QUANTITY,
            sizing_value=100.0,  # Wants 100 units
            max_notional=1000.0,  # But max notional is $1,000
            max_position_size=1000.0,
        )
        sizer = PositionSizer(cfg)
        # At price 100, max notional $1,000 allows only 10 units
        qty = sizer.calculate_order_quantity(
            current_equity=10000.0,
            current_cash=10000.0,
            current_price=100.0,
            current_position_qty=0.0,
        )
        assert qty == 10.0
