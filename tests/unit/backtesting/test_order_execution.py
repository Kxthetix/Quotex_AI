"""Unit tests for Order Engine and Execution Simulator (Phase 5).

Covers:
- Market, Limit, and Stop order types
- Order lifecycle states and transitions
- Order cancellation and rejection
- Bid/Ask spread application
- Deterministic slippage modeling
- Fee calculation (maker vs taker)
- Latency execution delay
- Partial fills and liquidity limits
"""

from datetime import UTC, datetime, timedelta

from backtesting.execution import (
    ExecutionSimulator,
    FixedLatencyModel,
    FixedSpreadModel,
    PercentageFeeModel,
    PercentageSlippageModel,
    PercentageSpreadModel,
    ZeroLatencyModel,
)
from backtesting.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from backtesting.order_engine import OrderEngine
from data.ingestion.models import Candle


def make_test_candle(
    open_p: float = 100.0,
    high_p: float = 105.0,
    low_p: float = 95.0,
    close_p: float = 102.0,
    volume: float = 1000.0,
    timestamp: datetime | None = None,
) -> Candle:
    ts = timestamp or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    return Candle(
        symbol="BTC/USDT",
        timeframe="1m",
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        volume=volume,
        start_time=ts - timedelta(minutes=1),
        end_time=ts,
        trades_count=50,
        is_closed=True,
    )


class TestOrderEngine:
    def test_create_market_order(self):
        engine = OrderEngine()
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        order = engine.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.5,
            order_type=OrderType.MARKET,
            timestamp=t0,
        )
        assert order.status == OrderStatus.SUBMITTED
        assert order.quantity == 1.5
        assert order.remaining_qty == 1.5
        assert order.filled_qty == 0.0
        assert order in engine.active_orders

    def test_reject_zero_or_negative_quantity(self):
        engine = OrderEngine()
        order = engine.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.0,
        )
        assert order.status == OrderStatus.REJECTED
        assert order not in engine.active_orders
        assert order in engine.order_history

    def test_cancel_active_order(self):
        engine = OrderEngine()
        t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        order = engine.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            timestamp=t0,
        )
        assert len(engine.active_orders) == 1
        cancelled = engine.cancel_order(order.order_id, timestamp=t0 + timedelta(seconds=5))
        assert cancelled is True
        assert len(engine.active_orders) == 0
        assert order.status == OrderStatus.CANCELLED
        assert order in engine.order_history


class TestSpreadAndSlippage:
    def test_percentage_spread_model(self):
        # 20 bps full spread = 0.0020 -> half spread = 0.0010
        model = PercentageSpreadModel(spread_pct=0.002)
        bid = model.get_bid_price(100.0)
        ask = model.get_ask_price(100.0)
        assert round(bid, 4) == 99.9000
        assert round(ask, 4) == 100.1000

        # Buy executes at ask, sell executes at bid
        buy_p = model.get_execution_price(100.0, is_buy=True)
        sell_p = model.get_execution_price(100.0, is_buy=False)
        assert round(buy_p, 4) == 100.1000
        assert round(sell_p, 4) == 99.9000

        # Spread cost on 2 units = 2 * 0.10 = 0.20
        assert round(model.get_spread_cost(100.0, 2.0), 4) == 0.2000

    def test_fixed_spread_model(self):
        model = FixedSpreadModel(fixed_spread=1.0)
        assert model.get_bid_price(100.0) == 99.5
        assert model.get_ask_price(100.0) == 100.5

    def test_percentage_slippage_model(self):
        # 5 bps slippage = 0.0005
        model = PercentageSlippageModel(base_slippage_pct=0.0005)
        # Buy slips upward
        buy_p, slip_cost = model.calculate_slippage(100.0, quantity=2.0, is_buy=True)
        assert buy_p > 100.0
        assert round(buy_p, 4) == 100.0500
        assert round(slip_cost, 4) == 0.1000

        # Sell slips downward
        sell_p, slip_cost = model.calculate_slippage(100.0, quantity=2.0, is_buy=False)
        assert sell_p < 100.0
        assert round(sell_p, 4) == 99.9500
        assert round(slip_cost, 4) == 0.1000


class TestFeeModel:
    def test_percentage_fee_model(self):
        # 2 bps maker (0.0002), 5 bps taker (0.0005), 1.0 fixed
        fee_model = PercentageFeeModel(maker_fee=0.0002, taker_fee=0.0005, fixed_fee=1.0)
        # Taker fee on $10,000 notional: 10000 * 0.0005 + 1.0 = 5.0 + 1.0 = 6.0
        taker_fee = fee_model.calculate_fee(price=100.0, quantity=100.0, is_maker=False)
        assert round(taker_fee, 4) == 6.0000

        # Maker fee on $10,000 notional: 10000 * 0.0002 + 1.0 = 2.0 + 1.0 = 3.0
        maker_fee = fee_model.calculate_fee(price=100.0, quantity=100.0, is_maker=True)
        assert round(maker_fee, 4) == 3.0000


class TestExecutionSimulator:
    def test_market_order_fill(self):
        sim = ExecutionSimulator(
            spread_model=PercentageSpreadModel(spread_pct=0.001),
            slippage_model=PercentageSlippageModel(base_slippage_pct=0.0005),
            fee_model=PercentageFeeModel(taker_fee=0.001),
            latency_model=ZeroLatencyModel(),
        )
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        candle = make_test_candle(
            open_p=100.0, high_p=105.0, low_p=95.0, close_p=100.0, volume=1000.0, timestamp=t0
        )
        order = Order(
            order_id="o1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            submitted_at=t0,
        )

        fill = sim.process_order(order, candle, current_time=t0)
        assert fill is not None
        assert fill.quantity == 1.0
        assert fill.side == OrderSide.BUY
        # Price should reflect spread (100.05) + slippage (approx 100.10)
        assert fill.price > 100.0
        assert fill.fee > 0.0
        assert order.status == OrderStatus.FILLED
        assert order.remaining_qty == 0.0

    def test_limit_order_execution(self):
        sim = ExecutionSimulator(latency_model=ZeroLatencyModel())
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        candle = make_test_candle(
            open_p=100.0, high_p=105.0, low_p=95.0, close_p=102.0, timestamp=t0
        )

        # Buy limit at 96.0 should execute because candle.low (95.0) <= 96.0
        order_buy_filled = Order(
            order_id="o_limit_buy",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=96.0,
            submitted_at=t0,
        )
        fill = sim.process_order(order_buy_filled, candle, current_time=t0)
        assert fill is not None
        assert order_buy_filled.status == OrderStatus.FILLED

        # Buy limit at 90.0 should NOT execute because candle.low (95.0) > 90.0
        order_buy_unfilled = Order(
            order_id="o_limit_buy_2",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=90.0,
            submitted_at=t0,
        )
        fill2 = sim.process_order(order_buy_unfilled, candle, current_time=t0)
        assert fill2 is None
        assert (
            order_buy_unfilled.status == OrderStatus.CREATED
            or order_buy_unfilled.status == OrderStatus.SUBMITTED
        )

    def test_stop_order_execution(self):
        sim = ExecutionSimulator(latency_model=ZeroLatencyModel())
        t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        candle = make_test_candle(
            open_p=100.0, high_p=105.0, low_p=95.0, close_p=102.0, timestamp=t0
        )

        # Stop buy at 104.0 executes because candle.high (105.0) >= 104.0
        order_stop = Order(
            order_id="o_stop_buy",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=1.0,
            stop_price=104.0,
            submitted_at=t0,
        )
        fill = sim.process_order(order_stop, candle, current_time=t0)
        assert fill is not None
        assert order_stop.status == OrderStatus.FILLED

    def test_latency_prevents_premature_fill(self):
        # 5-second latency delay
        sim = ExecutionSimulator(latency_model=FixedLatencyModel(latency_seconds=5.0))
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        candle = make_test_candle(timestamp=t0)
        order = Order(
            order_id="o_lat",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            submitted_at=t0,
        )

        # At t0 + 2s, should NOT execute
        fill_early = sim.process_order(order, candle, current_time=t0 + timedelta(seconds=2))
        assert fill_early is None
        assert order.status != OrderStatus.FILLED

        # At t0 + 5s, should execute
        fill_ontime = sim.process_order(order, candle, current_time=t0 + timedelta(seconds=5))
        assert fill_ontime is not None
        assert order.status == OrderStatus.FILLED

    def test_partial_fill_liquidity_cap(self):
        # Max 10% of bar volume per fill. Bar volume = 100.0 -> max fill = 10.0
        sim = ExecutionSimulator(
            latency_model=ZeroLatencyModel(),
            max_bar_volume_pct=0.10,
            allow_partial_fills=True,
        )
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        candle = make_test_candle(volume=100.0, timestamp=t0)
        order = Order(
            order_id="o_large",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=25.0,  # Requests 25.0, but capacity is 10.0
            submitted_at=t0,
        )

        fill = sim.process_order(order, candle, current_time=t0)
        assert fill is not None
        assert fill.is_partial is True
        assert round(fill.quantity, 4) == 10.0
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert round(order.filled_qty, 4) == 10.0
        assert round(order.remaining_qty, 4) == 15.0
