"""Execution Simulator coordinating spread, slippage, latency, fees, and partial fills (Phase 5).

Converts orders into one or more execution Fills based on realistic market microstructure constraints.
"""

from datetime import datetime

from backtesting.execution.fees import FeeModel, PercentageFeeModel
from backtesting.execution.latency import LatencyModel, ZeroLatencyModel
from backtesting.execution.slippage import PercentageSlippageModel, SlippageModel
from backtesting.execution.spread import PercentageSpreadModel, SpreadModel
from backtesting.models import Fill, Order, OrderSide, OrderStatus, OrderType
from data.ingestion.models import Candle


class ExecutionSimulator:
    """Simulates realistic exchange order execution with friction and liquidity constraints."""

    def __init__(
        self,
        spread_model: SpreadModel | None = None,
        slippage_model: SlippageModel | None = None,
        fee_model: FeeModel | None = None,
        latency_model: LatencyModel | None = None,
        max_bar_volume_pct: float = 0.10,
        allow_partial_fills: bool = True,
    ) -> None:
        self.spread_model = spread_model or PercentageSpreadModel()
        self.slippage_model = slippage_model or PercentageSlippageModel()
        self.fee_model = fee_model or PercentageFeeModel()
        self.latency_model = latency_model or ZeroLatencyModel()
        self.max_bar_volume_pct = max_bar_volume_pct
        self.allow_partial_fills = allow_partial_fills

    def process_order(
        self,
        order: Order,
        candle: Candle,
        current_time: datetime,
        volatility: float = 0.0,
    ) -> Fill | None:
        """Attempt execution of an order against the current market candle.

        Returns a Fill object if execution occurred, or None if conditions were not met.
        """
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            return None

        # 1. Check Latency: Order cannot execute before submitted_at + latency
        exec_eligible_time = self.latency_model.calculate_execution_time(
            order.submitted_at or current_time
        )
        if current_time < exec_eligible_time:
            # Latency delay has not elapsed yet
            return None

        is_buy = order.side == OrderSide.BUY

        # 2. Check Order Type Feasibility against candle range
        # Signals are formed from the previous completed candle. For market orders,
        # use this bar's open as the earliest observable execution price.
        ref_price = candle.open
        is_maker = False

        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                order.status = OrderStatus.REJECTED
                return None
            if is_buy:
                # Buy limit executes only if market traded at or below limit price
                if candle.low > order.limit_price:
                    return None
                ref_price = min(order.limit_price, candle.open)
                is_maker = True
            else:
                # Sell limit executes only if market traded at or above limit price
                if candle.high < order.limit_price:
                    return None
                ref_price = max(order.limit_price, candle.open)
                is_maker = True

        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                order.status = OrderStatus.REJECTED
                return None
            if is_buy:
                if candle.high < order.stop_price:
                    return None
                ref_price = max(order.stop_price, candle.open)
            else:
                if candle.low > order.stop_price:
                    return None
                ref_price = min(order.stop_price, candle.open)

        # 3. Determine Fill Quantity (Liquidity & Partial Fill Model)
        needed_qty = order.remaining_qty
        if needed_qty <= 0.0:
            order.status = OrderStatus.FILLED
            return None

        bar_vol = candle.volume
        # Max fill capacity based on bar volume limit
        if self.allow_partial_fills and bar_vol > 0:
            max_fillable = max(0.0001, bar_vol * self.max_bar_volume_pct)
            fill_qty = min(needed_qty, max_fillable)
        else:
            fill_qty = needed_qty

        is_partial = fill_qty < needed_qty

        # 4. Apply Bid/Ask Spread
        price_after_spread = self.spread_model.get_execution_price(ref_price, is_buy=is_buy)
        spread_cost = self.spread_model.get_spread_cost(ref_price, fill_qty)

        # 5. Apply Slippage (adverse execution movement)
        final_price, slippage_cost = self.slippage_model.calculate_slippage(
            base_price=price_after_spread,
            quantity=fill_qty,
            is_buy=is_buy,
            volatility=volatility,
        )

        # 6. Apply Transaction Fee
        fee = self.fee_model.calculate_fee(
            price=final_price,
            quantity=fill_qty,
            is_maker=is_maker,
        )

        # 7. Update Order State
        total_filled_prev = order.filled_qty
        total_qty_now = total_filled_prev + fill_qty
        # Update volume-weighted average fill price
        order.avg_fill_price = (
            (order.avg_fill_price * total_filled_prev) + (final_price * fill_qty)
        ) / total_qty_now

        order.filled_qty = total_qty_now
        order.remaining_qty = max(0.0, order.quantity - total_qty_now)
        order.cumulative_fees += fee
        order.cumulative_slippage += slippage_cost
        order.executed_at = current_time

        if order.remaining_qty <= 1e-7:
            order.status = OrderStatus.FILLED
            order.remaining_qty = 0.0
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        # 8. Create Fill Event
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            timestamp=current_time,
            price=final_price,
            quantity=fill_qty,
            fee=fee,
            slippage_cost=slippage_cost,
            spread_cost=spread_cost,
            is_partial=is_partial,
        )
