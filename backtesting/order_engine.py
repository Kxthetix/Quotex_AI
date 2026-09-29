"""Order Engine for Phase 5.

Manages order creation, lifecycle state transitions, order tracking, and validation.
"""

from datetime import datetime

from backtesting.models import Order, OrderSide, OrderStatus, OrderType


class OrderEngine:
    """Manages active, pending, and completed orders with explicit lifecycle tracking."""

    def __init__(self) -> None:
        self.active_orders: list[Order] = []
        self.order_history: list[Order] = []
        self._order_counter: int = 0

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        timestamp: datetime | None = None,
        metadata: dict | None = None,
    ) -> Order:
        """Create and validate a new order."""
        self._order_counter += 1
        order = Order(
            order_id=f"ord_{self._order_counter:04d}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            created_at=timestamp or datetime.now(),
            metadata=metadata or {},
        )

        # Pre-validation
        if quantity <= 1e-8:
            order.status = OrderStatus.REJECTED
            self.order_history.append(order)
            return order

        order.status = OrderStatus.SUBMITTED
        order.submitted_at = order.created_at
        self.active_orders.append(order)
        return order

    def cancel_order(self, order_id: str, timestamp: datetime) -> bool:
        """Cancel an active order."""
        for i, ord in enumerate(self.active_orders):
            if ord.order_id == order_id:
                ord.status = OrderStatus.CANCELLED
                self.active_orders.pop(i)
                self.order_history.append(ord)
                return True
        return False

    def update_order_status(self, order: Order) -> None:
        """Move filled or closed orders from active list to history."""
        if order.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ):
            if order in self.active_orders:
                self.active_orders.remove(order)
            if order not in self.order_history:
                self.order_history.append(order)
