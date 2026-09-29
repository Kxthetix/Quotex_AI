"""Execution Provider Abstraction and Simulated Broker (Section 28).

Defines the ExecutionProvider contract and provides a high-fidelity SimulatedExecutionProvider
for paper, backtest, and shadow execution.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from execution.models import (
    AccountState,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    Position,
)
from execution.state_machine import OrderStateMachine


class ExecutionProvider(ABC):
    """Abstract provider interface for order placement, querying, and balance reporting."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order for execution."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> Order:
        """Request cancellation of an active order."""
        pass

    @abstractmethod
    def query_order(self, order_id: str) -> Order | None:
        """Query state of an order."""
        pass

    @abstractmethod
    def query_position(self, symbol: str) -> Position:
        """Query current holding in a symbol."""
        pass

    @abstractmethod
    def query_account_state(self) -> AccountState:
        """Retrieve total equity, cash balance, and all positions."""
        pass


class SimulatedExecutionProvider(ExecutionProvider):
    """In-memory broker simulation for paper trading and backtesting."""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        fee_rate_bps: float = 5.0,
        slippage_bps: float = 2.0,
    ) -> None:
        self._initial_cash = initial_cash
        self.fee_rate_bps = fee_rate_bps
        self.slippage_bps = slippage_bps
        self._cash = initial_cash
        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}  # symbol -> Position
        self._connected = True
        self._lock = threading.Lock()

    @property
    def provider_name(self) -> str:
        return "simulated_broker"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def set_connected(self, connected: bool) -> None:
        self._connected = connected

    def submit_order(self, order: Order) -> Order:
        """Simulate immediate fill for market orders or register limit orders."""
        with self._lock:
            if not self._connected:
                OrderStateMachine.transition(order, OrderState.REJECTED, reason="Broker disconnected")
                self._orders[order.order_id] = order
                return order

            # 1. State: CREATED -> VALIDATED
            OrderStateMachine.transition(order, OrderState.VALIDATED)

            # 2. State: VALIDATED -> SUBMITTED
            OrderStateMachine.transition(order, OrderState.SUBMITTED)
            order.provider_order_id = f"sim_prov_{order.order_id[-8:]}"

            # 3. State: SUBMITTED -> ACKNOWLEDGED
            OrderStateMachine.transition(order, OrderState.ACKNOWLEDGED)
            self._orders[order.order_id] = order

            # For simulated market orders, execute immediately
            if order.order_type == OrderType.MARKET:
                self._execute_fill(order)

            return order

    def _execute_fill(self, order: Order) -> None:
        """Compute execution fill, fee, slippage, and update positions/cash."""
        fill_price = order.price if order.price and order.price > 0 else 50000.0

        # Apply slippage
        slip_mult = (self.slippage_bps / 10000.0)
        if order.side == OrderSide.BUY:
            effective_price = fill_price * (1.0 + slip_mult)
        else:
            effective_price = fill_price * (1.0 - slip_mult)

        notional = effective_price * order.quantity
        fee = notional * (self.fee_rate_bps / 10000.0)

        # Update order
        order.filled_quantity = order.quantity
        order.avg_fill_price = round(effective_price, 4)
        order.fees = round(fee, 4)
        order.slippage_bps = self.slippage_bps

        # Update cash and position
        sym = order.symbol.upper()
        pos = self._positions.get(sym, Position(symbol=sym, asset_id=order.asset_id, quantity=0.0))

        if order.side == OrderSide.BUY:
            self._cash -= (notional + fee)
            new_qty = pos.quantity + order.quantity
            pos.avg_entry_price = (
                ((pos.quantity * pos.avg_entry_price) + notional) / new_qty
                if new_qty > 0
                else effective_price
            )
            pos.quantity = new_qty
        else:
            self._cash += (notional - fee)
            pos.quantity -= order.quantity

        pos.updated_at = datetime.now(UTC)
        self._positions[sym] = pos

        # Transition to FILLED
        OrderStateMachine.transition(order, OrderState.FILLED)

    def cancel_order(self, order_id: str) -> Order:
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                raise KeyError(f"Order {order_id} not found.")

            if order.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED):
                return order

            OrderStateMachine.transition(order, OrderState.CANCEL_REQUESTED)
            OrderStateMachine.transition(order, OrderState.CANCELLED)
            return order

    def query_order(self, order_id: str) -> Order | None:
        with self._lock:
            return self._orders.get(order_id)

    def query_position(self, symbol: str) -> Position:
        with self._lock:
            sym = symbol.upper()
            return self._positions.get(sym, Position(symbol=sym, asset_id="", quantity=0.0))

    def query_account_state(self) -> AccountState:
        with self._lock:
            # Equity = cash + value of positions
            pos_equity = sum(p.quantity * p.avg_entry_price for p in self._positions.values())
            total_eq = self._cash + pos_equity
            return AccountState(
                account_id="sim_account_01",
                total_equity=round(total_eq, 2),
                cash=round(self._cash, 2),
                used_margin=0.0,
                positions={k: Position(**p.to_dict()) for k, p in self._positions.items()},
            )
