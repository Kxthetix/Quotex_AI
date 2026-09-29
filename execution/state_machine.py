"""Order State Machine (Section 32).

Enforces strictly validated order lifecycle transitions:
CREATED -> VALIDATED -> SUBMITTED -> ACKNOWLEDGED -> PARTIALLY_FILLED -> FILLED / CANCELLED.
Rejects illegal transitions and prevents state corruption.
"""

from __future__ import annotations

from datetime import UTC, datetime

from execution.models import Order, OrderState


class InvalidOrderStateTransition(ValueError):
    """Raised when an order transition violates lifecycle rules."""
    pass


class OrderStateMachine:
    """Manages legal transitions for the Order lifecycle."""

    # Map of source state to allowed target states
    VALID_TRANSITIONS: dict[OrderState, set[OrderState]] = {
        OrderState.CREATED: {
            OrderState.VALIDATED,
            OrderState.REJECTED,
            OrderState.ERROR,
        },
        OrderState.VALIDATED: {
            OrderState.SUBMITTED,
            OrderState.REJECTED,
            OrderState.ERROR,
        },
        OrderState.SUBMITTED: {
            OrderState.ACKNOWLEDGED,
            OrderState.REJECTED,
            OrderState.ERROR,
        },
        OrderState.ACKNOWLEDGED: {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        },
        OrderState.PARTIALLY_FILLED: {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        },
        OrderState.CANCEL_REQUESTED: {
            OrderState.CANCELLED,
            OrderState.FILLED,
            OrderState.PARTIALLY_FILLED,
            OrderState.ERROR,
        },
        # Terminal states cannot transition further
        OrderState.FILLED: set(),
        OrderState.CANCELLED: set(),
        OrderState.REJECTED: set(),
        OrderState.EXPIRED: set(),
        OrderState.ERROR: set(),
    }

    @classmethod
    def can_transition(cls, current_state: OrderState, target_state: OrderState) -> bool:
        """Check if transition from current_state to target_state is legal."""
        return target_state in cls.VALID_TRANSITIONS.get(current_state, set())

    @classmethod
    def transition(
        cls,
        order: Order,
        target_state: OrderState,
        reason: str = "",
    ) -> Order:
        """Execute state transition on order, or raise InvalidOrderStateTransition."""
        if not cls.can_transition(order.state, target_state):
            raise InvalidOrderStateTransition(
                f"Illegal order transition for {order.order_id}: "
                f"Cannot transition from {order.state} to {target_state} (Reason: {reason})"
            )

        order.state = target_state
        order.updated_at = datetime.now(UTC)
        if target_state in (OrderState.REJECTED, OrderState.ERROR) and reason:
            order.error_message = reason

        return order
