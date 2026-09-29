"""Execution Coordinator and Risk Engine Enforcement (Sections 27, 30, 31).

Enforces the mandatory execution pipeline:
    Prediction -> Signal -> Risk Engine -> Execution Policy -> Execution Provider
Prediction code can NEVER directly place orders.
If Risk Engine issues a VETO, NO ORDER IS SUBMITTED.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from data.assets.registry import AssetRegistry
from execution.models import Order, OrderSide, OrderState, OrderType
from execution.modes import ExecutionMode, ModeGuard
from execution.provider import ExecutionProvider

if TYPE_CHECKING:
    from production.audit.audit_logger import AuditLogger


class RiskVetoError(PermissionError):
    """Raised when an attempt is made to execute an order vetoed by the Risk Engine."""
    pass


class ExecutionCoordinator:
    """Coordinates order construction and execution only after independent Risk Engine approval."""

    def __init__(
        self,
        execution_provider: ExecutionProvider,
        mode: ExecutionMode = ExecutionMode.PAPER,
        asset_registry: AssetRegistry | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.provider = execution_provider
        self.mode = mode
        self.asset_registry = asset_registry or AssetRegistry.get_instance()
        self.audit_logger = audit_logger

        # Verify safety guard on initialization
        ModeGuard.validate_mode(self.mode)

    def execute_signal(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float | None = None,
        order_type: OrderType = OrderType.MARKET,
        risk_approved: bool = False,
        risk_decision_id: str | None = None,
        signal_id: str | None = None,
        correlation_id: str | None = None,
        risk_reason: str = "",
    ) -> tuple[bool, Order | None, str]:
        """Submit order ONLY IF the Risk Engine has explicitly approved.
        
        Section 27, 30, 57: If risk_approved is False, strictly rejects the request.
        No order is sent to the ExecutionProvider.
        """
        # 1. Independent Risk Engine Gate (Sections 27, 30)
        if not risk_approved:
            if self.audit_logger:
                self.audit_logger.record_action(
                    actor="ExecutionCoordinator",
                    action="RISK_VETO_ENFORCED",
                    resource=f"order:{symbol}:{side}",
                    reason=f"Risk Engine vetoed order execution: {risk_reason}",
                    previous_state=None,
                    new_state={"status": "REJECTED_BY_RISK", "signal_id": signal_id},
                    correlation_id=correlation_id,
                )
            return False, None, f"Order execution blocked by Risk Engine VETO: {risk_reason}"

        # 2. Asset & Symbol Validation (Section 31)
        asset = self.asset_registry.get_asset_by_symbol(symbol)
        asset_id = asset.asset_id if asset else ""

        # Validate minimum lot size if asset is registered
        if asset and quantity < asset.lot_size:
            return False, None, f"Quantity {quantity} is less than asset minimum lot size {asset.lot_size}"

        # 3. Construct Order
        order = Order(
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            asset_id=asset_id,
            signal_id=signal_id,
            risk_decision_id=risk_decision_id,
            correlation_id=correlation_id,
        )

        # 4. Dispatch to Execution Provider
        try:
            executed_order = self.provider.submit_order(order)
            success = executed_order.state in (OrderState.ACKNOWLEDGED, OrderState.FILLED, OrderState.SUBMITTED)

            if self.audit_logger:
                self.audit_logger.record_action(
                    actor="ExecutionCoordinator",
                    action=f"ORDER_{executed_order.state.value}",
                    resource=f"order:{executed_order.order_id}",
                    reason=f"Risk-approved order processed in {self.mode.value} mode",
                    previous_state=None,
                    new_state=executed_order.to_dict(),
                    correlation_id=correlation_id,
                )

            return success, executed_order, f"Order processed: state={executed_order.state.value}"
        except Exception as e:
            return False, order, f"Execution failed at provider level: {e}"
