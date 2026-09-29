"""Unit tests for Execution Integration, State Machine, and Risk Veto Enforcement (Sections 27-36, 57)."""

import pytest

from execution.coordinator import ExecutionCoordinator
from execution.models import (
    Order,
    OrderSide,
    OrderState,
    OrderType,
)
from execution.modes import ExecutionMode, ModeGuard
from execution.provider import SimulatedExecutionProvider
from execution.state_machine import (
    InvalidOrderStateTransition,
    OrderStateMachine,
)


def test_order_state_machine_legal_transitions():
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
    )
    assert order.state == OrderState.CREATED

    OrderStateMachine.transition(order, OrderState.VALIDATED)
    assert order.state == OrderState.VALIDATED

    OrderStateMachine.transition(order, OrderState.SUBMITTED)
    assert order.state == OrderState.SUBMITTED

    OrderStateMachine.transition(order, OrderState.ACKNOWLEDGED)
    assert order.state == OrderState.ACKNOWLEDGED

    OrderStateMachine.transition(order, OrderState.FILLED)
    assert order.state == OrderState.FILLED


def test_order_state_machine_illegal_transition_rejection():
    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
    )
    # Cannot jump directly from CREATED to FILLED
    with pytest.raises(InvalidOrderStateTransition):
        OrderStateMachine.transition(order, OrderState.FILLED)


def test_simulated_execution_provider_fill_and_portfolio_update():
    provider = SimulatedExecutionProvider(initial_cash=100_000.0, slippage_bps=2.0, fee_rate_bps=5.0)

    order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
        price=50000.0,
    )

    executed = provider.submit_order(order)
    assert executed.state == OrderState.FILLED
    assert executed.filled_quantity == 1.0
    assert executed.avg_fill_price > 50000.0  # Slippage applied
    assert executed.fees > 0

    # Verify account state
    acct = provider.query_account_state()
    assert acct.cash < 50000.0  # ~100k - 50k - fees
    pos = provider.query_position("BTC/USDT")
    assert pos.quantity == 1.0


def test_mode_guard_blocks_unauthorized_live_mode():
    """Section 29, 30: Accidentally switching to LIVE mode without env flag must raise PermissionError."""
    with pytest.raises(PermissionError) as exc:
        ModeGuard.validate_mode(ExecutionMode.LIVE)
    assert "CRITICAL SAFETY VIOLATION: LIVE execution mode blocked" in str(exc.value)


def test_critical_execution_test_section_57():
    """Section 57 Critical Execution Test:
    Prediction -> Signal -> Risk VETO -> verify NO ORDER IS SENT!
    Prediction -> Signal -> Risk APPROVED -> Execution Policy -> Order submitted only after all controls pass.
    """
    provider = SimulatedExecutionProvider()
    coordinator = ExecutionCoordinator(execution_provider=provider, mode=ExecutionMode.PAPER)

    # 1. RISK VETO Case
    ok_veto, order_veto, msg_veto = coordinator.execute_signal(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=2.0,
        price=50000.0,
        risk_approved=False,  # Risk Engine Vetoes
        risk_reason="Max drawdown limit reached",
        signal_id="sig_test_001",
    )

    # NO ORDER SENT!
    assert ok_veto is False
    assert order_veto is None
    assert "Risk Engine VETO" in msg_veto
    assert provider.query_account_state().cash == 100_000.0  # No cash touched

    # 2. RISK APPROVED Case
    ok_appr, order_appr, msg_appr = coordinator.execute_signal(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.5,
        price=50000.0,
        risk_approved=True,  # Risk Engine Approves
        risk_reason="Within portfolio risk limits",
        signal_id="sig_test_002",
        correlation_id="corr_9999",
    )

    # Order submitted and executed
    assert ok_appr is True
    assert order_appr is not None
    assert order_appr.state == OrderState.FILLED
    assert order_appr.filled_quantity == 0.5
    assert provider.query_position("BTC/USDT").quantity == 0.5
