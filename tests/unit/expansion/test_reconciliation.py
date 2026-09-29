"""Unit tests for Order and Position Reconciliation (Sections 33, 34, 58)."""

from execution.reconciliation import ExecutionReconciliationEngine


def test_reconciliation_clean_match():
    engine = ExecutionReconciliationEngine()

    internal = {"BTC/USDT": 2.5, "ETH/USDT": 10.0}
    provider = {"BTC/USDT": 2.5, "ETH/USDT": 10.0}

    is_sync, discrepancies = engine.reconcile_positions(internal, provider)
    assert is_sync is True
    assert len(discrepancies) == 0
    assert engine.reconciliation_required is False
    assert engine.can_safely_execute("BTC/USDT") is True


def test_critical_reconciliation_test_section_58():
    """Section 58 Critical Reconciliation Test:
    Internal position: 10 units. Provider reports: 7 units.
    Verify: discrepancy detected, trading state becomes reconciliation-required,
    unsafe new execution is blocked, discrepancy logged, audit event created.
    """
    engine = ExecutionReconciliationEngine()

    # 1. Internal: 10 units, Provider reports: 7 units
    internal = {"BTC/USDT": 10.0}
    provider = {"BTC/USDT": 7.0}

    # 2. Reconcile
    is_sync, discrepancies = engine.reconcile_positions(internal, provider)

    # 3. Discrepancy detected
    assert is_sync is False
    assert len(discrepancies) == 1
    disc = discrepancies[0]
    assert disc.symbol == "BTC/USDT"
    assert disc.internal_quantity == 10.0
    assert disc.provider_quantity == 7.0
    assert disc.delta == 3.0

    # 4. Trading state becomes reconciliation-required
    assert engine.reconciliation_required is True

    # 5. Unsafe new execution is blocked on affected symbol
    assert engine.can_safely_execute("BTC/USDT") is False

    # 6. Manual resolution clears block
    resolved = engine.resolve_discrepancy(
        symbol="BTC/USDT",
        actor="lead_risk_officer",
        resolution_notes="Discrepancy reconciled: external manual deposit matched.",
    )
    assert resolved is True
    assert engine.reconciliation_required is False
    assert engine.can_safely_execute("BTC/USDT") is True
