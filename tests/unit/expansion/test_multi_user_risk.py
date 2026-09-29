"""Unit tests for Multi-Asset Risk Engine and Multi-User Tenant Isolation (Sections 37-44, 59)."""

import pytest

from production.security.multi_user import (
    MultiUserManager,
    TenantIsolationViolationError,
    UserRole,
)
from risk.multi_asset_risk import MultiAssetRiskEngine


def test_multi_asset_risk_concentration_veto():
    """Section 44: Veto order when single-asset exposure exceeds concentration limit."""
    engine = MultiAssetRiskEngine(max_single_asset_concentration_pct=30.0, max_aggregate_leverage=1.0)

    total_equity = 100_000.0
    current_positions = {"ETH/USDT": 10_000.0}

    # Propose buying 40,000 BTC -> 40% concentration (exceeds 30% limit)
    assessment = engine.evaluate_order(
        symbol="BTC/USDT",
        order_side="BUY",
        order_value=40_000.0,
        current_positions_value=current_positions,
        total_equity=total_equity,
    )
    assert assessment.approved is False
    assert "Concentration 40.0% exceeds maximum" in assessment.veto_reason


def test_multi_asset_risk_leverage_and_correlation_veto():
    """Section 44: Veto order when aggregate leverage or high correlation limit is breached."""
    engine = MultiAssetRiskEngine(
        max_single_asset_concentration_pct=50.0,
        max_aggregate_leverage=1.0,
        asset_correlation_threshold=0.85,
    )

    total_equity = 100_000.0
    current_positions = {"BTC/USDT": 45_000.0, "ETH/USDT": 45_000.0}

    # 1. Total leverage breach: currently 90k, adding 20k -> 110k (1.10x > 1.0x limit)
    res_lev = engine.evaluate_order(
        symbol="SOL/USDT",
        order_side="BUY",
        order_value=20_000.0,
        current_positions_value=current_positions,
        total_equity=total_equity,
    )
    assert res_lev.approved is False
    assert "leverage" in res_lev.veto_reason.lower()

    # 2. Correlation breach: adding WBTC when holding BTC (correlation 0.98 > 0.85)
    correlations = {("WBTC/USDT", "BTC/USDT"): 0.98}
    res_corr = engine.evaluate_order(
        symbol="WBTC/USDT",
        order_side="BUY",
        order_value=5_000.0,
        current_positions_value={"BTC/USDT": 20_000.0},
        total_equity=total_equity,
        pairwise_correlations=correlations,
    )
    assert res_corr.approved is False
    assert "High correlation" in res_corr.veto_reason


def test_critical_multi_user_tenant_isolation_section_59():
    """Section 59 Critical Multi-User Test:
    User A owns Portfolio A. User B owns Portfolio B.
    Verify: User A cannot access Portfolio B, User B orders, positions, or credentials.
    """
    manager = MultiUserManager()

    # Tenant 1: User A
    user_a = manager.register_user(
        user_id="user_alice_01",
        tenant_id="tenant_alpha",
        username="alice",
        roles=[UserRole.TRADER],
    )
    port_a = manager.create_portfolio("port_alice_spot", user=user_a, initial_cash=50_000.0)

    # Tenant 2: User B
    user_b = manager.register_user(
        user_id="user_bob_02",
        tenant_id="tenant_beta",
        username="bob",
        roles=[UserRole.TRADER],
    )
    port_b = manager.create_portfolio("port_bob_spot", user=user_b, initial_cash=75_000.0)

    # 1. Alice can access Portfolio A
    retrieved_a = manager.authorize_portfolio_access(user_a, port_a.portfolio_id)
    assert retrieved_a.owner_user_id == "user_alice_01"

    # 2. Bob can access Portfolio B
    retrieved_b = manager.authorize_portfolio_access(user_b, port_b.portfolio_id)
    assert retrieved_b.owner_user_id == "user_bob_02"

    # 3. CRITICAL: Alice CANNOT access Portfolio B (cross-user & cross-tenant)
    with pytest.raises(TenantIsolationViolationError) as exc:
        manager.authorize_portfolio_access(user_a, port_b.portfolio_id)
    assert "Access denied" in str(exc.value)

    # 4. Bob CANNOT access Portfolio A
    with pytest.raises(TenantIsolationViolationError):
        manager.authorize_portfolio_access(user_b, port_a.portfolio_id)
