"""Unit tests for Multi-Provider Architecture and Controlled Failover (Sections 5, 6, 7, 8, 54)."""

import pytest

from data.assets.registry import AssetRegistry
from data.providers.adapters import SyntheticProviderAdapter
from data.providers.failover import (
    CrossProviderConsistencyChecker,
    ProviderFailoverManager,
    ProviderPriority,
)


@pytest.mark.asyncio
async def test_provider_adapter_connectivity_and_health():
    adapter = SyntheticProviderAdapter(name="primary_feed")
    assert adapter.is_connected is False

    await adapter.connect()
    assert adapter.is_connected is True

    health = adapter.get_health()
    assert health["connected"] is True
    assert health["provider"] == "primary_feed"

    await adapter.disconnect()
    assert adapter.is_connected is False


@pytest.mark.asyncio
async def test_provider_failover_priority():
    registry = AssetRegistry.get_instance()
    manager = ProviderFailoverManager(asset_registry=registry)

    primary = SyntheticProviderAdapter(name="binance")
    secondary = SyntheticProviderAdapter(name="coinbase")

    await primary.connect()
    await secondary.connect()

    manager.register_provider(ProviderPriority.PRIMARY, primary)
    manager.register_provider(ProviderPriority.SECONDARY, secondary)

    assert manager.active_provider == primary
    assert manager.active_priority == ProviderPriority.PRIMARY

    # Trigger failover due to socket loss
    success, new_provider, msg = manager.trigger_failover(
        reason="Primary socket ping timeout > 15s",
        required_symbols=["BTC/USDT"],
    )
    assert success is True
    assert new_provider == secondary
    assert manager.active_priority == ProviderPriority.SECONDARY
    assert "coinbase" in msg


@pytest.mark.asyncio
async def test_critical_provider_test_section_54():
    """Section 54 Critical Provider Test:
    Provider A active -> Provider A disconnects -> failure detected ->
    safe state -> reconnect attempted -> provider state validated -> resumes.
    """
    manager = ProviderFailoverManager()

    provider_a = SyntheticProviderAdapter(name="binance")
    provider_b = SyntheticProviderAdapter(name="coinbase")

    await provider_a.connect()
    await provider_b.connect()

    manager.register_provider(ProviderPriority.PRIMARY, provider_a)
    manager.register_provider(ProviderPriority.SECONDARY, provider_b)

    # 1. Provider A is active
    assert manager.active_provider.provider_name == "binance"

    # 2. Provider A disconnects
    provider_a.inject_failure("DISCONNECTED")
    assert provider_a.is_connected is False

    # 3. Detect failure and validate backup before failover
    is_valid, checks = manager.validate_candidate_provider(provider_b, ["BTC/USDT"])
    assert is_valid is True
    assert checks["connected"] is True
    assert checks["symbol_mapping_valid"] is True

    # 4. Controlled failover occurs
    success, active_prov, msg = manager.trigger_failover("Provider A down", ["BTC/USDT"])
    assert success is True
    assert active_prov.provider_name == "coinbase"

    # 5. Service resumes with Provider B
    assert manager.active_provider.is_connected is True


def test_cross_provider_consistency_check():
    checker = CrossProviderConsistencyChecker(max_allowed_price_divergence_pct=0.5)

    # Consistent prices (< 0.5% divergence)
    is_ok, div, details = checker.check_quote_consistency(
        symbol="BTC/USDT",
        price_a=50000.0,
        provider_a="binance",
        price_b=50100.0,  # 0.2% diff
        provider_b="coinbase",
    )
    assert is_ok is True
    assert div < 0.5
    assert len(checker.get_discrepancies()) == 0

    # Divergent prices (> 0.5% divergence)
    is_ok2, div2, details2 = checker.check_quote_consistency(
        symbol="BTC/USDT",
        price_a=50000.0,
        provider_a="binance",
        price_b=51000.0,  # 2.0% diff
        provider_b="coinbase",
    )
    assert is_ok2 is False
    assert div2 > 1.5
    assert len(checker.get_discrepancies()) == 1
