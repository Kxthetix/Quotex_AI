"""Unit tests for Multi-Asset Abstraction and Symbol Normalization (Sections 3, 4, 9)."""

import pytest

from data.assets.models import Asset, AssetClass, SymbolMapping
from data.assets.registry import AssetRegistry


def test_asset_creation_and_attributes():
    btc = Asset(
        asset_id="asset_btc_usdt",
        symbol="BTC/USDT",
        asset_class=AssetClass.CRYPTO,
        exchange="BINANCE",
        quote_currency="USDT",
        base_currency="BTC",
        tick_size=0.01,
        lot_size=0.0001,
    )
    assert btc.symbol == "BTC/USDT"
    assert btc.asset_class == AssetClass.CRYPTO
    assert btc.active is True
    data = btc.to_dict()
    assert data["quote_currency"] == "USDT"


def test_asset_registry_symbol_normalization():
    registry = AssetRegistry()

    # Normalize incoming Binance symbol
    asset_binance = registry.normalize_provider_symbol("binance", "BTCUSDT")
    assert asset_binance.symbol == "BTC/USDT"
    assert asset_binance.asset_id == "asset_btc_usdt"

    # Normalize incoming Coinbase symbol
    asset_coinbase = registry.normalize_provider_symbol("coinbase", "BTC-USD")
    assert asset_coinbase.symbol == "BTC/USDT"

    # Resolve to provider-specific symbol
    prov_sym = registry.resolve_to_provider_symbol("asset_btc_usdt", "kraken")
    assert prov_sym == "XBT/USD"


def test_unknown_symbol_rejection():
    """Section 9: Unknown symbols must be rejected rather than guessed."""
    registry = AssetRegistry()

    with pytest.raises(KeyError) as exc_info:
        registry.normalize_provider_symbol("binance", "UNKNOWN_JUNK_TICKER")
    assert "Unknown symbol" in str(exc_info.value)
