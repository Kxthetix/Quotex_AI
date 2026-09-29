"""Asset and Symbol Registry (Section 3, 9).

Maintains canonical assets, enforces validation, and normalizes incoming/outgoing symbols.
"""

from __future__ import annotations

import threading
from typing import Any

from data.assets.models import Asset, AssetClass, SymbolMapping


class AssetRegistry:
    """Thread-safe registry for assets and multi-provider symbol normalization."""

    _instance: AssetRegistry | None = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._assets: dict[str, Asset] = {}  # asset_id -> Asset
        self._symbol_to_id: dict[str, str] = {}  # canonical symbol -> asset_id
        self._mappings: dict[str, SymbolMapping] = {}  # asset_id -> SymbolMapping
        self._provider_to_canonical: dict[tuple[str, str], str] = {}  # (provider, prov_symbol) -> asset_id
        self._lock = threading.Lock()

        # Seed standard default assets
        self._seed_default_assets()

    @classmethod
    def get_instance(cls) -> AssetRegistry:
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = AssetRegistry()
            return cls._instance

    def _seed_default_assets(self) -> None:
        """Register default crypto assets with standard provider mappings."""
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
        self.register_asset(
            btc,
            provider_symbols={
                "binance": "BTCUSDT",
                "coinbase": "BTC-USD",
                "kraken": "XBT/USD",
                "mock": "BTC/USDT",
            },
        )

        eth = Asset(
            asset_id="asset_eth_usdt",
            symbol="ETH/USDT",
            asset_class=AssetClass.CRYPTO,
            exchange="BINANCE",
            quote_currency="USDT",
            base_currency="ETH",
            tick_size=0.01,
            lot_size=0.001,
        )
        self.register_asset(
            eth,
            provider_symbols={
                "binance": "ETHUSDT",
                "coinbase": "ETH-USD",
                "kraken": "ETH/USD",
                "mock": "ETH/USDT",
            },
        )

        sol = Asset(
            asset_id="asset_sol_usdt",
            symbol="SOL/USDT",
            asset_class=AssetClass.CRYPTO,
            exchange="BINANCE",
            quote_currency="USDT",
            base_currency="SOL",
            tick_size=0.01,
            lot_size=0.01,
        )
        self.register_asset(
            sol,
            provider_symbols={
                "binance": "SOLUSDT",
                "coinbase": "SOL-USD",
                "kraken": "SOL/USD",
                "mock": "SOL/USDT",
            },
        )

    def register_asset(
        self,
        asset: Asset,
        provider_symbols: dict[str, str] | None = None,
    ) -> None:
        """Register an asset and its provider symbol mappings."""
        with self._lock:
            self._assets[asset.asset_id] = asset
            clean_sym = asset.symbol.strip().upper()
            self._symbol_to_id[clean_sym] = asset.asset_id

            mapping = SymbolMapping(
                canonical_asset_id=asset.asset_id,
                canonical_symbol=asset.symbol,
                provider_symbols={k.lower(): v.strip().upper() for k, v in (provider_symbols or {}).items()},
            )
            self._mappings[asset.asset_id] = mapping

            # Reverse index for normalization
            for provider, prov_sym in mapping.provider_symbols.items():
                self._provider_to_canonical[(provider.lower(), prov_sym.upper())] = asset.asset_id

    def get_asset(self, asset_id: str) -> Asset | None:
        with self._lock:
            return self._assets.get(asset_id)

    def get_asset_by_symbol(self, symbol: str) -> Asset | None:
        with self._lock:
            clean = symbol.strip().upper()
            aid = self._symbol_to_id.get(clean)
            return self._assets.get(aid) if aid else None

    def normalize_provider_symbol(self, provider: str, provider_symbol: str) -> Asset:
        """Resolve incoming provider symbol to canonical Asset.
        
        Raises KeyError if symbol is unknown (Section 9: Unknown symbols must be rejected).
        """
        with self._lock:
            key = (provider.lower(), provider_symbol.strip().upper())
            aid = self._provider_to_canonical.get(key)
            if not aid:
                raise KeyError(
                    f"Unknown symbol: '{provider_symbol}' for provider '{provider}'. Rejecting event."
                )
            return self._assets[aid]

    def resolve_to_provider_symbol(self, asset_id: str, provider: str) -> str:
        """Get provider-specific symbol for an internal asset ID."""
        with self._lock:
            mapping = self._mappings.get(asset_id)
            if not mapping:
                raise KeyError(f"No mapping registered for asset_id: {asset_id}")
            return mapping.get_provider_symbol(provider)

    def list_assets(self, active_only: bool = True) -> list[Asset]:
        with self._lock:
            if active_only:
                return [a for a in self._assets.values() if a.active]
            return list(self._assets.values())
