"""Asset and Symbol Abstraction for Multi-Asset Trading (Section 3, 9).

Defines canonical Asset entities, contract specifications, and versioned symbol mappings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class AssetClass(StrEnum):
    CRYPTO = "CRYPTO"
    EQUITY = "EQUITY"
    FX = "FX"
    COMMODITY = "COMMODITY"
    FUTURES = "FUTURES"


@dataclass
class Asset:
    """Canonical asset representation with trading specifications (Section 3)."""
    asset_id: str
    symbol: str  # Canonical symbol, e.g. "BTC/USDT"
    asset_class: AssetClass
    exchange: str  # Primary exchange or market venue
    quote_currency: str  # e.g. "USDT", "USD"
    base_currency: str  # e.g. "BTC", "AAPL"
    trading_hours: str = "24/7"  # e.g. "24/7" or "09:30-16:00 EST"
    tick_size: float = 0.01
    lot_size: float = 0.0001
    contract_specs: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["asset_class"] = str(self.asset_class)
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass
class SymbolMapping:
    """Maps canonical asset ID to provider-specific ticker symbols (Section 9)."""
    canonical_asset_id: str
    canonical_symbol: str
    provider_symbols: dict[str, str] = field(default_factory=dict)
    version: str = "v1.0"
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_provider_symbol(self, provider: str) -> str:
        """Resolve canonical asset to provider-specific symbol or raise KeyError."""
        clean_provider = provider.lower()
        if clean_provider not in self.provider_symbols:
            raise KeyError(
                f"No mapping registered for provider '{provider}' and asset '{self.canonical_symbol}'"
            )
        return self.provider_symbols[clean_provider]
