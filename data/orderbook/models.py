"""Order Book Models and Schema (Sections 10, 11, 12).

Defines Level 2 order-book snapshots, individual price levels, and incremental updates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class OrderBookLevel:
    """Individual price level in the order book."""
    price: float
    size: float
    level: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrderBookSnapshot:
    """Complete Level 2 order-book snapshot at a specific point in time (Section 11)."""
    timestamp: datetime
    asset_id: str
    symbol: str
    provider: str
    sequence_number: int
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    mid_price: float = 0.0
    depth: int = 0

    def __post_init__(self) -> None:
        if self.bids:
            self.best_bid = self.bids[0].price
        if self.asks:
            self.best_ask = self.asks[0].price
        if self.best_bid > 0 and self.best_ask > 0:
            self.spread = self.best_ask - self.best_bid
            self.mid_price = (self.best_ask + self.best_bid) / 2.0
        self.depth = max(len(self.bids), len(self.asks))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset_id": self.asset_id,
            "symbol": self.symbol,
            "provider": self.provider,
            "sequence_number": self.sequence_number,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "mid_price": self.mid_price,
            "depth": self.depth,
            "bids": [b.to_dict() for b in self.bids],
            "asks": [a.to_dict() for a in self.asks],
        }


@dataclass
class OrderBookUpdate:
    """Incremental order-book update (Section 12).
    
    bids_delta and asks_delta contain tuples of (price, new_size).
    A size of 0.0 signals removal of that price level.
    """
    timestamp: datetime
    asset_id: str
    symbol: str
    provider: str
    sequence_number: int
    bids_delta: list[tuple[float, float]] = field(default_factory=list)
    asks_delta: list[tuple[float, float]] = field(default_factory=list)
    is_snapshot: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset_id": self.asset_id,
            "symbol": self.symbol,
            "provider": self.provider,
            "sequence_number": self.sequence_number,
            "bids_delta": self.bids_delta,
            "asks_delta": self.asks_delta,
            "is_snapshot": self.is_snapshot,
        }
