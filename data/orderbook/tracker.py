"""Order Book Tracker, Sequence Gap Detection, and Quality Validation (Sections 12, 16).

Maintains live order-book state from snapshot + incremental updates, detects
sequence gaps, validates price monotonicity, and halts feature generation on corruption.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from data.orderbook.models import (
    OrderBookLevel,
    OrderBookSnapshot,
    OrderBookUpdate,
)


class OrderBookStatus(StrEnum):
    VALID = "VALID"
    GAP_DETECTED = "GAP_DETECTED"
    CROSSED_BOOK = "CROSSED_BOOK"
    UNINITIALIZED = "UNINITIALIZED"
    INVALID_PRICING = "INVALID_PRICING"


class OrderBookTracker:
    """Maintains consistent Level 2 book, enforcing strict sequence continuity and validity."""

    def __init__(self, asset_id: str, symbol: str, provider: str = "binance") -> None:
        self.asset_id = asset_id
        self.symbol = symbol
        self.provider = provider
        self.status = OrderBookStatus.UNINITIALIZED
        self.last_sequence = 0
        self.last_timestamp: datetime = datetime.now(UTC)

        # Storage: price -> size
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._lock = threading.Lock()

    def apply_snapshot(self, snapshot: OrderBookSnapshot) -> bool:
        """Initialize or reset book state from a full snapshot."""
        with self._lock:
            self._bids.clear()
            self._asks.clear()

            for b in snapshot.bids:
                if b.size > 0:
                    self._bids[b.price] = b.size
            for a in snapshot.asks:
                if a.size > 0:
                    self._asks[a.price] = a.size

            self.last_sequence = snapshot.sequence_number
            self.last_timestamp = snapshot.timestamp
            self._validate_state()
            return self.status == OrderBookStatus.VALID

    def apply_update(self, update: OrderBookUpdate) -> tuple[bool, OrderBookStatus, str]:
        """Apply incremental update. Detects sequence gaps and invalid price states."""
        with self._lock:
            if self.status == OrderBookStatus.UNINITIALIZED:
                return False, self.status, "Book uninitialized; snapshot required."

            # Sequence continuity check (Section 12)
            expected_seq = self.last_sequence + 1
            if update.sequence_number != expected_seq:
                self.status = OrderBookStatus.GAP_DETECTED
                return (
                    False,
                    self.status,
                    f"Sequence gap detected: expected {expected_seq}, got {update.sequence_number}. Snapshot rebuild required.",
                )

            # Apply bid deltas
            for price, size in update.bids_delta:
                if size <= 0:
                    self._bids.pop(price, None)
                else:
                    self._bids[price] = size

            # Apply ask deltas
            for price, size in update.asks_delta:
                if size <= 0:
                    self._asks.pop(price, None)
                else:
                    self._asks[price] = size

            self.last_sequence = update.sequence_number
            self.last_timestamp = update.timestamp

            # Validate book invariants
            self._validate_state()

            if self.status != OrderBookStatus.VALID:
                return False, self.status, f"Order book entered invalid state: {self.status}"

            return True, self.status, "Update applied cleanly."

    def _validate_state(self) -> None:
        """Section 16: Validate price ordering, positivity, and uncrossed book."""
        if not self._bids or not self._asks:
            self.status = OrderBookStatus.INVALID_PRICING
            return

        sorted_bids = sorted(self._bids.keys(), reverse=True)
        sorted_asks = sorted(self._asks.keys())

        best_bid = sorted_bids[0]
        best_ask = sorted_asks[0]

        # Crossed book: best bid >= best ask
        if best_bid >= best_ask:
            self.status = OrderBookStatus.CROSSED_BOOK
            return

        # Check positive prices
        if best_bid <= 0 or best_ask <= 0:
            self.status = OrderBookStatus.INVALID_PRICING
            return

        self.status = OrderBookStatus.VALID

    def get_snapshot(self, depth: int = 10) -> OrderBookSnapshot | None:
        """Construct a standardized snapshot from current state. Returns None if book is corrupted."""
        with self._lock:
            if self.status != OrderBookStatus.VALID:
                return None

            sorted_bids = sorted(self._bids.keys(), reverse=True)[:depth]
            sorted_asks = sorted(self._asks.keys())[:depth]

            bid_levels = [
                OrderBookLevel(price=p, size=self._bids[p], level=i + 1)
                for i, p in enumerate(sorted_bids)
            ]
            ask_levels = [
                OrderBookLevel(price=p, size=self._asks[p], level=i + 1)
                for i, p in enumerate(sorted_asks)
            ]

            return OrderBookSnapshot(
                timestamp=self.last_timestamp,
                asset_id=self.asset_id,
                symbol=self.symbol,
                provider=self.provider,
                sequence_number=self.last_sequence,
                bids=bid_levels,
                asks=ask_levels,
            )
