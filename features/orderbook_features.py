"""Microstructure Order Book Feature Engine (Sections 13, 14, 15).

Feature Set Version: orderbook_features_v1
Implements 9 standard microstructure features with strict temporal leakage verification.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from data.orderbook.models import OrderBookSnapshot


@dataclass
class OrderBookFeatures:
    """Canonical microstructure feature vector (version: orderbook_features_v1)."""
    feature_version: str = "orderbook_features_v1"
    timestamp: datetime = datetime.now(UTC)
    symbol: str = ""
    bid_ask_spread: float = 0.0
    relative_spread: float = 0.0
    top_of_book_imbalance: float = 0.0
    n_level_imbalance: float = 0.0
    depth_weighted_imbalance: float = 0.0
    microprice: float = 0.0
    microprice_deviation: float = 0.0
    book_pressure: float = 0.5
    liquidity_concentration: float = 0.0
    depth_slope: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class OrderBookFeatureEngine:
    """Calculates normalized microstructure features strictly obeying temporal causality."""

    VERSION = "orderbook_features_v1"

    @classmethod
    def calculate_features(
        cls,
        snapshot: OrderBookSnapshot,
        as_of_time: datetime | None = None,
        n_levels: int = 5,
    ) -> OrderBookFeatures:
        """Compute microstructure features from an OrderBookSnapshot.
        
        Section 14: If as_of_time is provided and snapshot.timestamp > as_of_time,
        a ValueError is raised to enforce temporal leakage protection.
        """
        if as_of_time is not None:
            snap_t = snapshot.timestamp if snapshot.timestamp.tzinfo else snapshot.timestamp.replace(tzinfo=UTC)
            ref_t = as_of_time if as_of_time.tzinfo else as_of_time.replace(tzinfo=UTC)
            if snap_t > ref_t:
                raise ValueError(
                    f"Temporal leakage violation: Snapshot timestamp {snap_t} is in future of as_of_time {ref_t}"
                )

        if not snapshot.bids or not snapshot.asks:
            raise ValueError("Cannot calculate order-book features from an empty order book.")

        bids = snapshot.bids
        asks = snapshot.asks

        best_bid = bids[0].price
        best_ask = asks[0].price
        bid_size_1 = bids[0].size
        ask_size_1 = asks[0].size

        # 1. Spreads
        spread = best_ask - best_bid
        mid_price = (best_ask + best_bid) / 2.0
        rel_spread = (spread / mid_price) if mid_price > 0 else 0.0

        # 2. Top-of-book imbalance
        tob_denom = bid_size_1 + ask_size_1
        tob_imbalance = (bid_size_1 - ask_size_1) / tob_denom if tob_denom > 0 else 0.0

        # 3. N-level imbalance
        n = min(n_levels, len(bids), len(asks))
        sum_bid_n = sum(b.size for b in bids[:n])
        sum_ask_n = sum(a.size for a in asks[:n])
        n_denom = sum_bid_n + sum_ask_n
        n_imbalance = (sum_bid_n - sum_ask_n) / n_denom if n_denom > 0 else 0.0

        # 4. Depth-weighted imbalance (weight = 1 / level)
        weighted_bids = sum(b.size / b.level for b in bids[:n])
        weighted_asks = sum(a.size / a.level for a in asks[:n])
        weighted_denom = weighted_bids + weighted_asks
        depth_weighted_imbalance = (
            (weighted_bids - weighted_asks) / weighted_denom if weighted_denom > 0 else 0.0
        )

        # 5. Microprice and deviation
        # microprice = (ask_p * bid_q + bid_p * ask_q) / (bid_q + ask_q)
        microprice = (
            (best_ask * bid_size_1 + best_bid * ask_size_1) / tob_denom
            if tob_denom > 0
            else mid_price
        )
        microprice_dev = microprice - mid_price

        # 6. Book pressure (total bid volume vs total depth)
        total_bid_vol = sum(b.size for b in bids)
        total_ask_vol = sum(a.size for a in asks)
        total_depth_vol = total_bid_vol + total_ask_vol
        book_pressure = (total_bid_vol / total_depth_vol) if total_depth_vol > 0 else 0.5

        # 7. Liquidity concentration (top 3 levels vs total volume)
        top3_vol = sum(b.size for b in bids[:3]) + sum(a.size for a in asks[:3])
        liquidity_concentration = (top3_vol / total_depth_vol) if total_depth_vol > 0 else 0.0

        # 8. Depth slope (rate of volume accumulation across levels)
        if n > 1:
            avg_bid_slope = (bids[n - 1].size - bids[0].size) / (n - 1)
            avg_ask_slope = (asks[n - 1].size - asks[0].size) / (n - 1)
            depth_slope = (avg_bid_slope + avg_ask_slope) / 2.0
        else:
            depth_slope = 0.0

        return OrderBookFeatures(
            feature_version=cls.VERSION,
            timestamp=snapshot.timestamp,
            symbol=snapshot.symbol,
            bid_ask_spread=round(spread, 6),
            relative_spread=round(rel_spread, 6),
            top_of_book_imbalance=round(tob_imbalance, 6),
            n_level_imbalance=round(n_imbalance, 6),
            depth_weighted_imbalance=round(depth_weighted_imbalance, 6),
            microprice=round(microprice, 6),
            microprice_deviation=round(microprice_dev, 6),
            book_pressure=round(book_pressure, 6),
            liquidity_concentration=round(liquidity_concentration, 6),
            depth_slope=round(depth_slope, 6),
        )
