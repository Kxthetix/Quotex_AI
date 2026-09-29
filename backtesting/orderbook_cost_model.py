"""Order-Book Aware Execution Cost Model (Section 18).

Calculates realistic execution prices, slippage, and market impact
by walking through available Level 2 order-book depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data.orderbook.models import OrderBookSnapshot


@dataclass
class ExecutionEstimate:
    """Estimated fill details across order-book depth."""
    order_side: str  # "BUY" or "SELL"
    order_size: float
    effective_avg_price: float
    slippage_bps: float
    spread_cost: float
    depth_exhausted_levels: int
    unfilled_size: float = 0.0


class OrderBookCostModel:
    """Models realistic execution slippage and spread capture across order-book depth."""

    def __init__(self, fee_bps: float = 5.0) -> None:
        self.fee_bps = fee_bps

    def estimate_execution(
        self,
        snapshot: OrderBookSnapshot,
        side: str,
        size: float,
    ) -> ExecutionEstimate:
        """Calculate weighted average fill price walking the book."""
        clean_side = side.upper()
        if clean_side == "BUY":
            levels = snapshot.asks
            reference_price = snapshot.best_ask
        elif clean_side == "SELL":
            levels = snapshot.bids
            reference_price = snapshot.best_bid
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")

        if not levels or size <= 0:
            return ExecutionEstimate(
                order_side=clean_side,
                order_size=size,
                effective_avg_price=reference_price,
                slippage_bps=0.0,
                spread_cost=0.0,
                depth_exhausted_levels=0,
                unfilled_size=size,
            )

        remaining = size
        total_cost = 0.0
        levels_walked = 0

        for lvl in levels:
            levels_walked += 1
            fill_qty = min(remaining, lvl.size)
            total_cost += fill_qty * lvl.price
            remaining -= fill_qty
            if remaining <= 1e-9:
                break

        filled_size = size - remaining
        avg_price = total_cost / filled_size if filled_size > 0 else reference_price

        # Slippage in basis points relative to best quote
        if reference_price > 0:
            if clean_side == "BUY":
                slippage_bps = max(0.0, (avg_price - reference_price) / reference_price) * 10000.0
            else:
                slippage_bps = max(0.0, (reference_price - avg_price) / reference_price) * 10000.0
        else:
            slippage_bps = 0.0

        spread_cost = (snapshot.spread / 2.0) * filled_size

        return ExecutionEstimate(
            order_side=clean_side,
            order_size=size,
            effective_avg_price=round(avg_price, 4),
            slippage_bps=round(slippage_bps, 2),
            spread_cost=round(spread_cost, 4),
            depth_exhausted_levels=levels_walked,
            unfilled_size=round(remaining, 6),
        )
