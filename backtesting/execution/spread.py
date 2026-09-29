"""Bid/Ask spread models for realistic execution pricing (Phase 5).

Derives synthetic bid and ask prices from reference market prices:
- BUY orders execute at Ask (higher price).
- SELL orders execute at Bid (lower price).
"""

from abc import ABC, abstractmethod


class SpreadModel(ABC):
    """Abstract base class for bid/ask spread models."""

    @abstractmethod
    def get_bid_ask(self, mid_price: float) -> tuple[float, float]:
        """Derive (bid_price, ask_price) from mid/reference price."""
        pass

    def get_bid_price(self, mid_price: float) -> float:
        """Return derived bid price."""
        return self.get_bid_ask(mid_price)[0]

    def get_ask_price(self, mid_price: float) -> float:
        """Return derived ask price."""
        return self.get_bid_ask(mid_price)[1]

    def get_execution_price(self, mid_price: float, is_buy: bool) -> float:
        """Return execution price taking spread into account."""
        bid, ask = self.get_bid_ask(mid_price)
        return ask if is_buy else bid

    def get_spread_cost(self, mid_price: float, quantity: float) -> float:
        """Calculate the half-spread transaction cost."""
        bid, ask = self.get_bid_ask(mid_price)
        half_spread = (ask - bid) / 2.0
        return float(half_spread * quantity)


class PercentageSpreadModel(SpreadModel):
    """Calculates spread as a percentage of the mid price."""

    def __init__(self, spread_pct: float = 0.0002) -> None:
        """Initialize percentage spread model.

        Args:
            spread_pct: Full bid-ask spread fraction (e.g. 0.0002 = 2 bps = 0.02%).
        """
        self.spread_pct = max(0.0, spread_pct)

    def get_bid_ask(self, mid_price: float) -> tuple[float, float]:
        half = mid_price * (self.spread_pct / 2.0)
        bid = mid_price - half
        ask = mid_price + half
        return (bid, ask)


class FixedSpreadModel(SpreadModel):
    """Calculates spread as a fixed absolute currency amount."""

    def __init__(self, fixed_spread: float = 1.0) -> None:
        """Initialize fixed spread model (e.g. $1.00 spread on BTC)."""
        self.fixed_spread = max(0.0, fixed_spread)

    def get_bid_ask(self, mid_price: float) -> tuple[float, float]:
        half = self.fixed_spread / 2.0
        bid = mid_price - half
        ask = mid_price + half
        return (bid, ask)
