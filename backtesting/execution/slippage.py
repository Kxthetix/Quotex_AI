"""Slippage models for realistic execution simulation (Phase 5).

Models price impact and adverse execution movement:
- BUY orders slip upwards (execution price increases).
- SELL orders slip downwards (execution price decreases).
"""

from abc import ABC, abstractmethod


class SlippageModel(ABC):
    """Abstract base class for execution slippage."""

    @abstractmethod
    def calculate_slippage(
        self,
        base_price: float,
        quantity: float,
        is_buy: bool,
        volatility: float = 0.0,
    ) -> tuple[float, float]:
        """Calculate adjusted execution price and absolute dollar slippage cost.

        Returns:
            tuple[float, float]: (slipped_price, slippage_dollar_cost)
        """
        pass


class PercentageSlippageModel(SlippageModel):
    """Slippage modeled as a fixed percentage of price plus optional volatility scaling."""

    def __init__(
        self,
        base_slippage_pct: float = 0.0003,  # 3 bps (0.03%)
        volatility_factor: float = 0.0,
    ) -> None:
        self.base_slippage_pct = max(0.0, base_slippage_pct)
        self.volatility_factor = max(0.0, volatility_factor)

    def calculate_slippage(
        self,
        base_price: float,
        quantity: float,
        is_buy: bool,
        volatility: float = 0.0,
    ) -> tuple[float, float]:
        effective_pct = self.base_slippage_pct + (self.volatility_factor * volatility)
        delta_p = base_price * effective_pct
        slipped_price = base_price + delta_p if is_buy else max(0.0001, base_price - delta_p)

        dollar_cost = delta_p * quantity
        return float(slipped_price), float(dollar_cost)


class FixedTicksSlippageModel(SlippageModel):
    """Slippage modeled as a fixed number of price ticks / absolute delta."""

    def __init__(self, tick_size: float = 0.5, ticks: int = 1) -> None:
        self.tick_delta = tick_size * ticks

    def calculate_slippage(
        self,
        base_price: float,
        quantity: float,
        is_buy: bool,
        volatility: float = 0.0,
    ) -> tuple[float, float]:
        delta_p = self.tick_delta
        slipped_price = base_price + delta_p if is_buy else max(0.0001, base_price - delta_p)

        dollar_cost = delta_p * quantity
        return float(slipped_price), float(dollar_cost)
