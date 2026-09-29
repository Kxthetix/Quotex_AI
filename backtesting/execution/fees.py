"""Fee models for execution simulation (Phase 5).

Supports maker/taker percentage fees and fixed transaction fees.
"""

from abc import ABC, abstractmethod


class FeeModel(ABC):
    """Abstract base class for transaction fee calculation."""

    @abstractmethod
    def calculate_fee(self, price: float, quantity: float, is_maker: bool = False) -> float:
        """Calculate total fee charged for an executed fill."""
        pass


class PercentageFeeModel(FeeModel):
    """Calculates fee as a percentage of filled notional value."""

    def __init__(
        self,
        maker_fee: float = 0.0002,  # 0.02%
        taker_fee: float = 0.0005,  # 0.05%
        fixed_fee: float = 0.0,
    ) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.fixed_fee = fixed_fee

    def calculate_fee(self, price: float, quantity: float, is_maker: bool = False) -> float:
        notional = price * quantity
        rate = self.maker_fee if is_maker else self.taker_fee
        return float(notional * rate + self.fixed_fee)
