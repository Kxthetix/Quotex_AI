"""Position sizing and risk limit enforcement for Phase 5.

Calculates order quantity based on configurable sizing algorithms and clamps or rejects
orders exceeding exposure, leverage, or position limits.
"""

from backtesting.models import BacktestConfig, SizingMethod


class PositionSizer:
    """Calculates order execution quantity and enforces pre-trade position limits."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def calculate_order_quantity(
        self,
        current_equity: float,
        current_cash: float,
        current_price: float,
        current_position_qty: float,
        expected_volatility: float = 0.0,
    ) -> float:
        """Calculate recommended order quantity for trade entry.

        Returns 0.0 if limits are exceeded or available capital is insufficient.
        """
        if current_price <= 0.0 or current_equity <= 0.0:
            return 0.0

        method = self.config.sizing_method
        val = self.config.sizing_value

        # 1. Base quantity derivation
        if method == SizingMethod.FIXED_QUANTITY:
            raw_qty = val
        elif method == SizingMethod.FIXED_NOTIONAL:
            raw_qty = val / current_price
        elif method == SizingMethod.PERCENT_CAPITAL:
            allocated_dollars = current_equity * val
            raw_qty = allocated_dollars / current_price
        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            # Scale inversely with volatility (annualized volatility base e.g. 0.40)
            vol = max(0.05, expected_volatility)
            target_risk_dollars = current_equity * val
            raw_qty = target_risk_dollars / (current_price * vol)
        else:
            raw_qty = (current_equity * 0.05) / current_price

        # 2. Check Available Cash (avoid negative cash without leverage)
        max_cash_qty = max(0.0, (current_cash * 0.99) / current_price)
        if self.config.max_leverage <= 1.0:
            raw_qty = min(raw_qty, max_cash_qty)

        # 3. Enforce Max Position Size Limit
        remaining_capacity = max(0.0, self.config.max_position_size - current_position_qty)
        qty = min(raw_qty, remaining_capacity)

        # 4. Enforce Max Notional Limit
        max_notional_qty = self.config.max_notional / current_price
        qty = min(qty, max_notional_qty)

        # 5. Enforce Max Leverage Limit
        max_leverage_qty = (current_equity * self.config.max_leverage) / current_price
        qty = min(qty, max_leverage_qty)

        return float(max(0.0, round(qty, 6)))
