"""Portfolio Engine for Phase 5.

Maintains multi-asset cash, equity, exposures, and historical equity curve snapshots.
"""

from datetime import datetime

from backtesting.models import EquityPoint, Fill, OrderSide, Position, PositionSide
from backtesting.position import PositionEngine


class PortfolioEngine:
    """Manages cash balance, active positions, portfolio valuation, and equity curve."""

    def __init__(self, initial_capital: float = 10000.0) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.position_engines: dict[str, PositionEngine] = {}
        self.equity_curve: list[EquityPoint] = []
        self.cumulative_fees: float = 0.0
        self.cumulative_realized_pnl: float = 0.0
        self.peak_equity: float = float(initial_capital)

    def get_position_engine(self, symbol: str) -> PositionEngine:
        """Retrieve or create position engine for symbol."""
        if symbol not in self.position_engines:
            self.position_engines[symbol] = PositionEngine(symbol=symbol)
        return self.position_engines[symbol]

    def get_position(self, symbol: str) -> Position:
        """Get current position for symbol."""
        return self.get_position_engine(symbol).position

    def process_fill(self, fill: Fill, **kwargs) -> None:
        """Process fill into cash balances and position state."""
        pe = self.get_position_engine(fill.symbol)
        closed_trade = pe.apply_fill(fill, **kwargs)

        notional = fill.price * fill.quantity
        if fill.side == OrderSide.BUY:
            self.cash -= notional + fill.fee
        else:
            self.cash += notional - fill.fee

        self.cumulative_fees += fill.fee

        if closed_trade is not None:
            self.cumulative_realized_pnl += closed_trade.net_pnl

    def update_valuation(
        self, current_prices: dict[str, float], timestamp: datetime
    ) -> EquityPoint:
        """Update market valuations across all symbols and append an EquityPoint snapshot."""
        total_market_value = 0.0
        total_unrealized_pnl = 0.0
        gross_exposure = 0.0
        net_exposure = 0.0

        for symbol, pe in self.position_engines.items():
            price = current_prices.get(symbol, pe.position.current_price)
            pe.update_market_price(price, timestamp)
            pos = pe.position

            if pos.side != PositionSide.FLAT and pos.quantity > 0:
                abs_mv = pos.quantity * price
                gross_exposure += abs_mv
                if pos.side == PositionSide.LONG:
                    net_exposure += abs_mv
                    total_market_value += abs_mv
                else:
                    net_exposure -= abs_mv
                    total_market_value -= abs_mv
                total_unrealized_pnl += pos.unrealized_pnl

        # In cash accounting for spot assets: Equity = Cash + Total Market Value
        # For derivatives/margin: Equity = Initial + Realized + Unrealized - Fees
        # Here we use pure asset-liability accounting: Equity = Cash + Market Value
        current_equity = self.cash + total_market_value

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        dd = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        point = EquityPoint(
            timestamp=timestamp,
            equity=float(current_equity),
            cash=float(self.cash),
            market_value=float(total_market_value),
            gross_exposure=float(gross_exposure),
            net_exposure=float(net_exposure),
            realized_pnl=float(self.cumulative_realized_pnl),
            unrealized_pnl=float(total_unrealized_pnl),
            cumulative_fees=float(self.cumulative_fees),
            drawdown=float(dd),
        )

        self.equity_curve.append(point)
        return point
