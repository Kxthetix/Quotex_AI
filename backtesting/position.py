"""Position Engine for Phase 5.

Manages open positions, average entry prices, position increases/reductions,
unrealized and realized P&L, and trade record generation upon position closure.
"""

from datetime import datetime

from analysis.regime.models import MarketRegime
from backtesting.models import ExitReason, Fill, OrderSide, Position, PositionSide, TradeRecord


class PositionEngine:
    """Manages active trading positions and lifecycle trade records."""

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self.position = Position(symbol=symbol, side=PositionSide.FLAT, quantity=0.0)
        self.closed_trades: list[TradeRecord] = []
        self._entry_slippage: float = 0.0
        self._entry_spread: float = 0.0
        self._entry_fees: float = 0.0
        self._regime_at_entry: MarketRegime = MarketRegime.SIDEWAYS
        self._model_version: str = "v1.0.0"
        self._feature_version: str = "v1.0.0"

    def update_market_price(self, current_price: float, timestamp: datetime) -> None:
        """Update current mark price and timestamp for unrealized P&L calculation."""
        self.position.current_price = current_price
        self.position.last_update_time = timestamp

    def apply_fill(
        self,
        fill: Fill,
        exit_reason: ExitReason = ExitReason.SIGNAL,
        regime: MarketRegime = MarketRegime.SIDEWAYS,
        model_version: str = "v1.0.0",
        feature_version: str = "v1.0.0",
    ) -> TradeRecord | None:
        """Process an execution fill and update position state accordingly.

        Returns a TradeRecord if a position reduction or closure occurred.
        """
        fill_side = fill.side
        pos = self.position
        closed_trade: TradeRecord | None = None

        # Case 1: Position is currently FLAT -> Open new position
        if pos.side == PositionSide.FLAT or pos.quantity <= 1e-8:
            pos.side = PositionSide.LONG if fill_side == OrderSide.BUY else PositionSide.SHORT
            pos.quantity = fill.quantity
            pos.avg_entry_price = fill.price
            pos.current_price = fill.price
            pos.entry_time = fill.timestamp
            pos.last_update_time = fill.timestamp
            pos.cumulative_fees = fill.fee
            pos.realized_pnl = 0.0

            self._entry_fees = fill.fee
            self._entry_slippage = fill.slippage_cost
            self._entry_spread = fill.spread_cost
            self._regime_at_entry = regime
            self._model_version = model_version
            self._feature_version = feature_version
            return None

        # Case 2: Fill is in the SAME direction -> Increase existing position
        is_increasing = (pos.side == PositionSide.LONG and fill_side == OrderSide.BUY) or (
            pos.side == PositionSide.SHORT and fill_side == OrderSide.SELL
        )

        if is_increasing:
            total_qty = pos.quantity + fill.quantity
            pos.avg_entry_price = (
                (pos.avg_entry_price * pos.quantity) + (fill.price * fill.quantity)
            ) / total_qty
            pos.quantity = total_qty
            pos.cumulative_fees += fill.fee
            self._entry_fees += fill.fee
            self._entry_slippage += fill.slippage_cost
            self._entry_spread += fill.spread_cost
            return None

        # Case 3: Fill is in the OPPOSITE direction -> Reduce or Close position
        closed_qty = min(pos.quantity, fill.quantity)

        # Calculate gross P&L on closed portion
        if pos.side == PositionSide.LONG:
            gross_pnl = closed_qty * (fill.price - pos.avg_entry_price)
        else:
            gross_pnl = closed_qty * (pos.avg_entry_price - fill.price)

        total_trade_fees = (self._entry_fees * (closed_qty / pos.quantity)) + fill.fee
        total_trade_slippage = (
            self._entry_slippage * (closed_qty / pos.quantity)
        ) + fill.slippage_cost
        total_trade_spread = (self._entry_spread * (closed_qty / pos.quantity)) + fill.spread_cost
        net_pnl = gross_pnl - total_trade_fees

        ret_pct = (
            (net_pnl / (pos.avg_entry_price * closed_qty))
            if (pos.avg_entry_price * closed_qty) > 0
            else 0.0
        )
        holding_sec = max(0.0, (fill.timestamp - pos.entry_time).total_seconds())

        trade_record = TradeRecord(
            trade_id=f"tr_{len(self.closed_trades) + 1:04d}",
            symbol=self.symbol,
            side=pos.side,
            entry_time=pos.entry_time,
            exit_time=fill.timestamp,
            entry_price=pos.avg_entry_price,
            exit_price=fill.price,
            quantity=closed_qty,
            gross_pnl=float(gross_pnl),
            fees=float(total_trade_fees),
            slippage=float(total_trade_slippage),
            spread=float(total_trade_spread),
            net_pnl=float(net_pnl),
            return_pct=float(ret_pct),
            holding_period_seconds=holding_sec,
            exit_reason=exit_reason,
            model_version=self._model_version,
            feature_version=self._feature_version,
            regime_at_entry=self._regime_at_entry,
        )

        self.closed_trades.append(trade_record)
        closed_trade = trade_record

        # Update remaining position
        rem_qty = pos.quantity - closed_qty
        if rem_qty <= 1e-8:
            # Fully closed
            pos.side = PositionSide.FLAT
            pos.quantity = 0.0
            pos.avg_entry_price = 0.0
            pos.cumulative_fees = 0.0
            self._entry_fees = 0.0
            self._entry_slippage = 0.0
            self._entry_spread = 0.0

            # If the fill had leftover quantity in opposite direction, open position in that side
            residual = fill.quantity - closed_qty
            if residual > 1e-8:
                pos.side = PositionSide.LONG if fill_side == OrderSide.BUY else PositionSide.SHORT
                pos.quantity = residual
                pos.avg_entry_price = fill.price
                pos.entry_time = fill.timestamp
                self._entry_fees = fill.fee * (residual / fill.quantity)
                self._entry_slippage = fill.slippage_cost * (residual / fill.quantity)
                self._entry_spread = fill.spread_cost * (residual / fill.quantity)
                self._regime_at_entry = regime
        else:
            pos.quantity = rem_qty
            pos.cumulative_fees -= self._entry_fees * (closed_qty / (closed_qty + rem_qty))

        return closed_trade
