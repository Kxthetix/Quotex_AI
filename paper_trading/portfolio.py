"""Paper Portfolio Manager and Trade Ledger for Phase 7.

Maintains simulated positions, cash balances, equity calculations,
and generates completed PaperTradeRecords upon position closure or reduction.

NEVER connected to a real brokerage account.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from monitoring.logging import get_logger
from paper_trading.models import (
    ExitReason,
    PaperFill,
    PaperOrderSide,
    PaperPortfolio,
    PaperPosition,
    PaperTradeRecord,
)

logger = get_logger(__name__)


class PaperPortfolioManager:
    """Manages cash balances, active positions, portfolio valuation, and the Paper Trade Ledger."""

    def __init__(self, initial_capital: float = 10000.0) -> None:
        self.portfolio = PaperPortfolio(
            initial_capital=initial_capital,
            cash=initial_capital,
            peak_equity=initial_capital,
        )
        self.trade_ledger: list[PaperTradeRecord] = []
        self._entry_details: dict[str, dict[str, Any]] = {}

    def get_position(self, symbol: str) -> PaperPosition:
        """Get or initialize position for symbol."""
        sym = symbol.upper()
        if sym not in self.portfolio.positions:
            self.portfolio.positions[sym] = PaperPosition(symbol=sym)
        return self.portfolio.positions[sym]

    def update_market_price(self, symbol: str, current_price: float, timestamp: datetime | None = None) -> None:
        """Update current mark price for unrealized P&L calculation."""
        pos = self.get_position(symbol)
        pos.current_price = current_price
        pos.last_update = timestamp or datetime.now(UTC)
        self.portfolio.update_peak()

    def process_fill(
        self,
        fill: PaperFill,
        signal_id: str = "",
        exit_reason: ExitReason = ExitReason.SIGNAL,
        regime: str = "SIDEWAYS",
        model_version: str = "v1.0.0",
    ) -> PaperTradeRecord | None:
        """Process an executed PaperFill into cash, position state, and trade ledger.

        Returns PaperTradeRecord if position closure or reduction occurred.
        """
        pos = self.get_position(fill.symbol)
        now = fill.timestamp or datetime.now(UTC)
        is_buy = fill.side == PaperOrderSide.BUY

        # Track fees and slippage at portfolio level
        self.portfolio.total_fees += fill.fees
        self.portfolio.total_slippage += fill.slippage

        closed_trade: PaperTradeRecord | None = None

        # Case 1: Currently FLAT -> Open new position
        if pos.side == "FLAT" or pos.quantity <= 1e-8:
            pos.side = "LONG" if is_buy else "SHORT"
            pos.quantity = fill.quantity
            pos.avg_entry_price = fill.execution_price
            pos.current_price = fill.execution_price
            pos.entry_time = now
            pos.last_update = now
            pos.cumulative_fees = fill.fees
            pos.realized_pnl = 0.0

            # Cash impact: buying deducts cash, selling adds cash
            notional = fill.execution_price * fill.quantity
            if is_buy:
                self.portfolio.cash -= (notional + fill.fees)
            else:
                self.portfolio.cash += (notional - fill.fees)

            self._entry_details[pos.symbol] = {
                "signal_id": signal_id,
                "order_id": fill.order_id,
                "slippage": fill.slippage,
                "fees": fill.fees,
                "regime": regime,
                "model_version": model_version,
            }
            self.portfolio.update_peak()
            return None

        # Case 2: Fill is in SAME direction -> Increase existing position
        is_same_direction = (pos.side == "LONG" and is_buy) or (pos.side == "SHORT" and not is_buy)
        if is_same_direction:
            prev_value = pos.avg_entry_price * pos.quantity
            added_value = fill.execution_price * fill.quantity
            new_qty = pos.quantity + fill.quantity
            pos.avg_entry_price = (prev_value + added_value) / new_qty
            pos.quantity = new_qty
            pos.current_price = fill.execution_price
            pos.last_update = now
            pos.cumulative_fees += fill.fees

            notional = fill.execution_price * fill.quantity
            if is_buy:
                self.portfolio.cash -= (notional + fill.fees)
            else:
                self.portfolio.cash += (notional - fill.fees)

            if pos.symbol in self._entry_details:
                self._entry_details[pos.symbol]["fees"] += fill.fees
                self._entry_details[pos.symbol]["slippage"] += fill.slippage

            self.portfolio.update_peak()
            return None

        # Case 3: Fill is in OPPOSITE direction -> Reduce or Close position
        closing_qty = min(pos.quantity, fill.quantity)
        entry_price = pos.avg_entry_price
        exit_price = fill.execution_price

        # Calculate gross & net PnL for the closed portion
        if pos.side == "LONG":
            gross_pnl = closing_qty * (exit_price - entry_price)
        else:
            gross_pnl = closing_qty * (entry_price - exit_price)

        entry_meta = self._entry_details.get(pos.symbol, {})
        entry_fees = (entry_meta.get("fees", 0.0) / pos.quantity) * closing_qty if pos.quantity > 0 else 0.0
        total_trade_fees = entry_fees + fill.fees
        net_pnl = gross_pnl - total_trade_fees
        return_pct = (net_pnl / (entry_price * closing_qty)) if (entry_price * closing_qty) > 0 else 0.0

        holding_period = (now - pos.entry_time).total_seconds()

        # Update cash
        notional = exit_price * closing_qty
        if is_buy:  # closing a SHORT
            self.portfolio.cash -= (notional + fill.fees)
        else:  # closing a LONG
            self.portfolio.cash += (notional - fill.fees)

        pos.realized_pnl += net_pnl
        self.portfolio.cumulative_realized_pnl += net_pnl

        # Create PaperTradeRecord
        closed_trade = PaperTradeRecord(
            signal_id=signal_id or entry_meta.get("signal_id", ""),
            order_id=fill.order_id,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=closing_qty,
            fees=total_trade_fees,
            slippage=fill.slippage + entry_meta.get("slippage", 0.0),
            pnl=net_pnl,
            return_pct=return_pct,
            entry_time=pos.entry_time,
            exit_time=now,
            holding_period_seconds=holding_period,
            exit_reason=exit_reason,
            regime=entry_meta.get("regime", regime),
            model_version=entry_meta.get("model_version", model_version),
        )
        self.trade_ledger.append(closed_trade)

        # Update remaining position
        remaining_qty = pos.quantity - closing_qty
        if remaining_qty <= 1e-8:
            pos.side = "FLAT"
            pos.quantity = 0.0
            pos.avg_entry_price = 0.0
            pos.current_price = exit_price
            pos.last_update = now
            self._entry_details.pop(pos.symbol, None)
        else:
            pos.quantity = remaining_qty
            pos.current_price = exit_price
            pos.last_update = now

        self.portfolio.update_peak()
        return closed_trade

    def get_trade_ledger(self) -> list[PaperTradeRecord]:
        return list(self.trade_ledger)
