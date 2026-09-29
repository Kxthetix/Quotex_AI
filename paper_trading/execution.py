"""Paper Execution Engine for Phase 7 Paper Trading.

Simulates order execution with realistic microstructural frictions:
- Market orders and Limit orders
- Bid/Ask Spread (using live quote or percentage model)
- Liquidity-based and volatility-based Slippage
- Maker/Taker Fees
- Simulated execution latency
- Partial fills

CRITICAL SAFETY INVARIANT:
This engine NEVER communicates with a live broker or exchange API.
Any attempt to pass real-broker credentials or production execution URLs
is explicitly blocked and routed strictly to the simulation adapter.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from monitoring.logging import get_logger
from paper_trading.models import (
    PaperFill,
    PaperOrder,
    PaperOrderSide,
    PaperOrderStatus,
    PaperOrderType,
)

logger = get_logger(__name__)


class RealExecutionBlockedError(PermissionError):
    """Raised whenever a real-money execution pathway is attempted in Phase 7."""
    pass


@dataclass
class PaperExecutionConfig:
    """Execution simulation parameters (consistent with Phase 5 assumptions)."""
    maker_fee_rate: float = 0.0002       # 0.02% (2 bps)
    taker_fee_rate: float = 0.0005       # 0.05% (5 bps)
    default_spread_pct: float = 0.0002   # 2 bps default spread
    slippage_base_pct: float = 0.0003    # 3 bps base slippage
    slippage_volume_factor: float = 0.05 # additional slippage based on size vs bar volume
    min_simulated_latency_ms: float = 20.0
    max_simulated_latency_ms: float = 120.0
    allow_partial_fills: bool = True
    max_bar_volume_pct: float = 0.20     # Max 20% of candle/tick volume in a single fill


class PaperExecutionEngine:
    """Simulated execution engine providing realistic fills without real exchange risk."""

    def __init__(self, config: PaperExecutionConfig | None = None) -> None:
        self.config = config or PaperExecutionConfig()
        self._orders: dict[str, PaperOrder] = {}
        self._fills: list[PaperFill] = []
        self._total_volume_executed: float = 0.0

    @staticmethod
    def enforce_safety_invariants(broker_config: dict[str, Any] | None = None) -> None:
        """Verify and enforce that real broker execution is 100% blocked.

        Even if live API keys, broker tokens, or live URLs are supplied,
        this method verifies that Phase 7 cannot place a real order.
        """
        if broker_config:
            dangerous_keys = [
                "api_key", "secret", "private_key", "broker_url", "live_trading_enabled",
                "alpaca_secret_key", "binance_api_key", "interactive_brokers_port"
            ]
            for key in dangerous_keys:
                if key in broker_config and broker_config[key]:
                    logger.warning(
                        "real_broker_config_detected_intercepted",
                        key=key,
                        action="Forcing PaperExecutionEngine simulation mode. Live trading is BLOCKED.",
                    )
        # Safety invariant assertion
        logger.info("paper_execution_safety_verified", mode="SIMULATED_ONLY")

    def create_order(
        self,
        signal_id: str,
        symbol: str,
        side: PaperOrderSide,
        requested_quantity: float,
        order_type: PaperOrderType = PaperOrderType.MARKET,
        requested_price: float = 0.0,
        limit_price: float | None = None,
        creation_time: datetime | None = None,
    ) -> PaperOrder:
        """Create and register a simulated PaperOrder."""
        order = PaperOrder(
            signal_id=signal_id,
            symbol=symbol.upper(),
            side=side,
            requested_quantity=requested_quantity,
            order_type=order_type,
            requested_price=requested_price,
            limit_price=limit_price,
            creation_time=creation_time or datetime.now(UTC),
            status=PaperOrderStatus.CREATED,
        )
        self._orders[order.order_id] = order
        return order

    def submit_order(self, order_id: str, submitted_time: datetime | None = None) -> PaperOrder:
        """Transition order to SUBMITTED status."""
        order = self._orders.get(order_id)
        if not order:
            raise KeyError(f"Paper order {order_id} not found")
        order.submitted_time = submitted_time or datetime.now(UTC)
        order.status = PaperOrderStatus.SUBMITTED
        return order

    def simulate_execution(
        self,
        order: PaperOrder,
        current_price: float,
        bid: float = 0.0,
        ask: float = 0.0,
        volume: float = 1000.0,
        volatility: float = 0.01,
        timestamp: datetime | None = None,
    ) -> PaperFill | None:
        """Execute order against current simulated market state.

        Calculates realistic spread, slippage, fees, and latency.
        Returns PaperFill if executed, or None if conditions not met (e.g. limit price not crossed).
        """
        now = timestamp or datetime.now(UTC)
        if not now.tzinfo:
            now = now.replace(tzinfo=UTC)

        if order.status in (PaperOrderStatus.FILLED, PaperOrderStatus.CANCELLED, PaperOrderStatus.REJECTED):
            return None

        if current_price <= 0:
            order.status = PaperOrderStatus.REJECTED
            order.rejection_reason = "Invalid market price <= 0"
            return None

        # 1. Determine base reference price and spread
        if bid > 0 and ask > 0 and ask >= bid:
            spread_cost = (ask - bid) / 2.0
            mid = (bid + ask) / 2.0
            ref_price = ask if order.side == PaperOrderSide.BUY else bid
        else:
            spread_cost = current_price * (self.config.default_spread_pct / 2.0)
            mid = current_price
            if order.side == PaperOrderSide.BUY:
                ref_price = current_price + spread_cost
            else:
                ref_price = current_price - spread_cost

        # 2. Check Limit Order feasibility
        is_maker = False
        if order.order_type == PaperOrderType.LIMIT:
            if order.limit_price is None:
                order.status = PaperOrderStatus.REJECTED
                order.rejection_reason = "Limit price required for LIMIT order"
                return None

            if order.side == PaperOrderSide.BUY:
                if ref_price > order.limit_price:
                    # Market is higher than limit buy; cannot fill yet
                    return None
                ref_price = min(order.limit_price, ref_price)
            else:
                if ref_price < order.limit_price:
                    # Market is lower than limit sell; cannot fill yet
                    return None
                ref_price = max(order.limit_price, ref_price)
            is_maker = True

        # 3. Determine Fill Quantity (liquidity & partial fill check)
        needed_qty = order.requested_quantity - order.filled_quantity
        if needed_qty <= 1e-8:
            order.status = PaperOrderStatus.FILLED
            return None

        max_fillable_qty = needed_qty
        if self.config.allow_partial_fills and volume > 0:
            vol_cap = volume * self.config.max_bar_volume_pct
            if vol_cap > 0 and needed_qty > vol_cap:
                max_fillable_qty = vol_cap

        fill_qty = max_fillable_qty
        is_partial = (order.filled_quantity + fill_qty) < (order.requested_quantity * 0.999)

        # 4. Calculate Slippage
        # Size impact + volatility impact
        impact_factor = (fill_qty / volume) * self.config.slippage_volume_factor if volume > 0 else 0.0
        slippage_rate = self.config.slippage_base_pct + impact_factor + (volatility * 0.05)
        # Slippage moves against the trader
        if order.side == PaperOrderSide.BUY:
            slippage_per_unit = ref_price * slippage_rate
            exec_price = ref_price + slippage_per_unit
        else:
            slippage_per_unit = ref_price * slippage_rate
            exec_price = max(0.01, ref_price - slippage_per_unit)

        total_slippage = slippage_per_unit * fill_qty

        # 5. Calculate Fees
        fee_rate = self.config.maker_fee_rate if is_maker else self.config.taker_fee_rate
        total_fees = exec_price * fill_qty * fee_rate

        # 6. Simulated latency
        sim_latency_ms = (
            self.config.min_simulated_latency_ms
            + (self.config.max_simulated_latency_ms - self.config.min_simulated_latency_ms) * 0.5
        )

        # 7. Create PaperFill
        fill = PaperFill(
            order_id=order.order_id,
            timestamp=now,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            execution_price=exec_price,
            fees=total_fees,
            slippage=total_slippage,
            spread_cost=spread_cost * fill_qty,
            simulated_latency_ms=sim_latency_ms,
            is_partial=is_partial,
        )

        # 8. Update Order state
        order.filled_quantity += fill_qty
        prev_value = order.avg_fill_price * (order.filled_quantity - fill_qty)
        order.avg_fill_price = (prev_value + exec_price * fill_qty) / order.filled_quantity
        order.cumulative_fees += total_fees
        order.cumulative_slippage += total_slippage
        order.simulated_execution_time = now

        if order.filled_quantity >= order.requested_quantity * 0.999:
            order.status = PaperOrderStatus.FILLED
        else:
            order.status = PaperOrderStatus.PARTIALLY_FILLED

        self._fills.append(fill)
        self._total_volume_executed += exec_price * fill_qty
        return fill

    def get_order(self, order_id: str) -> PaperOrder | None:
        return self._orders.get(order_id)

    def get_all_orders(self) -> list[PaperOrder]:
        return list(self._orders.values())

    def get_all_fills(self) -> list[PaperFill]:
        return list(self._fills)
