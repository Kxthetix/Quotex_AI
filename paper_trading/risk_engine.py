"""Risk Engine for Phase 7 — Paper Trading.

Independent risk evaluation layer between Signal Engine and Paper Execution.
Evaluates signals against portfolio constraints, drawdown limits, exposure limits,
and circuit breakers before permitting simulated execution.

The Risk Engine can:
- APPROVE: Allow execution at requested size
- MODIFY: Reduce size to fit within limits
- VETO: Block execution entirely

Circuit breakers:
- Max daily loss
- Max drawdown
- Max exposure
- Abnormal trade frequency
- Manual emergency kill switch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from monitoring.logging import get_logger
from paper_trading.models import (
    MonitoringEvent,
    MonitoringEventType,
    MonitoringSeverity,
    PaperPortfolio,
    PaperPosition,
    RiskDecision,
    RiskEvaluation,
)
from signals.models import SignalAction, TradingSignal

logger = get_logger(__name__)


@dataclass
class RiskConfig:
    """Externalized risk parameters."""
    version: str = "v1.0.0"
    max_position_pct: float = 0.10           # Max 10% of equity per position
    max_total_exposure_pct: float = 0.50     # Max 50% of equity total exposure
    max_daily_loss_pct: float = 0.03         # Max 3% daily loss
    max_drawdown_pct: float = 0.10           # Max 10% drawdown from peak
    max_trades_per_hour: int = 20            # Circuit breaker: abnormal frequency
    min_signal_conviction: float = 0.10      # Minimum conviction to trade
    max_correlated_positions: int = 3        # Max positions in same direction


class CircuitBreakerState:
    """Tracks circuit breaker activation status."""

    def __init__(self) -> None:
        self.is_tripped: bool = False
        self.reason: str = ""
        self.tripped_at: datetime | None = None
        self.daily_loss: float = 0.0
        self.daily_trade_count: int = 0
        self.last_reset_date: str = ""
        self._events: list[MonitoringEvent] = []

    def trip(self, reason: str) -> None:
        self.is_tripped = True
        self.reason = reason
        self.tripped_at = datetime.now(UTC)
        event = MonitoringEvent(
            severity=MonitoringSeverity.CRITICAL,
            component="risk_engine",
            event_type=MonitoringEventType.CIRCUIT_BREAKER,
            message=f"Circuit breaker tripped: {reason}",
            metadata={"reason": reason, "tripped_at": self.tripped_at.isoformat()},
        )
        self._events.append(event)
        logger.error("circuit_breaker_tripped", reason=reason)

    def reset(self) -> None:
        self.is_tripped = False
        self.reason = ""
        self.tripped_at = None
        logger.info("circuit_breaker_reset")

    def reset_daily_counters(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self.last_reset_date != today:
            self.daily_loss = 0.0
            self.daily_trade_count = 0
            self.last_reset_date = today

    def get_events(self) -> list[MonitoringEvent]:
        return list(self._events)


class RiskEngine:
    """Independent risk evaluation gate.

    Sits between Signal Engine and Paper Execution. Evaluates every signal
    against portfolio constraints and circuit breakers.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.breaker = CircuitBreakerState()

    def evaluate(
        self,
        signal: TradingSignal,
        portfolio: PaperPortfolio,
        current_price: float,
    ) -> RiskEvaluation:
        """Evaluate a signal against risk constraints.

        Returns RiskEvaluation with APPROVE, MODIFY, or VETO decision.
        """
        self.breaker.reset_daily_counters()
        checks: dict[str, Any] = {}

        # 0. Circuit breaker check
        if self.breaker.is_tripped:
            return RiskEvaluation(
                decision=RiskDecision.VETO,
                reason=f"Circuit breaker active: {self.breaker.reason}",
                checks={"circuit_breaker": True, "reason": self.breaker.reason},
            )

        # 1. NO_TRADE signals pass through (no risk evaluation needed)
        if signal.action == SignalAction.NO_TRADE:
            return RiskEvaluation(
                decision=RiskDecision.APPROVE,
                reason="NO_TRADE signal — no execution required",
                checks={"no_trade": True},
            )

        # Check existing position for risk reduction vs risk expansion
        existing_pos = portfolio.positions.get(signal.symbol)
        is_closing_or_reducing = False
        if existing_pos and existing_pos.side != "FLAT" and existing_pos.quantity > 0:
            if existing_pos.side == "LONG" and signal.action == SignalAction.SELL:
                is_closing_or_reducing = True
            elif existing_pos.side == "SHORT" and signal.action == SignalAction.BUY:
                is_closing_or_reducing = True

        # If reducing risk/closing position, allow execution regardless of exposure/drawdown
        if is_closing_or_reducing:
            close_qty = existing_pos.quantity if existing_pos else 0.0
            return RiskEvaluation(
                decision=RiskDecision.APPROVE,
                original_quantity=close_qty,
                approved_quantity=close_qty,
                reason="Risk-reducing order approved to close/reduce exposure",
                checks={"risk_reduction": True},
            )

        # 2. Minimum conviction threshold
        if signal.conviction < self.config.min_signal_conviction:
            checks["conviction_check"] = {
                "conviction": signal.conviction,
                "minimum": self.config.min_signal_conviction,
            }
            return RiskEvaluation(
                decision=RiskDecision.VETO,
                reason=f"Conviction {signal.conviction:.4f} below minimum {self.config.min_signal_conviction}",
                checks=checks,
            )

        # 3. Drawdown check
        dd = portfolio.drawdown
        checks["drawdown"] = {"current": round(dd, 4), "max": self.config.max_drawdown_pct}
        if dd >= self.config.max_drawdown_pct:
            self.breaker.trip(f"Drawdown {dd:.2%} exceeds limit {self.config.max_drawdown_pct:.2%}")
            return RiskEvaluation(
                decision=RiskDecision.VETO,
                reason=f"Drawdown {dd:.2%} exceeds limit {self.config.max_drawdown_pct:.2%}",
                checks=checks,
            )

        # 4. Daily loss check
        checks["daily_loss"] = {
            "current": round(self.breaker.daily_loss, 2),
            "max_pct": self.config.max_daily_loss_pct,
        }
        equity = portfolio.total_equity
        if equity > 0:
            daily_loss_pct = abs(self.breaker.daily_loss) / equity
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                self.breaker.trip(f"Daily loss {daily_loss_pct:.2%} exceeds limit")
                return RiskEvaluation(
                    decision=RiskDecision.VETO,
                    reason=f"Daily loss limit {self.config.max_daily_loss_pct:.2%} exceeded",
                    checks=checks,
                )

        # 5. Trade frequency check
        self.breaker.daily_trade_count += 1
        checks["trade_frequency"] = {
            "count": self.breaker.daily_trade_count,
            "max_per_hour": self.config.max_trades_per_hour,
        }
        if self.breaker.daily_trade_count > self.config.max_trades_per_hour * 24:
            self.breaker.trip("Abnormal trade frequency")
            return RiskEvaluation(
                decision=RiskDecision.VETO,
                reason="Abnormal trade frequency detected",
                checks=checks,
            )

        # 6. Position sizing
        if current_price <= 0:
            return RiskEvaluation(
                decision=RiskDecision.VETO,
                reason="Invalid current price <= 0",
                checks=checks,
            )

        max_position_value = equity * self.config.max_position_pct
        max_quantity = max_position_value / current_price

        # 7. Total exposure check
        current_exposure = portfolio.total_exposure
        max_total = equity * self.config.max_total_exposure_pct
        remaining_capacity = max(0.0, max_total - current_exposure)
        max_capacity_qty = remaining_capacity / current_price if current_price > 0 else 0.0
        max_quantity = min(max_quantity, max_capacity_qty)

        checks["exposure"] = {
            "current": round(current_exposure, 2),
            "max_total": round(max_total, 2),
            "remaining": round(remaining_capacity, 2),
        }

        if max_quantity <= 1e-8:
            return RiskEvaluation(
                decision=RiskDecision.VETO,
                reason="Maximum exposure reached; no capacity for new positions",
                checks=checks,
            )

        # 8. Sizing & conviction scaling
        requested_qty = max_position_value / current_price
        # Scale slightly by conviction if conviction is between min and 1.0
        approved_qty = min(requested_qty, max_quantity)

        if approved_qty < requested_qty * 0.99:
            decision = RiskDecision.MODIFY
            reason = f"Size reduced from {requested_qty:.6f} to {approved_qty:.6f} due to exposure limit"
        else:
            decision = RiskDecision.APPROVE
            reason = "Signal approved within risk limits"

        return RiskEvaluation(
            decision=decision,
            original_quantity=requested_qty,
            approved_quantity=approved_qty,
            reason=reason,
            checks=checks,
        )

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a completed trade's P&L for daily loss tracking."""
        self.breaker.reset_daily_counters()
        if pnl < 0:
            self.breaker.daily_loss += pnl
