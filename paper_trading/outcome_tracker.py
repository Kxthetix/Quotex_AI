"""Outcome Tracker for Phase 7 Paper Trading.

Evaluates predictions once their configured evaluation horizon has elapsed.
Calculates:
- Realized forward return
- Maximum Favorable Excursion (MFE)
- Maximum Adverse Excursion (MAE)
- Prediction direction accuracy and prediction error

CRITICAL INVARIANT:
Distinguishes Prediction Outcome from Trade Outcome.
Predictions are evaluated unconditionally, even if the signal was VETOED by the
Risk Engine, unexecuted, or rejected by execution limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from data.ingestion.models import Candle
from monitoring.logging import get_logger
from paper_trading.models import (
    MonitoringEvent,
    MonitoringEventType,
    MonitoringSeverity,
    PaperTradeRecord,
    PredictionRecord,
    PredictionStatus,
)
from paper_trading.prediction_ledger import PredictionLedger

logger = get_logger(__name__)


@dataclass
class QualityBreakdown:
    """Distinct multidimensional evaluation of system layers (Section 24)."""
    prediction_quality: dict[str, Any] = field(default_factory=dict)
    signal_quality: dict[str, Any] = field(default_factory=dict)
    risk_quality: dict[str, Any] = field(default_factory=dict)
    execution_quality: dict[str, Any] = field(default_factory=dict)
    portfolio_result: dict[str, Any] = field(default_factory=dict)


class OutcomeTracker:
    """Tracks and evaluates prediction horizons and distinguishes prediction quality from trade quality."""

    def __init__(
        self,
        prediction_ledger: PredictionLedger,
        default_horizon_seconds: float = 900.0,  # 15 minutes default
    ) -> None:
        self.ledger = prediction_ledger
        self.default_horizon_seconds = default_horizon_seconds
        # Historical candle cache per symbol for MFE/MAE calculation: symbol -> list[Candle]
        self._candle_history: dict[str, list[Candle]] = {}
        self._events: list[MonitoringEvent] = []

    def record_candle(self, candle: Candle) -> None:
        """Cache incoming candle for MFE/MAE excursion calculations."""
        sym = candle.symbol.upper()
        if sym not in self._candle_history:
            self._candle_history[sym] = []
        self._candle_history[sym].append(candle)
        # Keep last 1,000 candles per symbol in memory
        if len(self._candle_history[sym]) > 1000:
            self._candle_history[sym] = self._candle_history[sym][-1000:]

    def check_pending_evaluations(
        self,
        current_time: datetime | None = None,
        current_prices: dict[str, float] | None = None,
    ) -> list[PredictionRecord]:
        """Check all pending predictions and evaluate those whose horizon has elapsed."""
        now = current_time or datetime.now(UTC)
        if not now.tzinfo:
            now = now.replace(tzinfo=UTC)

        evaluated: list[PredictionRecord] = []
        pending = self.ledger.get_pending_predictions()

        for pred in pending:
            pred_time = pred.timestamp if pred.timestamp.tzinfo else pred.timestamp.replace(tzinfo=UTC)
            horizon_sec = pred.evaluation_horizon_seconds or self.default_horizon_seconds
            elapsed_sec = (now - pred_time).total_seconds()

            if elapsed_sec >= horizon_sec:
                # Horizon has elapsed -> evaluate!
                eval_record = self.evaluate_prediction(pred, now, current_prices)
                if eval_record:
                    evaluated.append(eval_record)

        return evaluated

    def evaluate_prediction(
        self,
        prediction: PredictionRecord,
        evaluation_time: datetime,
        current_prices: dict[str, float] | None = None,
    ) -> PredictionRecord | None:
        """Compute realized movement, MFE, MAE, and append evaluation to ledger."""
        sym = prediction.symbol.upper()
        entry_price = prediction.entry_price

        # Determine price at evaluation horizon
        eval_price = 0.0
        if current_prices and sym in current_prices:
            eval_price = current_prices[sym]
        else:
            candles = self._candle_history.get(sym, [])
            if candles:
                eval_price = candles[-1].close

        if entry_price <= 0 or eval_price <= 0:
            logger.warning(
                "cannot_evaluate_prediction_missing_prices",
                prediction_id=prediction.prediction_id,
                entry_price=entry_price,
                eval_price=eval_price,
            )
            return None

        # 1. Calculate Realized Return
        realized_return = (eval_price - entry_price) / entry_price

        # 2. Calculate MFE and MAE from candle history across prediction window
        pred_time = prediction.timestamp if prediction.timestamp.tzinfo else prediction.timestamp.replace(tzinfo=UTC)
        relevant_candles = [
            c for c in self._candle_history.get(sym, [])
            if (c.end_time if c.end_time.tzinfo else c.end_time.replace(tzinfo=UTC)) >= pred_time
        ]

        if relevant_candles:
            max_high = max(c.high for c in relevant_candles)
            min_low = min(c.low for c in relevant_candles)
        else:
            max_high = max(entry_price, eval_price)
            min_low = min(entry_price, eval_price)

        if prediction.signal_action == "BUY":
            mfe = (max_high - entry_price) / entry_price
            mae = (entry_price - min_low) / entry_price
            # Correct if market moved upwards as expected
            is_correct = realized_return > 0
        elif prediction.signal_action == "SELL":
            mfe = (entry_price - min_low) / entry_price
            mae = (max_high - entry_price) / entry_price
            # Correct if market moved downwards as expected
            is_correct = realized_return < 0
        else:
            # NO_TRADE
            mfe = abs(eval_price - entry_price) / entry_price
            mae = mfe
            is_correct = abs(realized_return) < 0.005  # Correct if market remained quiet

        actual_outcome = "CORRECT" if is_correct else "INCORRECT"
        if abs(realized_return) < 0.0005:
            actual_outcome = "NEUTRAL"

        # 3. Prediction Error
        # If expected return was provided, difference between realized and expected
        prediction_error = abs(realized_return - prediction.expected_return)

        # 4. Append evaluation to ledger
        evaluated_pred = self.ledger.append_evaluation(
            prediction_id=prediction.prediction_id,
            evaluation_time=evaluation_time,
            realized_return=realized_return,
            mfe=mfe,
            mae=mae,
            actual_outcome=actual_outcome,
            prediction_error=prediction_error,
            metadata={"eval_price": eval_price, "candle_count": len(relevant_candles)},
        )

        event = MonitoringEvent(
            severity=MonitoringSeverity.INFO,
            component="outcome_tracker",
            event_type=MonitoringEventType.PREDICTION_EVALUATED,
            message=f"Prediction {prediction.prediction_id} evaluated: {actual_outcome} (Return: {realized_return:+.2%})",
            metadata={
                "prediction_id": prediction.prediction_id,
                "outcome": actual_outcome,
                "realized_return": realized_return,
                "mfe": mfe,
                "mae": mae,
            },
        )
        self._events.append(event)
        return evaluated_pred

    def analyze_quality(
        self,
        trade_ledger: list[PaperTradeRecord],
        risk_veto_count: int,
        total_signals: int,
        portfolio_equity: float,
        initial_capital: float,
    ) -> QualityBreakdown:
        """Produce separate multidimensional quality metrics (Section 24)."""
        evaluated_preds = self.ledger.get_evaluated_predictions()

        # A. Prediction Quality
        total_eval = len(evaluated_preds)
        correct_count = sum(1 for p in evaluated_preds if p.actual_outcome == "CORRECT")
        pred_accuracy = (correct_count / total_eval) if total_eval > 0 else 0.0
        avg_mfe = (sum(p.mfe or 0.0 for p in evaluated_preds) / total_eval) if total_eval > 0 else 0.0
        avg_mae = (sum(p.mae or 0.0 for p in evaluated_preds) / total_eval) if total_eval > 0 else 0.0
        avg_err = (sum(p.prediction_error or 0.0 for p in evaluated_preds) / total_eval) if total_eval > 0 else 0.0

        pred_quality = {
            "total_evaluated_predictions": total_eval,
            "correct_predictions": correct_count,
            "prediction_accuracy": round(pred_accuracy, 4),
            "average_mfe": round(avg_mfe, 6),
            "average_mae": round(avg_mae, 6),
            "average_prediction_error": round(avg_err, 6),
        }

        # B. Signal Quality
        signals_by_action = self.ledger.count_by_action()
        signal_quality = {
            "total_signals": total_signals,
            "action_counts": signals_by_action,
            "no_trade_ratio": round(signals_by_action.get("NO_TRADE", 0) / max(1, total_signals), 4),
        }

        # C. Risk Quality
        risk_veto_rate = (risk_veto_count / max(1, total_signals))
        risk_quality = {
            "risk_veto_count": risk_veto_count,
            "risk_veto_rate": round(risk_veto_rate, 4),
            "circuit_breaker_active": False,
        }

        # D. Execution Quality
        total_trades = len(trade_ledger)
        total_fees = sum(t.fees for t in trade_ledger)
        total_slippage = sum(t.slippage for t in trade_ledger)
        exec_quality = {
            "total_trades": total_trades,
            "total_fees": round(total_fees, 4),
            "total_slippage": round(total_slippage, 4),
            "avg_slippage_per_trade": round(total_slippage / max(1, total_trades), 6),
        }

        # E. Portfolio Result
        net_pnl = portfolio_equity - initial_capital
        return_pct = net_pnl / initial_capital if initial_capital > 0 else 0.0
        winning_trades = sum(1 for t in trade_ledger if t.pnl > 0)
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

        portfolio_result = {
            "initial_capital": initial_capital,
            "current_equity": round(portfolio_equity, 2),
            "net_pnl": round(net_pnl, 2),
            "total_return_pct": round(return_pct, 4),
            "trade_win_rate": round(win_rate, 4),
        }

        return QualityBreakdown(
            prediction_quality=pred_quality,
            signal_quality=signal_quality,
            risk_quality=risk_quality,
            execution_quality=exec_quality,
            portfolio_result=portfolio_result,
        )

    def get_events(self) -> list[MonitoringEvent]:
        return list(self._events)
