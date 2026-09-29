"""Reporting, Validation Evaluation, and Backtest Comparison for Phase 7.

Generates:
1. End-of-Session Comprehensive Reports (Section 41)
2. Daily Paper-Trading Summary Reports (Section 42)
3. Backtest-vs-Paper Reality Comparison (Section 43)
4. Paper Validation Period Criteria Verification (Section 45)
5. Governance Evidence Package (NO automatic promotion - Section 46)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_trading.execution import PaperExecutionEngine
from paper_trading.models import (
    PaperTradingSession,
    PaperValidationCriteria,
    PredictionStatus,
)
from paper_trading.monitoring import SystemHealthMonitor
from paper_trading.outcome_tracker import OutcomeTracker
from paper_trading.portfolio import PaperPortfolioManager
from paper_trading.prediction_ledger import PredictionLedger
from paper_trading.risk_engine import RiskEngine


class PaperTradingReporter:
    """Produces auditable session summaries, daily reports, and backtest comparison analytics."""

    def __init__(self, reports_dir: Path | str = "reports/paper_trading") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_session_report(
        self,
        session: PaperTradingSession,
        portfolio_mgr: PaperPortfolioManager,
        execution_engine: PaperExecutionEngine,
        prediction_ledger: PredictionLedger,
        risk_engine: RiskEngine,
        health_monitor: SystemHealthMonitor,
        data_coverage_pct: float = 100.0,
        reconnect_count: int = 0,
    ) -> dict[str, Any]:
        """Generate comprehensive end-of-session report (Section 41)."""
        now = datetime.now(UTC)
        start_time = session.start_time if session.start_time.tzinfo else session.start_time.replace(tzinfo=UTC)
        runtime_seconds = (now - start_time).total_seconds()

        # Counts
        all_preds = prediction_ledger.get_all_predictions()
        action_counts = prediction_ledger.count_by_action()
        outcome_counts = prediction_ledger.count_by_outcome()
        orders = execution_engine.get_all_orders()
        fills = execution_engine.get_all_fills()
        trades = portfolio_mgr.get_trade_ledger()

        # Risk & Portfolio Metrics
        equity = portfolio_mgr.portfolio.total_equity
        initial = portfolio_mgr.portfolio.initial_capital
        net_pnl = equity - initial
        drawdown = portfolio_mgr.portfolio.drawdown
        total_fees = portfolio_mgr.portfolio.total_fees
        total_slippage = portfolio_mgr.portfolio.total_slippage

        winning_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = (winning_trades / len(trades)) if trades else 0.0

        report = {
            "session_metadata": session.to_dict(),
            "runtime_duration_seconds": round(runtime_seconds, 2),
            "data_coverage_pct": round(data_coverage_pct, 2),
            "reconnection_count": reconnect_count,
            "prediction_statistics": {
                "total_predictions": len(all_preds),
                "actions": action_counts,
                "outcomes": outcome_counts,
                "pending_evaluations": len(prediction_ledger.get_pending_predictions()),
            },
            "signals": {
                "total_signals": len(all_preds),
                "buy_count": action_counts.get("BUY", 0),
                "sell_count": action_counts.get("SELL", 0),
                "no_trade_count": action_counts.get("NO_TRADE", 0),
            },
            "risk_evaluation": {
                "circuit_breaker_tripped": risk_engine.breaker.is_tripped,
                "circuit_breaker_reason": risk_engine.breaker.reason,
                "daily_loss": round(risk_engine.breaker.daily_loss, 2),
                "trade_frequency_count": risk_engine.breaker.daily_trade_count,
            },
            "execution_statistics": {
                "orders_created": len(orders),
                "fills_executed": len(fills),
                "trades_completed": len(trades),
                "total_fees": round(total_fees, 4),
                "total_slippage": round(total_slippage, 4),
            },
            "portfolio_performance": {
                "initial_capital": initial,
                "current_equity": round(equity, 2),
                "net_pnl": round(net_pnl, 2),
                "total_return_pct": round(net_pnl / initial, 4) if initial > 0 else 0.0,
                "max_drawdown_pct": round(drawdown, 4),
                "trade_win_rate": round(win_rate, 4),
                "open_positions": {s: p.to_dict() for s, p in portfolio_mgr.portfolio.positions.items()},
            },
            "model_performance": health_monitor.performance_tracker.get_regime_performance(),
            "latency_diagnostics": health_monitor.latency_tracker.get_all_summary(),
            "error_and_anomalies": {
                "error_count": health_monitor.error_count,
                "model_anomaly": health_monitor.anomaly_detector.is_abnormal,
                "model_anomaly_reason": health_monitor.anomaly_detector.anomaly_reason,
            },
        }

        # Save to disk
        out_file = self.reports_dir / f"session_{session.session_id}_report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def generate_daily_report(
        self,
        portfolio_mgr: PaperPortfolioManager,
        prediction_ledger: PredictionLedger,
        health_monitor: SystemHealthMonitor,
        risk_engine: RiskEngine,
        date_str: str | None = None,
    ) -> dict[str, Any]:
        """Generate daily paper-trading summary report (Section 42)."""
        target_date = date_str or datetime.now(UTC).strftime("%Y-%m-%d")
        trades = portfolio_mgr.get_trade_ledger()
        daily_trades = [
            t for t in trades
            if t.exit_time.strftime("%Y-%m-%d") == target_date
        ]

        evaluated_preds = [
            p for p in prediction_ledger.get_evaluated_predictions()
            if p.evaluation_time and p.evaluation_time.strftime("%Y-%m-%d") == target_date
        ]

        net_pnl = sum(t.pnl for t in daily_trades)
        wins = sum(1 for t in daily_trades if t.pnl > 0)
        win_rate = (wins / len(daily_trades)) if daily_trades else 0.0

        daily_report = {
            "date": target_date,
            "prediction_count": len(prediction_ledger.get_all_predictions()),
            "daily_evaluated_predictions": len(evaluated_preds),
            "daily_trade_count": len(daily_trades),
            "net_pnl": round(net_pnl, 2),
            "win_rate": round(win_rate, 4),
            "portfolio_equity": round(portfolio_mgr.portfolio.total_equity, 2),
            "drawdown": round(portfolio_mgr.portfolio.drawdown, 4),
            "total_exposure": round(portfolio_mgr.portfolio.total_exposure, 2),
            "latency_p95_ms": health_monitor.latency_tracker.get_percentiles().get("p95", 0.0),
            "circuit_breaker_active": risk_engine.breaker.is_tripped,
            "system_health": str(health_monitor.determine_system_state(
                "CONNECTED", False, False, risk_engine.breaker.is_tripped
            )),
        }

        out_file = self.reports_dir / f"daily_report_{target_date}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(daily_report, f, indent=2, default=str)

        return daily_report

    def compare_with_backtest(
        self,
        paper_summary: dict[str, Any],
        backtest_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare live paper-trading results against historical Phase 5 backtest expectations (Section 43)."""
        paper_exec = paper_summary.get("execution_statistics", {})
        paper_port = paper_summary.get("portfolio_performance", {})
        paper_sig = paper_summary.get("signals", {})

        p_trades = paper_exec.get("trades_completed", 0)
        b_trades = backtest_metrics.get("total_trades", 0)

        p_win_rate = paper_port.get("trade_win_rate", 0.0)
        b_win_rate = backtest_metrics.get("win_rate", 0.0)

        p_fees = paper_exec.get("total_fees", 0.0)
        b_fees = backtest_metrics.get("total_fees", 0.0)

        p_slippage = paper_exec.get("total_slippage", 0.0)
        b_slippage = backtest_metrics.get("total_slippage", 0.0)

        comparison = {
            "trade_frequency": {
                "paper_trades": p_trades,
                "backtest_trades": b_trades,
                "delta": p_trades - b_trades,
            },
            "win_rate": {
                "paper_win_rate": round(p_win_rate, 4),
                "backtest_win_rate": round(b_win_rate, 4),
                "delta": round(p_win_rate - b_win_rate, 4),
            },
            "fee_comparison": {
                "paper_fees": round(p_fees, 4),
                "backtest_fees": round(b_fees, 4),
            },
            "slippage_comparison": {
                "paper_slippage": round(p_slippage, 4),
                "backtest_slippage": round(b_slippage, 4),
            },
            "governance_flag": (
                "FLAG_FOR_INVESTIGATION"
                if abs(p_win_rate - b_win_rate) > 0.15
                else "ALIGNED_WITH_BACKTEST"
            ),
        }
        return comparison

    def evaluate_validation_period(
        self,
        session_days: int,
        prediction_count: int,
        trade_count: int,
        data_coverage_pct: float,
        criteria: PaperValidationCriteria | None = None,
    ) -> dict[str, Any]:
        """Check if sufficient paper evidence exists for Phase 8 governance review (Section 45)."""
        crit = criteria or PaperValidationCriteria()
        checks = {
            "paper_days": {
                "observed": session_days,
                "required": crit.minimum_paper_days,
                "passed": session_days >= crit.minimum_paper_days,
            },
            "prediction_count": {
                "observed": prediction_count,
                "required": crit.minimum_predictions,
                "passed": prediction_count >= crit.minimum_predictions,
            },
            "trade_count": {
                "observed": trade_count,
                "required": crit.minimum_trades,
                "passed": trade_count >= crit.minimum_trades,
            },
            "data_coverage": {
                "observed": round(data_coverage_pct, 2),
                "required": crit.minimum_data_coverage_pct,
                "passed": data_coverage_pct >= crit.minimum_data_coverage_pct,
            },
        }
        all_passed = all(c["passed"] for c in checks.values())

        return {
            "validation_period_complete": all_passed,
            "criteria_checks": checks,
            "governance_note": (
                "Sufficient paper trading evidence produced. Ready for Model Auditor in Phase 8."
                if all_passed
                else "Insufficient evidence. Continue paper trading before submission to governance."
            ),
        }
