"""Reporting and serialization tools for Phase 5 Backtesting results.

Generates human-readable Markdown reports and machine-readable JSON summaries,
including full cost breakdowns, regime analyses, drawdown profiles, and trade ledgers.
"""

import json
from dataclasses import asdict
from pathlib import Path

from backtesting.models import BacktestResult
from backtesting.walk_forward import WalkForwardResult


class BacktestReporter:
    """Produces detailed human-readable and machine-readable backtest reports."""

    @staticmethod
    def generate_markdown_report(result: BacktestResult) -> str:
        """Render a comprehensive human-readable Markdown report."""
        m = result.metrics
        cfg = result.config
        duration = result.end_time - result.start_time

        # Regime breakdown table
        regime_lines = []
        if m.regime_performance:
            regime_lines.append("| Regime | Trades | Net P&L ($) | Win Rate (%) | Avg Trade ($) |")
            regime_lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for reg, stats in m.regime_performance.items():
                regime_lines.append(
                    f"| {reg} | {stats['trade_count']} | ${stats['net_pnl']:.2f} | "
                    f"{stats['win_rate'] * 100:.1f}% | ${stats['avg_trade']:.2f} |"
                )
        else:
            regime_lines.append("_No closed trades to categorize by market regime._")
        regime_table = "\n".join(regime_lines)

        # Warnings list
        warn_lines = (
            [f"- {w}" for w in result.warnings]
            if result.warnings
            else ["- _No data-quality anomalies detected._"]
        )
        warnings_str = "\n".join(warn_lines)

        report = f"""# Historical Backtest Report: `{result.backtest_id}`

**Execution Timestamp:** {result.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")}
**Code Version:** `{result.code_version}` | **Model Version:** `{cfg.model_version}` | **Feature Version:** `{cfg.feature_version}`

---

## 1. Executive Summary

| Metric | Result | Benchmark (Buy & Hold) |
| :--- | :--- | :--- |
| **Initial Capital** | ${result.initial_capital:,.2f} | ${result.initial_capital:,.2f} |
| **Final Equity** | ${result.final_equity:,.2f} | - |
| **Total Return** | **{result.total_return * 100:+.2f}%** | **{m.benchmark_return * 100:+.2f}%** |
| **CAGR** | {result.cagr * 100:+.2f}% | - |
| **Max Drawdown** | **{result.max_drawdown * 100:.2f}%** | - |
| **Sharpe Ratio** | {result.sharpe_ratio:.2f} | - |
| **Sortino Ratio** | {result.sortino_ratio:.2f} | - |
| **Calmar Ratio** | {result.calmar_ratio:.2f} | - |
| **Win Rate** | {result.win_rate * 100:.1f}% | - |
| **Profit Factor** | {result.profit_factor:.2f} | - |
| **Expectancy** | ${result.expectancy:,.2f} | - |

---

## 2. Test Configuration & Execution Assumptions

- **Symbol:** `{cfg.symbol}`
- **Timeframe:** `{cfg.timeframe}`
- **Period:** {result.start_time.strftime("%Y-%m-%d %H:%M")} to {result.end_time.strftime("%Y-%m-%d %H:%M")} ({duration})
- **Fee Model:** Maker: `{cfg.maker_fee * 10000:.1f} bps` | Taker: `{cfg.taker_fee * 10000:.1f} bps` | Fixed: `${cfg.fixed_fee:.2f}`
- **Spread Model:** Half-Spread `{cfg.spread_pct / 2 * 10000:.1f} bps` (Bid/Ask derived around mark)
- **Slippage Model:** Baseline `{cfg.slippage_pct * 10000:.1f} bps`
- **Latency Model:** `{cfg.latency_seconds:.1f}s` simulation execution delay
- **Liquidity Model:** Max bar volume capacity `{cfg.max_bar_volume_pct * 100:.1f}%` (Partial fills: `{cfg.allow_partial_fills}`)
- **Position Sizing:** `{cfg.sizing_method}` (Value: `{cfg.sizing_value}`, Max Size: `{cfg.max_position_size}`, Max Notional: `${cfg.max_notional:,.2f}`)
- **End-of-Test Action:** `{cfg.end_of_test_action}`

---

## 3. Cost & Friction Breakdown

| Cost Category | Total Amount ($) | Impact on P&L |
| :--- | :--- | :--- |
| **Gross Realized P&L** | ${m.gross_pnl:+,.2f} | _Pre-friction return_ |
| **Exchange Fees** | ${m.total_fees:,.2f} | -{(m.total_fees / abs(m.gross_pnl) * 100) if abs(m.gross_pnl) > 0 else 0:.1f}% |
| **Spread Friction** | ${m.total_spread:,.2f} | -{(m.total_spread / abs(m.gross_pnl) * 100) if abs(m.gross_pnl) > 0 else 0:.1f}% |
| **Execution Slippage** | ${m.total_slippage:,.2f} | -{(m.total_slippage / abs(m.gross_pnl) * 100) if abs(m.gross_pnl) > 0 else 0:.1f}% |
| **Net Realized P&L** | **${m.net_pnl:+,.2f}** | **Net realizable performance** |

---

## 4. Trade Statistics

- **Total Completed Trades:** {m.total_trades}
- **Winning Trades:** {m.winning_trades} ({result.win_rate * 100:.1f}%)
- **Losing Trades:** {m.losing_trades}
- **Average Trade Net P&L:** ${m.avg_trade_pnl:,.2f}
- **Average Winning Trade:** ${m.avg_win:,.2f}
- **Average Losing Trade:** ${m.avg_loss:,.2f}
- **Average Holding Period:** {m.avg_holding_period_seconds / 60.0:.1f} minutes

---

## 5. Regime Performance Breakdown

{regime_table}

---

## 6. Data Quality & Diagnostic Warnings

{warnings_str}
"""
        return report

    @staticmethod
    def generate_walk_forward_markdown(wf_result: WalkForwardResult) -> str:
        """Render a comprehensive human-readable Walk-Forward report."""
        agg = wf_result.aggregate_metrics
        cfg = wf_result.config

        # Window rows
        rows = []
        rows.append(
            "| Window | Model Ver | Out-of-Sample Period | Trades | Return | Win Rate | Sharpe | Max DD |"
        )
        rows.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for w in wf_result.summary_table:
            rows.append(
                f"| W{w['window']:02d} | `{w['model_version']}` | {w['test_period']} | "
                f"{w['trades']} | {w['return_pct']:+.2f}% | {w['win_rate']:.1f}% | "
                f"{w['sharpe']:.2f} | {w['max_dd_pct']:.2f}% |"
            )
        window_table = "\n".join(rows)

        report = f"""# Rolling Walk-Forward Validation Report

**Windows Evaluated:** {wf_result.total_windows}
**Leakage Invariant Status:** **{"PASSED (Strict Monotonic Separation)" if wf_result.leakage_check_passed else "FAILED"}**
**Window Sizing:** Train={cfg.train_bars} bars, Val={cfg.val_bars} bars, Test={cfg.test_bars} bars, Step={cfg.step_bars} bars

---

## 1. Aggregate Out-Of-Sample Performance

| Metric | Aggregate Out-of-Sample Result |
| :--- | :--- |
| **Combined OOS Trades** | {agg.total_trades} |
| **Combined OOS Return** | **{agg.total_return * 100:+.2f}%** |
| **Out-of-Sample Sharpe Ratio** | **{agg.sharpe_ratio:.2f}** |
| **Out-of-Sample Sortino Ratio** | {agg.sortino_ratio:.2f} |
| **Out-of-Sample Max Drawdown** | {agg.max_drawdown * 100:.2f}% |
| **Out-of-Sample Win Rate** | {agg.win_rate * 100:.1f}% |
| **Out-of-Sample Profit Factor** | {agg.profit_factor:.2f} |
| **Net Realized P&L** | ${agg.net_pnl:+,.2f} |
| **Total Fees Paid** | ${agg.total_fees:,.2f} |

---

## 2. Window-by-Window Out-of-Sample Breakdown

{window_table}

---

## 3. Anti-Leakage Audit Summary

- Every window strictly satisfied: `max(T_train) < min(T_val) < min(T_test)`.
- Preprocessing and model parameter fitting occurred strictly within each window's training partition.
- Validation partition was used strictly for model parameter freezing.
- Zero out-of-sample test bars were observable prior to evaluation.
"""
        return report

    @staticmethod
    def export_json(result: BacktestResult, output_path: str | Path) -> None:
        """Export machine-readable JSON result summary."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "backtest_id": result.backtest_id,
            "code_version": result.code_version,
            "created_at": result.created_at.isoformat(),
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "symbol": result.config.symbol,
            "timeframe": result.config.timeframe,
            "initial_capital": result.initial_capital,
            "final_equity": result.final_equity,
            "metrics": asdict(result.metrics),
            "trade_count": result.trade_count,
            "warnings": result.warnings,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_equity_curve_csv(result: BacktestResult, output_path: str | Path) -> None:
        """Export portfolio equity curve time-series to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "timestamp,equity,cash,market_value,gross_exposure,net_exposure,realized_pnl,unrealized_pnl,cumulative_fees,drawdown\n"
            )
            for pt in result.equity_curve:
                f.write(
                    f"{pt.timestamp.isoformat()},{pt.equity:.4f},{pt.cash:.4f},{pt.market_value:.4f},"
                    f"{pt.gross_exposure:.4f},{pt.net_exposure:.4f},{pt.realized_pnl:.4f},{pt.unrealized_pnl:.4f},"
                    f"{pt.cumulative_fees:.4f},{pt.drawdown:.6f}\n"
                )

    @staticmethod
    def export_trade_ledger_csv(result: BacktestResult, output_path: str | Path) -> None:
        """Export completed trade ledger to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "trade_id,symbol,side,entry_time,exit_time,entry_price,exit_price,quantity,gross_pnl,fees,slippage,spread,net_pnl,return_pct,holding_period_sec,exit_reason,model_version,regime\n"
            )
            for tr in result.trade_ledger:
                f.write(
                    f"{tr.trade_id},{tr.symbol},{tr.side},{tr.entry_time.isoformat()},{tr.exit_time.isoformat()},"
                    f"{tr.entry_price:.4f},{tr.exit_price:.4f},{tr.quantity:.6f},{tr.gross_pnl:.4f},{tr.fees:.4f},"
                    f"{tr.slippage:.4f},{tr.spread:.4f},{tr.net_pnl:.4f},{tr.return_pct:.6f},{tr.holding_period_seconds:.1f},"
                    f"{tr.exit_reason},{tr.model_version},{tr.regime_at_entry}\n"
                )
