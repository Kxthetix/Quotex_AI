"""Performance and risk metrics calculation for Phase 5 Backtesting Engine.

Computes Total Return, CAGR, Max Drawdown, Sharpe, Sortino, Calmar, Win Rate,
Profit Factor, Expectancy, Cost Analysis, Benchmark Comparison, and Regime Breakdowns.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from backtesting.models import EquityPoint, TradeRecord


@dataclass
class PerformanceMetrics:
    """Comprehensive portfolio performance and risk metrics."""

    total_return: float
    cagr: float
    max_drawdown: float
    max_drawdown_duration_seconds: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    expectancy: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    avg_holding_period_seconds: float
    gross_pnl: float
    net_pnl: float
    total_fees: float
    total_slippage: float
    total_spread: float
    benchmark_return: float
    regime_performance: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to clean rounded dictionary."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 6)
        return d


class MetricsEngine:
    """Calculates standardized quantitative performance metrics from equity curves and trade ledgers."""

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 525600.0,  # 1-minute bars per year (365 days * 1440 min)
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor


    def calculate_metrics(
        self,
        equity_curve: list[EquityPoint],
        trade_ledger: list[TradeRecord],
        initial_capital: float,
        benchmark_initial_price: float = 0.0,
        benchmark_final_price: float = 0.0,
    ) -> PerformanceMetrics:
        """Compute complete performance, risk, and cost metrics."""
        if not equity_curve:
            return self._empty_metrics()

        equities = np.array([pt.equity for pt in equity_curve])
        final_equity = equities[-1]
        total_return = (
            (final_equity - initial_capital) / initial_capital if initial_capital > 0 else 0.0
        )

        # Time duration for CAGR
        t_start = equity_curve[0].timestamp
        t_end = equity_curve[-1].timestamp
        duration_seconds = max(1.0, (t_end - t_start).total_seconds())
        duration_years = duration_seconds / (365.25 * 86400.0)

        cagr = (
            (final_equity / initial_capital) ** (1.0 / duration_years) - 1.0
            if (duration_years > 0.05 and initial_capital > 0 and final_equity > 0)
            else total_return
        )

        # Drawdown analysis
        peaks = np.maximum.accumulate(equities)
        drawdowns = (peaks - equities) / peaks
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Maximum drawdown duration
        max_dd_duration = 0.0
        current_dd_start = None
        for pt in equity_curve:
            if pt.drawdown > 0:
                if current_dd_start is None:
                    current_dd_start = pt.timestamp
                dd_dur = (pt.timestamp - current_dd_start).total_seconds()
                if dd_dur > max_dd_duration:
                    max_dd_duration = dd_dur
            else:
                current_dd_start = None

        # Period returns for Sharpe and Sortino
        if len(equities) > 1:
            rets = np.diff(equities) / equities[:-1]
            mean_ret = float(np.mean(rets))
            std_ret = float(np.std(rets))

            # Annualized Sharpe
            excess_mean = mean_ret - (self.risk_free_rate / self.annualization_factor)
            sharpe_ratio = (
                (excess_mean / std_ret) * np.sqrt(self.annualization_factor)
                if std_ret > 1e-9
                else 0.0
            )

            # Annualized Sortino (downside deviation)
            negative_rets = rets[rets < 0]
            downside_std = (
                float(np.sqrt(np.mean(negative_rets**2))) if len(negative_rets) > 0 else 0.0
            )
            sortino_ratio = (
                (excess_mean / downside_std) * np.sqrt(self.annualization_factor)
                if downside_std > 1e-9
                else 0.0
            )
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        # Calmar Ratio
        calmar_ratio = (
            (cagr / max_drawdown) if max_drawdown > 1e-6 else (0.0 if cagr <= 0 else 100.0)
        )

        # Trade Ledger Analysis
        total_trades = len(trade_ledger)
        winning_trades = [t for t in trade_ledger if t.net_pnl > 0]
        losing_trades = [t for t in trade_ledger if t.net_pnl <= 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades) if total_trades > 0 else 0.0

        gross_profit = sum(t.net_pnl for t in winning_trades)
        gross_loss = abs(sum(t.net_pnl for t in losing_trades))

        profit_factor = (
            (gross_profit / gross_loss)
            if gross_loss > 1e-6
            else (100.0 if gross_profit > 0 else 0.0)
        )

        avg_win = (gross_profit / win_count) if win_count > 0 else 0.0
        avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0.0
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

        avg_trade_pnl = (
            (sum(t.net_pnl for t in trade_ledger) / total_trades) if total_trades > 0 else 0.0
        )
        avg_holding_sec = (
            (sum(t.holding_period_seconds for t in trade_ledger) / total_trades)
            if total_trades > 0
            else 0.0
        )

        # Cost Breakdown
        gross_pnl = sum(t.gross_pnl for t in trade_ledger)
        total_fees = sum(t.fees for t in trade_ledger)
        total_slippage = sum(t.slippage for t in trade_ledger)
        total_spread = sum(t.spread for t in trade_ledger)
        net_pnl = sum(t.net_pnl for t in trade_ledger)

        # Benchmark return
        benchmark_return = (
            (benchmark_final_price - benchmark_initial_price) / benchmark_initial_price
            if benchmark_initial_price > 0
            else 0.0
        )

        # Regime Performance Breakdown
        regime_perf: dict[str, dict[str, float]] = {}
        regimes_present = {t.regime_at_entry.value for t in trade_ledger}
        for reg in regimes_present:
            reg_trades = [t for t in trade_ledger if t.regime_at_entry.value == reg]
            reg_wins = [t for t in reg_trades if t.net_pnl > 0]
            reg_pnl = sum(t.net_pnl for t in reg_trades)
            reg_win_rate = (len(reg_wins) / len(reg_trades)) if reg_trades else 0.0
            regime_perf[reg] = {
                "trade_count": len(reg_trades),
                "net_pnl": round(float(reg_pnl), 4),
                "win_rate": round(float(reg_win_rate), 4),
                "avg_trade": round(float(reg_pnl / len(reg_trades)), 4) if reg_trades else 0.0,
            }

        return PerformanceMetrics(
            total_return=float(total_return),
            cagr=float(cagr),
            max_drawdown=float(max_drawdown),
            max_drawdown_duration_seconds=float(max_dd_duration),
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            calmar_ratio=float(calmar_ratio),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            expectancy=float(expectancy),
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            avg_trade_pnl=float(avg_trade_pnl),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            avg_holding_period_seconds=float(avg_holding_sec),
            gross_pnl=float(gross_pnl),
            net_pnl=float(net_pnl),
            total_fees=float(total_fees),
            total_slippage=float(total_slippage),
            total_spread=float(total_spread),
            benchmark_return=float(benchmark_return),
            regime_performance=regime_perf,
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            max_drawdown=0.0,
            max_drawdown_duration_seconds=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_trade_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_holding_period_seconds=0.0,
            gross_pnl=0.0,
            net_pnl=0.0,
            total_fees=0.0,
            total_slippage=0.0,
            total_spread=0.0,
            benchmark_return=0.0,
            regime_performance={},
        )


def annualization_factor_for_timeframe(timeframe: str) -> float:
    """Return bars per 365.25-day year for timeframes such as ``5m`` or ``1h``."""
    if len(timeframe) < 2 or not timeframe[:-1].isdigit():
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")

    interval = int(timeframe[:-1])
    unit_seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(timeframe[-1].lower())
    if interval <= 0 or unit_seconds is None:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")

    seconds_per_year = 365.25 * 24 * 60 * 60
    return seconds_per_year / (interval * unit_seconds)
