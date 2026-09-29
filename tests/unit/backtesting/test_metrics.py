"""Unit tests for Performance Metrics and Risk Engine (Phase 5).

Covers:
- Total Return, CAGR, and Drawdown calculations
- Sharpe, Sortino, and Calmar ratios with explicit assumptions
- Win rate, Profit factor, and Expectancy formulations
- Cost breakdown (Gross vs Fees vs Slippage vs Spread vs Net)
- Benchmark comparison and Market Regime performance breakdowns
"""

from datetime import UTC, datetime, timedelta

from analysis.regime.models import MarketRegime
from backtesting.metrics import MetricsEngine
from backtesting.models import EquityPoint, ExitReason, PositionSide, TradeRecord


class TestMetricsEngine:
    def test_basic_metrics_calculation(self):
        engine = MetricsEngine(risk_free_rate=0.0, annualization_factor=525600.0)
        t0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        equity_curve = [
            EquityPoint(
                timestamp=t0,
                equity=10000.0,
                cash=10000.0,
                market_value=0.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                cumulative_fees=0.0,
            ),
            EquityPoint(
                timestamp=t0 + timedelta(minutes=10),
                equity=10500.0,
                cash=10500.0,
                market_value=0.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=500.0,
                unrealized_pnl=0.0,
                cumulative_fees=10.0,
            ),
            EquityPoint(
                timestamp=t0 + timedelta(minutes=20),
                equity=10200.0,
                cash=10200.0,
                market_value=0.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=200.0,
                unrealized_pnl=0.0,
                cumulative_fees=15.0,
            ),
            EquityPoint(
                timestamp=t0 + timedelta(minutes=30),
                equity=11000.0,
                cash=11000.0,
                market_value=0.0,
                gross_exposure=0.0,
                net_exposure=0.0,
                realized_pnl=1000.0,
                unrealized_pnl=0.0,
                cumulative_fees=20.0,
            ),
        ]

        trade_ledger = [
            TradeRecord(
                trade_id="t1",
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_time=t0,
                exit_time=t0 + timedelta(minutes=10),
                entry_price=100.0,
                exit_price=105.0,
                quantity=100.0,
                gross_pnl=500.0,
                fees=10.0,
                slippage=2.0,
                spread=1.0,
                net_pnl=490.0,
                return_pct=0.049,
                holding_period_seconds=600.0,
                exit_reason=ExitReason.SIGNAL,
                model_version="v1",
                feature_version="v1",
                regime_at_entry=MarketRegime.TREND_UP,
            ),
            TradeRecord(
                trade_id="t2",
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_time=t0 + timedelta(minutes=10),
                exit_time=t0 + timedelta(minutes=20),
                entry_price=105.0,
                exit_price=102.0,
                quantity=100.0,
                gross_pnl=-300.0,
                fees=10.0,
                slippage=2.0,
                spread=1.0,
                net_pnl=-310.0,
                return_pct=-0.0295,
                holding_period_seconds=600.0,
                exit_reason=ExitReason.SIGNAL,
                model_version="v1",
                feature_version="v1",
                regime_at_entry=MarketRegime.SIDEWAYS,
            ),
        ]

        metrics = engine.calculate_metrics(
            equity_curve=equity_curve,
            trade_ledger=trade_ledger,
            initial_capital=10000.0,
            benchmark_initial_price=100.0,
            benchmark_final_price=110.0,
        )

        assert metrics.total_return == 0.10  # (11000 - 10000) / 10000 = 10%
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.50

        # Drawdown: peak was 10500, dropped to 10200 -> (10500 - 10200) / 10500 = 2.857%
        assert round(metrics.max_drawdown, 4) == round(300.0 / 10500.0, 4)

        # Cost Analysis
        assert metrics.gross_pnl == 200.0  # 500 - 300
        assert metrics.total_fees == 20.0
        assert metrics.net_pnl == 180.0  # 490 - 310
        assert metrics.total_slippage == 4.0
        assert metrics.total_spread == 2.0

        # Expectancy: (0.5 * 490) - (0.5 * 310) = 245 - 155 = 90.0
        assert round(metrics.expectancy, 2) == 90.0

        # Profit Factor: 490 / 310 = 1.5806
        assert round(metrics.profit_factor, 2) == 1.58

        # Benchmark: (110 - 100) / 100 = 10%
        assert metrics.benchmark_return == 0.10

        # Regime breakdown
        assert "TREND_UP" in metrics.regime_performance
        assert "SIDEWAYS" in metrics.regime_performance
        assert metrics.regime_performance["TREND_UP"]["trade_count"] == 1
        assert metrics.regime_performance["TREND_UP"]["net_pnl"] == 490.0
        assert metrics.regime_performance["SIDEWAYS"]["trade_count"] == 1
        assert metrics.regime_performance["SIDEWAYS"]["net_pnl"] == -310.0

    def test_zero_division_safety(self):
        engine = MetricsEngine()
        # Test empty equity curve
        empty_m = engine.calculate_metrics([], [], initial_capital=10000.0)
        assert empty_m.total_return == 0.0
        assert empty_m.win_rate == 0.0
        assert empty_m.profit_factor == 0.0
        assert empty_m.sharpe_ratio == 0.0
        assert empty_m.sortino_ratio == 0.0
        assert empty_m.calmar_ratio == 0.0
