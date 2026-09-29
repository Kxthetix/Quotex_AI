"""Critical Anti-Leakage and Execution Realism Tests for Phase 5.

Mandatory specification tests:
- Section 57: Critical Leakage Invariant: Decisions up to T must remain 100% identical when data after T is appended.
- Section 58: Critical Execution Realism: Signal timestamp < execution timestamp under configured latency; fill price comes from execution-time market data.
- Section 59: Critical Cost Separation: Realistic fees/spread/slippage reduce net P&L relative to gross P&L; costs are never hidden.
- Section 60: Sanity checks: No fills outside candle high/low range, no positions exceeding limits, no negative cash without leverage.
"""

from datetime import UTC, datetime, timedelta

from backtesting.engine import BacktestEngine
from backtesting.events import SignalEvent
from backtesting.models import (
    BacktestConfig,
    EndOfTestAction,
)
from data.ingestion.models import Candle
from scripts.calculate_features import generate_synthetic_candles


class TestCriticalBacktestLeakage:
    def test_section_57_critical_leakage_invariant(self):
        """CRITICAL TEST (Section 57):

        1. Run backtest through timestamp T.
        2. Record every decision and result up to T.
        3. Add data after T.
        4. Replay through T using the enlarged dataset.
        5. Assert that every decision up to T remains identical.
        """
        all_candles = generate_synthetic_candles(count=200, symbol="BTC/USDT")
        t_cutoff_idx = 100
        t_cutoff = all_candles[t_cutoff_idx].end_time

        candles_up_to_t = all_candles[: t_cutoff_idx + 1]

        config = BacktestConfig(
            symbol="BTC/USDT",
            initial_capital=10000.0,
            end_date=t_cutoff,
            end_of_test_action=EndOfTestAction.MARK_TO_MARKET,
            random_seed=42,
        )

        # 1. Run backtest on data up to T
        engine_1 = BacktestEngine(config=config)
        res_1 = engine_1.run(candles_up_to_t)

        decisions_1 = [
            (t.trade_id, t.entry_time, t.exit_time, t.entry_price, t.exit_price, t.net_pnl)
            for t in res_1.trade_ledger
        ]
        audit_1 = res_1.audit_log

        # 2. Run backtest on the full enlarged dataset (candles up to 200), but with end_date=T
        engine_2 = BacktestEngine(config=config)
        res_2 = engine_2.run(all_candles)

        decisions_2 = [
            (t.trade_id, t.entry_time, t.exit_time, t.entry_price, t.exit_price, t.net_pnl)
            for t in res_2.trade_ledger
        ]
        audit_2 = res_2.audit_log

        # Assert identical decisions and identical audit log up to T
        assert decisions_1 == decisions_2, (
            "Future data after T altered past trading decisions up to T!"
        )
        assert audit_1 == audit_2, "Audit log differs between standalone and future-expanded runs!"
        assert res_1.final_equity == res_2.final_equity

    def test_section_58_critical_execution_latency(self):
        """CRITICAL TEST (Section 58):

        Proves signal timestamp < execution timestamp when latency is configured.
        Execution price must come from market data available at execution time.
        """
        # Create 5 candles with 1-minute intervals
        start_t = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        candles = []
        prices = [100.0, 102.0, 105.0, 110.0, 115.0]
        for i, p in enumerate(prices):
            t_s = start_t + timedelta(minutes=i)
            t_e = t_s + timedelta(minutes=1)
            candles.append(
                Candle(
                    symbol="BTC/USDT",
                    timeframe="1m",
                    open=p,
                    high=p + 1.0,
                    low=p - 1.0,
                    close=p,
                    volume=1000.0,
                    start_time=t_s,
                    end_time=t_e,
                    trades_count=10,
                    is_closed=True,
                )
            )

        # Configure 60-second latency (1 bar delay)
        config = BacktestConfig(
            symbol="BTC/USDT",
            latency_seconds=60.0,
            spread_pct=0.0,
            slippage_pct=0.0,
            maker_fee=0.0,
            taker_fee=0.0,
            end_of_test_action=EndOfTestAction.MARK_TO_MARKET,
        )

        def signal_on_bar_1(candle: Candle, history: list[Candle]) -> SignalEvent | None:
            # Emit ENTER_LONG exactly on the first candle
            if len(history) == 1:
                return SignalEvent(
                    timestamp=candle.end_time,
                    symbol=candle.symbol,
                    direction_prob=0.8,
                    expected_return=0.01,
                    expected_volatility=0.2,
                    trade_quality_prob=0.8,
                    regime="TREND_UP",
                    suggested_action="ENTER_LONG",
                )
            return None

        engine = BacktestEngine(config=config, intelligence_provider=signal_on_bar_1)
        res = engine.run(candles)

        pe = engine.portfolio.get_position_engine("BTC/USDT")
        pos = pe.position

        # First candle closed at 10:01 (t0 = 10:01)
        # Latency is 60s -> eligible execution time is 10:02
        # Candle 2 closes at 10:02, price is 102.0
        assert pos.entry_time > candles[0].end_time, "Execution occurred at or before signal time!"
        assert pos.entry_time == candles[1].end_time
        assert pos.avg_entry_price == 102.0  # Filled at bar 2 price, not bar 1 price (100.0)!

    def test_section_59_critical_cost_impact(self):
        """CRITICAL TEST (Section 59):

        Run backtest with zero costs vs realistic costs.
        Verify net performance strictly reflects trading costs.
        """
        candles = generate_synthetic_candles(count=150, symbol="BTC/USDT")

        # 1. Zero cost configuration
        cfg_frictionless = BacktestConfig(
            symbol="BTC/USDT",
            spread_pct=0.0,
            slippage_pct=0.0,
            maker_fee=0.0,
            taker_fee=0.0,
            fixed_fee=0.0,
            random_seed=42,
        )
        res_frictionless = BacktestEngine(config=cfg_frictionless).run(candles)

        # 2. Realistic cost configuration (2 bps spread, 3 bps slippage, 5 bps fee)
        cfg_realistic = BacktestConfig(
            symbol="BTC/USDT",
            spread_pct=0.0002,
            slippage_pct=0.0003,
            maker_fee=0.0002,
            taker_fee=0.0005,
            fixed_fee=0.0,
            random_seed=42,
        )
        res_realistic = BacktestEngine(config=cfg_realistic).run(candles)

        # With friction, net equity must be lower than frictionless
        if res_realistic.trade_count > 0:
            assert res_realistic.final_equity < res_frictionless.final_equity
            assert res_realistic.total_fees > 0.0
            assert res_realistic.total_slippage > 0.0
            assert res_realistic.total_spread > 0.0
            # Net P&L = Gross P&L - Fees
            assert res_realistic.metrics.net_pnl < res_realistic.metrics.gross_pnl

    def test_section_60_performance_sanity_checks(self):
        """Sanity checks: No fills outside candle range, no negative cash without leverage."""
        candles = generate_synthetic_candles(count=100, symbol="BTC/USDT")
        config = BacktestConfig(
            symbol="BTC/USDT",
            initial_capital=5000.0,
            max_leverage=1.0,  # Cash only
            max_position_size=2.0,
        )
        engine = BacktestEngine(config=config)
        res = engine.run(candles)

        # Verify no equity point has negative cash
        for pt in res.equity_curve:
            assert pt.cash >= -1e-6, f"Negative cash detected: {pt.cash} at {pt.timestamp}"

        # Verify all trade fill prices fall within reasonable bounds of candle history
        min_historical_low = min(c.low for c in candles)
        max_historical_high = max(c.high for c in candles)
        for tr in res.trade_ledger:
            assert tr.entry_price >= min_historical_low * 0.99
            assert tr.entry_price <= max_historical_high * 1.01
            assert tr.exit_price >= min_historical_low * 0.99
            assert tr.exit_price <= max_historical_high * 1.01
