"""Walk-Forward Testing Engine for Phase 5.

Implements chronological rolling-window out-of-sample validation.
Enforces strict anti-leakage invariants:
    max(training_time) < min(validation_time) < min(test_time)
Trains preprocessing and models strictly on training partitions, validates on validation partitions,
and evaluates performance strictly out-of-sample on test partitions.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backtesting.engine import BacktestEngine
from backtesting.events import SignalEvent
from backtesting.metrics import (
    MetricsEngine,
    PerformanceMetrics,
    annualization_factor_for_timeframe,
)
from backtesting.models import BacktestConfig, BacktestResult, EquityPoint, TradeRecord
from data.ingestion.models import Candle

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for rolling walk-forward cross-validation."""

    train_bars: int = 500  # Number of bars for training
    val_bars: int = 100  # Number of bars for validation
    test_bars: int = 100  # Number of bars for out-of-sample testing
    step_bars: int = 100  # Stride / step size to roll window forward
    min_train_bars: int = 300
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)


@dataclass
class WindowResult:
    """Result from a single out-of-sample walk-forward window."""

    window_index: int
    model_version: str
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    train_bars: int
    test_bars: int
    metrics: PerformanceMetrics
    test_result: BacktestResult


@dataclass
class WalkForwardResult:
    """Aggregated out-of-sample walk-forward evaluation result."""

    config: WalkForwardConfig
    windows: list[WindowResult]
    total_windows: int
    combined_equity_curve: list[EquityPoint]
    combined_trades: list[TradeRecord]
    aggregate_metrics: PerformanceMetrics
    leakage_check_passed: bool
    summary_table: list[dict[str, Any]]


class WalkForwardEngine:
    """Executes rolling walk-forward model training and out-of-sample backtesting."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config
        self.metrics_engine = MetricsEngine(
            annualization_factor=annualization_factor_for_timeframe(config.timeframe)
        )

    def run(
        self,
        candles: list[Candle],
        model_trainer: Callable[
            [list[Candle], list[Candle], int], Callable[[Candle, list[Candle]], SignalEvent | None]
        ]
        | None = None,
    ) -> WalkForwardResult:
        """Run rolling walk-forward validation across chronological candles."""
        total_bars = len(candles)
        req_window = self.config.train_bars + self.config.val_bars + self.config.test_bars
        if total_bars < req_window:
            raise ValueError(
                f"Insufficient historical bars ({total_bars}) for walk-forward window requirements ({req_window})"
            )

        # Sort candles strictly chronologically
        sorted_candles = sorted(candles, key=lambda c: c.end_time)

        windows: list[WindowResult] = []
        combined_trades: list[TradeRecord] = []
        combined_equity: list[EquityPoint] = []
        running_capital = self.config.backtest_config.initial_capital
        leakage_passed = True

        window_idx = 0
        start_idx = 0

        while (start_idx + req_window) <= total_bars:
            train_start_idx = start_idx
            train_end_idx = train_start_idx + self.config.train_bars
            val_start_idx = train_end_idx
            val_end_idx = val_start_idx + self.config.val_bars
            test_start_idx = val_end_idx
            test_end_idx = test_start_idx + self.config.test_bars

            train_slice = sorted_candles[train_start_idx:train_end_idx]
            val_slice = sorted_candles[val_start_idx:val_end_idx]
            test_slice = sorted_candles[test_start_idx:test_end_idx]

            # Invariant Anti-Leakage Verification (Section 43)
            max_train_ts = max(c.end_time for c in train_slice)
            min_val_ts = min(c.end_time for c in val_slice)
            max_val_ts = max(c.end_time for c in val_slice)
            min_test_ts = min(c.end_time for c in test_slice)

            if not (max_train_ts < min_val_ts and max_val_ts < min_test_ts):
                leakage_passed = False
                raise AssertionError(
                    f"Leakage invariant violated in Window {window_idx}: "
                    f"max(train)={max_train_ts} >= min(val)={min_val_ts} or max(val)={max_val_ts} >= min(test)={min_test_ts}"
                )

            # Model training & versioning strictly on training & validation partitions
            model_ver = f"wf_w{window_idx:02d}_v1"
            window_cfg = BacktestConfig(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                initial_capital=running_capital,
                start_date=min_test_ts,
                end_date=max(c.end_time for c in test_slice),
                maker_fee=self.config.backtest_config.maker_fee,
                taker_fee=self.config.backtest_config.taker_fee,
                spread_pct=self.config.backtest_config.spread_pct,
                slippage_pct=self.config.backtest_config.slippage_pct,
                latency_seconds=self.config.backtest_config.latency_seconds,
                model_version=model_ver,
                feature_version=self.config.backtest_config.feature_version,
            )

            # Train or fit intelligence model
            if model_trainer:
                intelligence_provider = model_trainer(train_slice, val_slice, window_idx)
            else:
                intelligence_provider = self._fit_baseline_window_model(train_slice, model_ver)

            # Execute backtest on the out-of-sample test partition
            engine = BacktestEngine(config=window_cfg, intelligence_provider=intelligence_provider)
            test_result = engine.run(test_slice)

            # Record window result
            win_res = WindowResult(
                window_index=window_idx,
                model_version=model_ver,
                train_start=min(c.end_time for c in train_slice),
                train_end=max_train_ts,
                val_start=min_val_ts,
                val_end=max_val_ts,
                test_start=min_test_ts,
                test_end=max(c.end_time for c in test_slice),
                train_bars=len(train_slice),
                test_bars=len(test_slice),
                metrics=test_result.metrics,
                test_result=test_result,
            )
            windows.append(win_res)

            # Accumulate out-of-sample trades and equity
            combined_trades.extend(test_result.trade_ledger)
            combined_equity.extend(test_result.equity_curve)

            # Update running capital for next out-of-sample window
            running_capital = test_result.final_equity

            # Advance rolling window by step_bars
            start_idx += self.config.step_bars
            window_idx += 1

        # Calculate aggregate out-of-sample performance metrics
        initial_cap = self.config.backtest_config.initial_capital
        agg_metrics = self.metrics_engine.calculate_metrics(
            equity_curve=combined_equity,
            trade_ledger=combined_trades,
            initial_capital=initial_cap,
            benchmark_initial_price=sorted_candles[0].open,
            benchmark_final_price=sorted_candles[-1].close,
        )

        summary_rows = [
            {
                "window": w.window_index,
                "model_version": w.model_version,
                "train_period": f"{w.train_start.strftime('%Y-%m-%d %H:%M')} to {w.train_end.strftime('%Y-%m-%d %H:%M')}",
                "test_period": f"{w.test_start.strftime('%Y-%m-%d %H:%M')} to {w.test_end.strftime('%Y-%m-%d %H:%M')}",
                "trades": w.metrics.total_trades,
                "return_pct": round(w.metrics.total_return * 100, 2),
                "win_rate": round(w.metrics.win_rate * 100, 1),
                "sharpe": round(w.metrics.sharpe_ratio, 2),
                "max_dd_pct": round(w.metrics.max_drawdown * 100, 2),
            }
            for w in windows
        ]

        return WalkForwardResult(
            config=self.config,
            windows=windows,
            total_windows=len(windows),
            combined_equity_curve=combined_equity,
            combined_trades=combined_trades,
            aggregate_metrics=agg_metrics,
            leakage_check_passed=leakage_passed,
            summary_table=summary_rows,
        )

    def _fit_baseline_window_model(
        self, train_candles: list[Candle], model_version: str
    ) -> Callable[[Candle, list[Candle]], SignalEvent | None]:
        """Fit a simple threshold model strictly on training candles."""
        closes = [c.close for c in train_candles]
        rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        avg_ret = sum(rets) / len(rets) if rets else 0.0

        # Create closure that uses only parameters derived from training partition
        def model_inference(candle: Candle, history: list[Candle]) -> SignalEvent | None:
            if len(history) < 15:
                return None
            sma_fast = sum(c.close for c in history[-5:]) / 5.0
            sma_slow = sum(c.close for c in history[-15:]) / 15.0

            action = "HOLD"
            direction_prob = 0.5
            if sma_fast > sma_slow * (1.0 + max(0.0005, abs(avg_ret))):
                action = "ENTER_LONG"
                direction_prob = 0.65
            elif sma_fast < sma_slow * (1.0 - max(0.0005, abs(avg_ret))):
                action = "EXIT"
                direction_prob = 0.35

            return SignalEvent(
                timestamp=candle.end_time,
                symbol=candle.symbol,
                direction_prob=direction_prob,
                expected_return=0.005 if action == "ENTER_LONG" else 0.0,
                expected_volatility=0.25,
                trade_quality_prob=0.60,
                regime="TREND_UP" if sma_fast > sma_slow else "TREND_DOWN",
                suggested_action=action,
                strength=0.75,
                model_version=model_version,
            )

        return model_inference
