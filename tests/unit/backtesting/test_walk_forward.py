"""Unit and integration tests for Rolling Walk-Forward Validation (Phase 5).

Covers:
- Window generation with strict chronological partitioning
- Section 43 Critical Test: max(train) < min(val) < min(test) anti-leakage verification
- Out-of-sample trade aggregation and equity curve stitching
- Model version uniqueness per rolling window
- Markdown and machine-readable report generation
"""

import pytest

from backtesting.models import BacktestConfig
from backtesting.reporting import BacktestReporter
from backtesting.walk_forward import WalkForwardConfig, WalkForwardEngine
from scripts.calculate_features import generate_synthetic_candles


class TestWalkForwardEngine:
    def test_section_43_walk_forward_leakage_and_rolling_windows(self):
        """CRITICAL TEST (Section 43):

        Verify:
            max(training_time) < min(validation_time) < min(test_time)
        for every walk-forward window.
        """
        candles = generate_synthetic_candles(count=400, symbol="BTC/USDT")

        wf_config = WalkForwardConfig(
            train_bars=150,
            val_bars=50,
            test_bars=50,
            step_bars=50,
            symbol="BTC/USDT",
            backtest_config=BacktestConfig(
                symbol="BTC/USDT",
                initial_capital=10000.0,
            ),
        )

        engine = WalkForwardEngine(config=wf_config)
        result = engine.run(candles)

        assert result.leakage_check_passed is True
        assert result.total_windows >= 3

        # Explicitly verify the temporal inequality for each produced window
        for win in result.windows:
            assert win.train_end < win.val_start, (
                f"Window {win.window_index} train overlaps validation!"
            )
            assert win.val_end < win.test_start, (
                f"Window {win.window_index} validation overlaps test!"
            )
            # Verify distinct model versions
            assert win.model_version.startswith(f"wf_w{win.window_index:02d}")
            # Verify test results are present
            assert win.test_result is not None
            assert win.metrics is not None

        # Verify aggregate metrics
        assert result.aggregate_metrics is not None
        assert isinstance(result.aggregate_metrics.total_return, float)
        assert len(result.summary_table) == result.total_windows

        # Verify report generation works cleanly
        md_report = BacktestReporter.generate_walk_forward_markdown(result)
        assert "# Rolling Walk-Forward Validation Report" in md_report
        assert "PASSED (Strict Monotonic Separation)" in md_report
        assert f"Windows Evaluated:** {result.total_windows}" in md_report

    def test_walk_forward_insufficient_bars_error(self):
        candles = generate_synthetic_candles(count=100, symbol="BTC/USDT")
        wf_config = WalkForwardConfig(
            train_bars=200,
            val_bars=50,
            test_bars=50,
        )
        engine = WalkForwardEngine(config=wf_config)
        with pytest.raises(ValueError, match="Insufficient historical bars"):
            engine.run(candles)
