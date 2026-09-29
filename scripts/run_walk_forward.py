"""Command line script to execute rolling walk-forward validation (Phase 5).

Usage:
    python scripts/run_walk_forward.py --symbol BTC/USDT --candles 800 --train-bars 400 --test-bars 100
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.models import BacktestConfig
from backtesting.reporting import BacktestReporter
from backtesting.walk_forward import WalkForwardConfig, WalkForwardEngine
from scripts.calculate_features import generate_synthetic_candles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling walk-forward cross-validation.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair symbol")
    parser.add_argument("--candles", type=int, default=800, help="Total historical candles")
    parser.add_argument("--train-bars", type=int, default=400, help="Training partition bars")
    parser.add_argument("--val-bars", type=int, default=100, help="Validation partition bars")
    parser.add_argument(
        "--test-bars", type=int, default=100, help="Out-of-sample test partition bars"
    )
    parser.add_argument("--step-bars", type=int, default=100, help="Rolling stride step bars")
    parser.add_argument(
        "--report-out",
        type=str,
        default="reports/walk_forward_report.md",
        help="Markdown report output path",
    )
    args = parser.parse_args()

    print("\n=======================================================")
    print("TRADING INTELLIGENCE SYSTEM — WALK-FORWARD VALIDATION")
    print("=======================================================")
    print(f"Symbol:      {args.symbol}")
    print(
        f"Partitions:  Train={args.train_bars}, Val={args.val_bars}, Test={args.test_bars}, Step={args.step_bars}"
    )
    print(f"Total Bars:  {args.candles}")

    candles = generate_synthetic_candles(count=args.candles, symbol=args.symbol)
    print(f"Loaded {len(candles)} historical candles.\n")

    wf_config = WalkForwardConfig(
        train_bars=args.train_bars,
        val_bars=args.val_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        symbol=args.symbol,
        backtest_config=BacktestConfig(
            symbol=args.symbol,
            initial_capital=10000.0,
            maker_fee=0.0002,
            taker_fee=0.0005,
            spread_pct=0.0002,
            slippage_pct=0.0003,
        ),
    )

    wf_engine = WalkForwardEngine(config=wf_config)
    wf_result = wf_engine.run(candles)

    print("----------------- WALK-FORWARD WINDOWS ----------------")
    for w in wf_result.summary_table:
        print(
            f"Window {w['window']:02d} [{w['model_version']}]: "
            f"Trades={w['trades']:2d} | Return={w['return_pct']:+6.2f}% | "
            f"WinRate={w['win_rate']:4.1f}% | Sharpe={w['sharpe']:5.2f} | MaxDD={w['max_dd_pct']:5.2f}%"
        )
    print("-------------------------------------------------------")

    agg = wf_result.aggregate_metrics
    print("\n----------- AGGREGATE OUT-OF-SAMPLE METRICS -----------")
    print(f"Total Windows:      {wf_result.total_windows}")
    print(
        f"Leakage Check:      {'PASSED (Monotonic)' if wf_result.leakage_check_passed else 'FAILED'}"
    )
    print(f"Combined Trades:    {agg.total_trades}")
    print(f"Combined Return:    {agg.total_return * 100:+.2f}%")
    print(f"OOS Sharpe Ratio:   {agg.sharpe_ratio:.2f}")
    print(f"OOS Sortino Ratio:  {agg.sortino_ratio:.2f}")
    print(f"OOS Max Drawdown:   {agg.max_drawdown * 100:.2f}%")
    print(f"OOS Win Rate:       {agg.win_rate * 100:.1f}%")
    print(f"Net Realized P&L:   ${agg.net_pnl:+,.2f}")
    print(f"Total Fees Paid:    ${agg.total_fees:,.2f}")
    print("-------------------------------------------------------")

    rep_path = Path(args.report_out)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = BacktestReporter.generate_walk_forward_markdown(wf_result)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nWalk-forward report written to: {rep_path.resolve()}\n")


if __name__ == "__main__":
    main()
