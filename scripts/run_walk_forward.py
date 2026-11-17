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
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to real historical Parquet dataset (e.g. data/datasets/market_regimes/...)",
    )
    parser.add_argument("--candles", type=int, default=None, help="Total historical candles")
    parser.add_argument(
        "--synthetic-smoke-test",
        action="store_true",
        help="Use generated candles for a smoke run; results are not performance evidence",
    )
    parser.add_argument("--train-bars", type=int, default=400, help="Training partition bars")
    parser.add_argument("--val-bars", type=int, default=100, help="Validation partition bars")
    parser.add_argument(
        "--test-bars", type=int, default=100, help="Out-of-sample test partition bars"
    )
    parser.add_argument("--step-bars", type=int, default=100, help="Rolling stride step bars")
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Markdown report output path",
    )
    args = parser.parse_args()
    if bool(args.data_file) == args.synthetic_smoke_test:
        parser.error("Provide --data-file for historical evaluation or --synthetic-smoke-test explicitly")
    if args.synthetic_smoke_test:
        args.report_out = args.report_out or "reports/synthetic_walk_forward_report.md"
        print("WARNING: generated candles are for smoke testing, not performance evaluation.")
    else:
        args.report_out = args.report_out or "reports/walk_forward_report.md"

    print("\n=======================================================")
    print("TRADING INTELLIGENCE SYSTEM — WALK-FORWARD VALIDATION")
    print("=======================================================")
    print(f"Symbol:      {args.symbol}")
    print(
        f"Partitions:  Train={args.train_bars}, Val={args.val_bars}, Test={args.test_bars}, Step={args.step_bars}"
    )

    if args.data_file:
        from data.storage.parquet_store import load_candles_from_parquet

        data_path = Path(args.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {data_path}")
        candles = load_candles_from_parquet(data_path)
        if args.candles and len(candles) > args.candles:
            candles = candles[: args.candles]
        print(f"Loaded {len(candles)} real candles from {data_path.resolve()}.\n")
        args.symbol = candles[0].symbol if candles else args.symbol
        timeframe = candles[0].timeframe if candles else "1m"
    else:
        candle_count = args.candles if args.candles is not None else 800
        candles = generate_synthetic_candles(count=candle_count, symbol=args.symbol)
        print(f"Loaded {len(candles)} synthetic candles.\n")
        timeframe = candles[0].timeframe if candles else "1m"

    wf_config = WalkForwardConfig(
        train_bars=args.train_bars,
        val_bars=args.val_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        symbol=args.symbol,
        timeframe=timeframe,
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
