"""Command line script to execute a reproducible historical backtest (Phase 5).

Usage:
    python scripts/run_backtest.py --symbol BTC/USDT --candles 600 --initial-capital 10000 --latency-sec 1.0
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig, SizingMethod
from backtesting.reporting import BacktestReporter
from scripts.calculate_features import generate_synthetic_candles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run realistic event-driven backtest simulation.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1m", help="Candle timeframe")
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to real historical Parquet dataset (e.g. data/datasets/market_regimes/...)",
    )
    parser.add_argument(
        "--synthetic-smoke-test",
        action="store_true",
        help="Use generated candles for a smoke run; results are not performance evidence",
    )
    parser.add_argument("--candles", type=int, default=None, help="Number of historical candles")
    parser.add_argument(
        "--initial-capital", type=float, default=10000.0, help="Initial portfolio capital"
    )
    parser.add_argument(
        "--maker-fee", type=float, default=0.0002, help="Maker fee rate (e.g. 0.0002 for 2 bps)"
    )
    parser.add_argument(
        "--taker-fee", type=float, default=0.0005, help="Taker fee rate (e.g. 0.0005 for 5 bps)"
    )
    parser.add_argument(
        "--spread-bps", type=float, default=2.0, help="Bid/Ask spread in basis points"
    )
    parser.add_argument(
        "--slippage-bps", type=float, default=3.0, help="Execution slippage in basis points"
    )
    parser.add_argument(
        "--latency-sec", type=float, default=0.0, help="Execution latency delay in seconds"
    )
    parser.add_argument(
        "--sizing-method", type=str, default="PERCENT_CAPITAL", help="Position sizing method"
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Markdown report output path",
    )
    parser.add_argument(
        "--equity-out",
        type=str,
        default=None,
        help="Equity curve CSV output path",
    )
    parser.add_argument(
        "--trades-out",
        type=str,
        default=None,
        help="Trade ledger CSV output path",
    )
    args = parser.parse_args()
    if bool(args.data_file) == args.synthetic_smoke_test:
        parser.error("Provide --data-file for historical evaluation or --synthetic-smoke-test explicitly")

    if args.synthetic_smoke_test:
        args.report_out = args.report_out or "reports/synthetic_backtest_report.md"
        args.equity_out = args.equity_out or "reports/synthetic_equity_curve.csv"
        args.trades_out = args.trades_out or "reports/synthetic_trade_ledger.csv"
        print("WARNING: generated candles are for smoke testing, not performance evaluation.")
    else:
        args.report_out = args.report_out or "reports/backtest_report.md"
        args.equity_out = args.equity_out or "reports/equity_curve.csv"
        args.trades_out = args.trades_out or "reports/trade_ledger.csv"

    print("\n=======================================================")
    print("TRADING INTELLIGENCE SYSTEM — PHASE 5 BACKTEST")
    print("=======================================================")
    print(f"Symbol:          {args.symbol}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(
        f"Friction:        Spread={args.spread_bps}bps, Slippage={args.slippage_bps}bps, Fees={args.taker_fee * 10000:.1f}bps"
    )
    print(f"Latency:         {args.latency_sec:.1f}s")
    print(f"Sizing:          {args.sizing_method}")

    # Generate or load candles
    if args.data_file:
        from data.storage.parquet_store import load_candles_from_parquet

        data_path = Path(args.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {data_path}")
        candles = load_candles_from_parquet(data_path)
        if args.candles and len(candles) > args.candles:
            candles = candles[: args.candles]
        print(f"Loaded {len(candles)} real historical candles from {data_path.resolve()}.")
        if candles:
            args.symbol = candles[0].symbol
            args.timeframe = candles[0].timeframe
    else:
        candle_count = args.candles if args.candles is not None else 500
        candles = generate_synthetic_candles(count=candle_count, symbol=args.symbol)
        print(f"Loaded {len(candles)} synthetic candles.")

    # Configure backtest
    sizing_enum = getattr(SizingMethod, args.sizing_method, SizingMethod.PERCENT_CAPITAL)
    config = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_capital=args.initial_capital,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        spread_pct=args.spread_bps / 10000.0,
        slippage_pct=args.slippage_bps / 10000.0,
        latency_seconds=args.latency_sec,
        sizing_method=sizing_enum,
        sizing_value=0.10,
    )

    engine = BacktestEngine(config=config)
    result = engine.run(candles)

    # Print summary metrics to console
    m = result.metrics
    print("\n----------------- BACKTEST RESULTS -------------------")
    print(f"Backtest ID:       {result.backtest_id}")
    print(f"Final Equity:      ${result.final_equity:,.2f} ({result.total_return * 100:+.2f}%)")
    print(f"Benchmark Return:  {m.benchmark_return * 100:+.2f}%")
    print(f"Max Drawdown:      {result.max_drawdown * 100:.2f}%")
    print(f"Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:     {result.sortino_ratio:.2f}")
    print(f"Calmar Ratio:      {result.calmar_ratio:.2f}")
    print(
        f"Win Rate:          {result.win_rate * 100:.1f}% ({m.winning_trades}/{m.total_trades} trades)"
    )
    print(f"Profit Factor:     {result.profit_factor:.2f}")
    print(f"Expectancy:        ${result.expectancy:,.2f}")
    print(f"Gross P&L:         ${m.gross_pnl:+,.2f}")
    print(f"Exchange Fees:     ${m.total_fees:,.2f}")
    print(f"Spread Cost:       ${m.total_spread:,.2f}")
    print(f"Slippage Cost:     ${m.total_slippage:,.2f}")
    print(f"Net P&L:           ${m.net_pnl:+,.2f}")
    print("------------------------------------------------------")

    # Export reports and ledgers
    rep_path = Path(args.report_out)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = BacktestReporter.generate_markdown_report(result)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Report exported:       {rep_path.resolve()}")

    if args.equity_out:
        BacktestReporter.export_equity_curve_csv(result, args.equity_out)
        print(f"Equity curve exported: {Path(args.equity_out).resolve()}")

    if args.trades_out:
        BacktestReporter.export_trade_ledger_csv(result, args.trades_out)
        print(f"Trade ledger exported: {Path(args.trades_out).resolve()}")


if __name__ == "__main__":
    main()
