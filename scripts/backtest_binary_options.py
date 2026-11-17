"""Run a payout-aware, out-of-sample paper backtest on real candle data.

Example:
    python scripts/backtest_binary_options.py --data-file path/to/otc_candles.parquet \
        --payout 0.93 --expiry-bars 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.binary_options import run_binary_options_backtest
from data.storage.parquet_store import load_candles_from_parquet
from signals.binary_scalping_engine import BinaryScalpingEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the candle strategy on historical data; this does not place trades."
    )
    parser.add_argument("--data-file", required=True, help="Parquet file of completed OHLCV candles")
    parser.add_argument(
        "--payout",
        type=float,
        required=True,
        help="Net win payout as a decimal (93%% payout = 0.93)",
    )
    parser.add_argument("--expiry-bars", type=int, default=1, help="Bars from signal close to expiry")
    parser.add_argument("--stake", type=float, default=10.0, help="Fixed simulated stake per trade")
    parser.add_argument(
        "--max-stake-fraction",
        type=float,
        default=0.01,
        help="Maximum stake as a fraction of starting balance (default 1%%)",
    )
    parser.add_argument(
        "--starting-balance", type=float, default=1_000.0, help="Simulated starting balance"
    )
    parser.add_argument(
        "--oos-fraction",
        type=float,
        default=0.30,
        help="Final chronological fraction used for out-of-sample evaluation",
    )
    args = parser.parse_args()

    if not 0.1 <= args.oos_fraction < 0.8:
        parser.error("--oos-fraction must be at least 0.10 and less than 0.80")

    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    candles = load_candles_from_parquet(data_path)
    if len(candles) < 60:
        raise ValueError("At least 60 candles are required for warm-up and out-of-sample evaluation.")

    split_index = max(50, int(len(candles) * (1.0 - args.oos_fraction)))
    engine = BinaryScalpingEngine()

    # Evaluate In-Sample (IS) if sufficient bars exist
    is_candles = candles[:split_index]
    is_result = None
    if len(is_candles) > 60:
        is_result = run_binary_options_backtest(
            is_candles,
            engine,
            payout_rate=args.payout,
            expiry_bars=args.expiry_bars,
            stake=args.stake,
            starting_balance=args.starting_balance,
            max_stake_fraction=args.max_stake_fraction,
            first_signal_index=50,
        )

    # Evaluate Out-of-Sample (OOS)
    oos_result = run_binary_options_backtest(
        candles,
        engine,
        payout_rate=args.payout,
        expiry_bars=args.expiry_bars,
        stake=args.stake,
        starting_balance=args.starting_balance,
        max_stake_fraction=args.max_stake_fraction,
        first_signal_index=split_index,
    )

    symbol = candles[0].symbol
    timeframe = candles[0].timeframe
    is_synthetic = "SYNTHETIC" in symbol.upper() or "synthetic" in str(data_path).lower()

    print("=" * 75)
    print("  AI-TRADER: PAYOUT-AWARE BINARY OPTIONS RESEARCH BACKTEST")
    print("=" * 75)

    if is_synthetic:
        print("\n" + "!" * 75)
        print("  [SMOKE-TEST ONLY - SYNTHETIC DATA DETECTED]")
        print("  This run uses synthetically generated demonstration candles.")
        print("  Synthetic results MUST NOT be used for strategy validation or performance claims.")
        print("!" * 75 + "\n")

    print(f"Asset / Timeframe       : {symbol} / {timeframe}")
    print(f"Contract Expiry         : {args.expiry_bars} bar(s) ({args.expiry_bars} x {timeframe})")
    print(f"Net Payout Rate         : {args.payout:.1%} (Win multiplier: +{args.payout:.2f}x stake)")
    print(f"Break-Even Win Rate     : {oos_result.break_even_win_rate:.2%}  [Formula: 1 / (1 + net_payout_rate)]")
    print(f"Simulated Capital & Cap : {args.starting_balance:.2f} start | Stake: {args.stake:.2f} (Fixed, capped at {args.max_stake_fraction:.1%})")
    print("Sizing Methodology      : Strict Fixed-Stake (Martingale & progressive risk prohibited)")

    # Data matching verification & limitation warning
    is_matching_data = (timeframe == "1m" and not is_synthetic and "OTC" in symbol.upper())
    if not is_matching_data:
        print("\n" + "-" * 75)
        print("  [RESEARCH LIMITATION NOTICE]")
        print(f"  Target strategy horizon is 1-minute binary options, but evaluated dataset")
        print(f"  is {symbol} ({timeframe}). Performance on proxy assets cannot validate or claim")
        print(f"  profitability for binary OTC broker instruments due to differing feeds & spreads.")
        print("-" * 75)

    print("\n" + "=" * 75)
    print(f"  {'METRIC':<28} | {'IN-SAMPLE (IS)':<18} | {'OUT-OF-SAMPLE (OOS)':<18}")
    print("-" * 75)
    print(f"  {'Data Partition':<28} | {f'First {1.0 - args.oos_fraction:.0%} ({len(is_candles)} bars)':<18} | {f'Final {args.oos_fraction:.0%} ({len(candles) - split_index} bars)':<18}")
    print(f"  {'Total Trades Settled':<28} | {f'{is_result.total_trades if is_result else 0}':<18} | {f'{oos_result.total_trades}':<18}")
    print(f"  {'Wins / Losses':<28} | {f'{is_result.wins}W / {is_result.losses}L' if is_result else 'N/A':<18} | {f'{oos_result.wins}W / {oos_result.losses}L':<18}")
    print(f"  {'Win Rate':<28} | {f'{is_result.win_rate:.1%}' if is_result else 'N/A':<18} | {f'{oos_result.win_rate:.1%}':<18}")
    print(f"  {'Break-Even Win Rate':<28} | {f'{oos_result.break_even_win_rate:.1%}':<18} | {f'{oos_result.break_even_win_rate:.1%}':<18}")
    print(f"  {'Net P&L':<28} | {f'{is_result.net_pnl:+.2f}' if is_result else 'N/A':<18} | {f'{oos_result.net_pnl:+.2f}':<18}")
    print(f"  {'Expectancy / Trade':<28} | {f'{is_result.expectancy_per_trade:+.2f}' if is_result else 'N/A':<18} | {f'{oos_result.expectancy_per_trade:+.2f}':<18}")
    print(f"  {'Profit Factor':<28} | {f'{is_result.profit_factor:.3f}' if is_result else 'N/A':<18} | {f'{oos_result.profit_factor:.3f}':<18}")
    print(f"  {'Max Drawdown':<28} | {f'{is_result.max_drawdown:.1%}' if is_result else 'N/A':<18} | {f'{oos_result.max_drawdown:.1%}':<18}")
    print(f"  {'Ending Balance':<28} | {f'{is_result.ending_balance:.2f}' if is_result else 'N/A':<18} | {f'{oos_result.ending_balance:.2f}':<18}")
    print("=" * 75)

    if oos_result.trades:
        print("\nOut-of-Sample Settled Trades (Chronological Audit):")
        for trade in oos_result.trades:
            print(
                f"  {trade.signal_time} | {trade.direction.value:4} | Entry={trade.entry_price:.6f} "
                f"-> Expiry={trade.settlement_price:.6f} | {'WIN ' if trade.won else 'LOSS':4} "
                f"| P&L={trade.net_pnl:+.2f} | Score={trade.score:+.3f}"
            )
    else:
        print("\nNo out-of-sample trades met all confluence criteria (0 trades executed).")

    print("\n" + "-" * 75)
    print("  Paper-trading backtest only. Historical simulation does not guarantee")
    print("  future returns. Binary options carry substantial negative expectancy risks.")
    print("-" * 75)


if __name__ == "__main__":
    main()
