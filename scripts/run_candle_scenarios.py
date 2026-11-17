"""Compare candle-strategy confidence settings with chronological holdout evaluation.

The validation partition selects a rule-confluence threshold. The final test partition
is evaluated once with that selected threshold and is never used for selection.
This is a research tool; its score is not a calibrated win probability.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.binary_options import BinaryBacktestResult, run_binary_options_backtest
from data.storage.parquet_store import load_candles_from_parquet
from signals.binary_scalping_engine import BinaryScalpingEngine


def evaluate(
    candles,
    *,
    threshold: float,
    payout: float,
    expiry_bars: int,
    stake: float,
    starting_balance: float,
    max_stake_fraction: float,
    first_signal_index: int,
) -> BinaryBacktestResult:
    engine = BinaryScalpingEngine(min_confidence=threshold)
    return run_binary_options_backtest(
        candles,
        engine,
        payout_rate=payout,
        expiry_bars=expiry_bars,
        stake=stake,
        starting_balance=starting_balance,
        max_stake_fraction=max_stake_fraction,
        first_signal_index=first_signal_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a candle-strategy what-if comparison: choose a score threshold on a "
            "validation period, then evaluate it on a later untouched period. No trades are placed."
        )
    )
    parser.add_argument("--data-file", required=True, help="Historical completed-candle Parquet file")
    parser.add_argument("--payout", type=float, required=True, help="Net win payout (93%% = 0.93)")
    parser.add_argument("--expiry-bars", type=int, default=1)
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--starting-balance", type=float, default=1_000.0)
    parser.add_argument("--max-stake-fraction", type=float, default=0.01)
    parser.add_argument("--min-validation-trades", type=int, default=20)
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="Use only the most recent N bars to bound exploratory run time",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.55, 0.65, 0.75],
        help="Rule-confluence thresholds to compare on validation only",
    )
    args = parser.parse_args()

    if len(args.thresholds) < 2 or any(not 0.0 < value <= 1.0 for value in args.thresholds):
        parser.error("Provide at least two thresholds, each in the range (0, 1].")
    if args.min_validation_trades < 1:
        parser.error("--min-validation-trades must be positive.")
    if args.max_bars is not None and args.max_bars < 180:
        parser.error("--max-bars must be at least 180.")

    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    candles = sorted(load_candles_from_parquet(data_path), key=lambda candle: candle.end_time)
    if args.max_bars is not None:
        candles = candles[-args.max_bars :]
    if len(candles) < 180:
        raise ValueError("At least 180 candles are required for warm-up and three chronological partitions.")

    # Keep the validation and test partitions large enough to produce useful evidence.
    train_end = int(len(candles) * 0.60)
    validation_end = int(len(candles) * 0.80)
    if train_end < 50 or validation_end - train_end < 30 or len(candles) - validation_end < 30:
        raise ValueError("The data set is too short for 60/20/20 train-validation-test partitions.")

    # Prefix bars provide indicator warm-up, while first_signal_index fences each period.
    validation_results = [
        (
            threshold,
            evaluate(
                candles,
                threshold=threshold,
                payout=args.payout,
                expiry_bars=args.expiry_bars,
                stake=args.stake,
                starting_balance=args.starting_balance,
                max_stake_fraction=args.max_stake_fraction,
                first_signal_index=train_end,
            ),
        )
        for threshold in args.thresholds
    ]

    eligible = [
        pair for pair in validation_results if pair[1].total_trades >= args.min_validation_trades
    ]
    selection_pool = eligible or validation_results
    selected_threshold, selected_validation = max(
        selection_pool,
        key=lambda pair: (
            pair[1].expectancy_per_trade,
            -pair[1].max_drawdown,
            pair[1].total_trades,
        ),
    )
    test_result = evaluate(
        candles,
        threshold=selected_threshold,
        payout=args.payout,
        expiry_bars=args.expiry_bars,
        stake=args.stake,
        starting_balance=args.starting_balance,
        max_stake_fraction=args.max_stake_fraction,
        first_signal_index=validation_end,
    )
    stress_payout = max(0.01, args.payout - 0.05)
    stress_result = evaluate(
        candles,
        threshold=selected_threshold,
        payout=stress_payout,
        expiry_bars=args.expiry_bars,
        stake=args.stake,
        starting_balance=args.starting_balance,
        max_stake_fraction=args.max_stake_fraction,
        first_signal_index=validation_end,
    )

    symbol = candles[0].symbol
    timeframe = candles[0].timeframe
    is_synthetic = "SYNTHETIC" in symbol.upper() or "synthetic" in str(data_path).lower()
    print("CANDLE STRATEGY WHAT-IF SIMULATION — PAPER ONLY")
    print(f"Data: {symbol} / {timeframe} | {len(candles)} completed candles")
    print(
        f"Partitions: train 0–{train_end - 1}, validation {train_end}–{validation_end - 1}, "
        f"untouched test {validation_end}–{len(candles) - 1}"
    )
    print(f"Payout: {args.payout:.1%} | Break-even win rate: {1 / (1 + args.payout):.2%}")
    if is_synthetic:
        print("WARNING: Synthetic data detected. Results are smoke-test output, not performance evidence.")
    if not (timeframe == "1m" and "OTC" in symbol.upper()):
        print("LIMITATION: This data does not match a 1-minute OTC binary feed; it cannot validate that market.")

    print("\nValidation-only threshold comparison (selection period):")
    print(" threshold | trades | win rate | expectancy | max drawdown")
    for threshold, result in validation_results:
        print(
            f" {threshold:9.2f} | {result.total_trades:6d} | {result.win_rate:8.2%} | "
            f"{result.expectancy_per_trade:10.3f} | {result.max_drawdown:11.2%}"
        )

    if not eligible:
        print(
            f"\nINSUFFICIENT VALIDATION EVIDENCE: no threshold reached "
            f"{args.min_validation_trades} trades; selected setting is exploratory only."
        )
    print(
        f"\nSelected threshold: {selected_threshold:.2f} "
        f"(validation expectancy {selected_validation.expectancy_per_trade:+.3f} per trade)"
    )
    print("Untouched test period:")
    print(
        f" trades={test_result.total_trades}, win_rate={test_result.win_rate:.2%}, "
        f"expectancy={test_result.expectancy_per_trade:+.3f}, "
        f"profit_factor={test_result.profit_factor:.3f}, "
        f"max_drawdown={test_result.max_drawdown:.2%}, net_pnl={test_result.net_pnl:+.2f}"
    )
    print(f"Payout stress test ({stress_payout:.1%} payout, same test period):")
    print(
        f" trades={stress_result.total_trades}, win_rate={stress_result.win_rate:.2%}, "
        f"expectancy={stress_result.expectancy_per_trade:+.3f}, "
        f"max_drawdown={stress_result.max_drawdown:.2%}, net_pnl={stress_result.net_pnl:+.2f}"
    )
    print(
        "\nThe engine's score is rule confluence, not a calibrated win probability. "
        "A threshold is not promoted by this script, and historical outcomes do not guarantee future results."
    )


if __name__ == "__main__":
    main()
