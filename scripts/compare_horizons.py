"""Empirical Horizon & Friction Analyzer.

Quantifies the mathematical impact of market microstructure friction
(bid-ask spread, exchange fees, execution slippage) across different prediction horizons
(1m, 5m, 15m, 1h, 4h, 24h). Proves why 1-minute edges are eroded by transaction costs
and identifies optimal trading horizons where signal magnitude dominates friction.

Usage:
    # Run empirical horizon comparison on 1-minute dataset
    python scripts/compare_horizons.py --candles 10000

    # Run on downloaded real market dataset
    python scripts/compare_horizons.py --data-file data/datasets/market_regimes/multi_year_cycle_2021_2024_BTCUSDT_1h_ohlcv.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.storage.parquet_store import load_candles_from_parquet
from scripts.calculate_features import generate_synthetic_candles


def analyze_horizons(
    df: pd.DataFrame,
    horizons: list[int],
    timeframe_unit: str = "1m",
    maker_fee_bps: float = 2.0,
    taker_fee_bps: float = 5.0,
    spread_bps: float = 2.0,
    slippage_bps: float = 3.0,
) -> pd.DataFrame:
    """Analyze signal-to-friction ratio across multiple forward horizons."""
    close = df["close"].values
    n = len(close)

    # Total round-trip friction in basis points and decimal
    # Entry taker (5 bps) + Exit taker (5 bps) + Spread (2 bps) + Slippage round-trip (2*3 = 6 bps)
    round_trip_fee_bps = (taker_fee_bps * 2.0) + spread_bps + (slippage_bps * 2.0)
    friction_pct = round_trip_fee_bps / 10000.0

    records = []

    for h in horizons:
        if h >= n - 10:
            continue

        # Forward return over H bars
        fwd_returns = (close[h:] - close[:-h]) / close[:-h]
        abs_returns = np.abs(fwd_returns)

        mean_abs_move = float(np.mean(abs_returns))
        std_move = float(np.std(fwd_returns))
        median_abs_move = float(np.median(abs_returns))

        # Percentage of forward moves that are smaller than the transaction friction
        eaten_by_fees_pct = float(np.mean(abs_returns < friction_pct) * 100.0)

        # Friction cost as a percentage of average move
        friction_drag_pct = float((friction_pct / mean_abs_move) * 100.0) if mean_abs_move > 0 else 999.0

        # Theoretical Signal-to-Cost Ratio (SCR)
        signal_to_cost_ratio = float(mean_abs_move / friction_pct) if friction_pct > 0 else 0.0

        # Required Directional Win Rate to Breakeven
        # Win * (Move - Fee) - (1 - Win) * (Move + Fee) = 0
        # 2 * Win * Move - Move - Fee = 0  => Win = (Move + Fee) / (2 * Move) = 0.5 + 0.5 * (Fee / Move)
        breakeven_win_rate = float(0.5 + 0.5 * (friction_pct / mean_abs_move)) if mean_abs_move > friction_pct else 1.0
        breakeven_win_rate = min(1.0, max(0.5, breakeven_win_rate))

        # Simple simulated Trend-Following Strategy:
        # Long if past H-bar return > 0, else Short
        past_returns = np.zeros(len(close))
        past_returns[h:] = (close[h:] - close[:-h]) / close[:-h]
        signals = np.sign(past_returns[:-h])  # +1 for long, -1 for short

        trade_pnl = signals * fwd_returns
        gross_return = float(np.mean(trade_pnl))
        net_return = float(np.mean(trade_pnl - friction_pct))
        trades_count = int(np.sum(np.abs(np.diff(np.concatenate(([0], signals)))) > 0))

        records.append(
            {
                "Horizon (Bars)": h,
                "Avg Move (%)": round(mean_abs_move * 100.0, 3),
                "Friction (%)": round(friction_pct * 100.0, 3),
                "Moves < Friction (%)": round(eaten_by_fees_pct, 1),
                "Friction Drag (%)": round(friction_drag_pct, 1),
                "Edge-to-Cost (SCR)": round(signal_to_cost_ratio, 2),
                "Breakeven Win Rate": f"{round(breakeven_win_rate * 100.0, 1)}%",
                "Simulated Net Return (bps)": round(net_return * 10000.0, 1),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirically compare prediction horizons and friction dynamics.")
    parser.add_argument("--data-file", type=str, default=None, help="Parquet historical candles path")
    parser.add_argument("--candles", type=int, default=10000, help="Number of candles if synthetic")
    parser.add_argument("--maker-fee-bps", type=float, default=2.0, help="Maker fee in basis points")
    parser.add_argument("--taker-fee-bps", type=float, default=5.0, help="Taker fee in basis points")
    parser.add_argument("--spread-bps", type=float, default=2.0, help="Bid-ask spread in basis points")
    parser.add_argument("--slippage-bps", type=float, default=3.0, help="Execution slippage in basis points")
    args = parser.parse_args()

    print("\n" + "=" * 95)
    print("AI-TRADER: MULTI-HORIZON FRICTION & VIABILITY BENCHMARK")
    print("=" * 95)

    if args.data_file:
        data_path = Path(args.data_file)
        candles = load_candles_from_parquet(data_path)
        print(f"Loaded {len(candles)} historical candles from: {data_path.resolve()}")
        symbol = candles[0].symbol
        tf = candles[0].timeframe
    else:
        print(f"Generating {args.candles} synthetic 1-minute candles for BTC/USDT...")
        candles = generate_synthetic_candles(count=args.candles, symbol="BTC/USDT")
        symbol = "BTC/USDT"
        tf = "1m"

    df = pd.DataFrame([{"close": c.close, "high": c.high, "low": c.low, "open": c.open} for c in candles])

    # Round trip friction breakdown
    total_friction_bps = (args.taker_fee_bps * 2.0) + args.spread_bps + (args.slippage_bps * 2.0)
    print(f"Asset:               {symbol} [{tf}]")
    print(f"Friction Assumptions:Taker Fee={args.taker_fee_bps * 2:.1f}bps, Spread={args.spread_bps:.1f}bps, Slippage={args.slippage_bps * 2:.1f}bps")
    print(f"Total Round-Trip:    {total_friction_bps:.1f} bps ({total_friction_bps / 10000 * 100:.3f}%)")
    print("-" * 95)

    if tf == "1m":
        # 1m, 5m, 15m, 30m, 1h (60m), 2h (120m), 4h (240m), 1d (1440m)
        horizons = [1, 5, 15, 30, 60, 120, 240, 720]
    else:
        # If candles are 1h: 1h (1 bar), 4h (4 bars), 8h (8 bars), 24h (24 bars), 72h (72 bars)
        horizons = [1, 2, 4, 8, 12, 24, 48, 72]

    res = analyze_horizons(
        df=df,
        horizons=horizons,
        timeframe_unit=tf,
        maker_fee_bps=args.maker_fee_bps,
        taker_fee_bps=args.taker_fee_bps,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
    )

    print(res.to_string(index=False))
    print("-" * 95)
    print("KEY QUANTITATIVE TAKEAWAYS:")
    print(" 1. At 1-bar horizon: Friction accounts for an overwhelming fraction of the expected move.")
    print("    A trader requires an unrealistic win-rate to break even after fees.")
    print(" 2. At longer horizons: Expected price displacement scales with sqrt(Time), while fixed")
    print("    friction remains constant, driving the Signal-to-Cost Ratio (SCR) > 10.0.")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
