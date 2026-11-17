"""Command line tool to fetch large historical datasets across diverse market regimes.

Supports downloading curated historical regime periods (Bull Mania, Bear Winter,
Chop/Consolidation, Liquidity Shock) or custom multi-month/multi-year windows.
Validates, calculates indicators, classifies market regimes, and exports versioned Parquet datasets.

Usage:
    # List all available regime presets
    python scripts/fetch_regime_datasets.py --list

    # Download 2020-2021 secular bull market on 1h timeframe
    python scripts/fetch_regime_datasets.py --preset bull_parabolic_2020_2021 --timeframe 1h

    # Download 2022 crypto winter bear market
    python scripts/fetch_regime_datasets.py --preset bear_market_2022 --timeframe 1h

    # Download a full 3-year continuous macro cycle (2021-2024)
    python scripts/fetch_regime_datasets.py --preset multi_year_cycle_2021_2024 --timeframe 1h

    # Download custom multi-year range
    python scripts/fetch_regime_datasets.py --start 2022-01-01 --end 2024-01-01 --name btc_2yr_macro --timeframe 4h
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.regime.catalog import REGIME_PRESETS, get_regime_preset, list_regime_presets
from analysis.regime.dataset_builder import RegimeDatasetBuilder, RegimeProfile
from configs.config import get_settings
from monitoring.logging import setup_logging


def print_presets_table() -> None:
    """Print formatted list of curated market regime presets."""
    presets = list_regime_presets()
    print("\n" + "=" * 105)
    print(f"{'PRESET ID':<30} | {'CATEGORY':<22} | {'DAYS':<5} | {'START -> END':<25}")
    print("=" * 105)
    for p in presets:
        dates_str = f"{p.start.strftime('%Y-%m-%d')} -> {p.end.strftime('%Y-%m-%d')}"
        print(f"{p.preset_id:<30} | {p.category.value:<22} | {p.duration_days:<5} | {dates_str:<25}")
        print(f"  Description: {p.description}")
        print(f"  Primary Regimes: {', '.join(r.value for r in p.expected_primary_regimes)}")
        print("-" * 105)
    print()


def print_profile_summary(profile: RegimeProfile) -> None:
    """Print visual summary of the generated dataset and its regime distribution."""
    print("\n" + "=" * 75)
    print(f"DATASET REGIME PROFILE: {profile.dataset_name.upper()}")
    print("=" * 75)
    print(f"Symbol:           {profile.symbol}")
    print(f"Timeframe:        {profile.timeframe}")
    print(f"Total Candles:    {profile.total_candles:,}")
    print(f"Time Range:       {profile.start_time[:19]} -> {profile.end_time[:19]}")
    print(f"Price Action:     ${profile.open_price:,.2f} -> ${profile.close_price:,.2f} (High: ${profile.high_price:,.2f}, Low: ${profile.low_price:,.2f})")
    print(f"Total Return:     {profile.total_return_pct:+.2f}%")
    print(f"Max Drawdown:     -{profile.max_drawdown_pct:.2f}%")
    print("-" * 75)
    print("DETECTED REGIME DISTRIBUTION:")
    for regime_name, data in sorted(profile.regime_distribution.items(), key=lambda x: x[1]["count"], reverse=True):
        pct = data["pct"]
        cnt = data["count"]
        bar_len = int(pct / 2.5)
        bar = "#" * bar_len
        avg_streak = profile.average_regime_streak_bars.get(regime_name, 0.0)
        print(f"  {regime_name:<16} : {pct:>5.1f}% ({cnt:>6,} bars) [avg streak: {avg_streak:>4.1f} bars] {bar}")
    print("-" * 75)
    print("OUTPUT FILES:")
    print(f"  OHLCV Parquet:    {profile.output_parquet_path}")
    print(f"  Annotated Parquet:{profile.output_annotated_parquet_path}")
    print("=" * 75 + "\n")


async def run_fetch(args: argparse.Namespace) -> None:
    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)
    builder = RegimeDatasetBuilder()

    if args.list:
        print_presets_table()
        return

    # Case 1: Fetch all presets
    if args.all_presets:
        print(f"\nFetching all {len(REGIME_PRESETS)} curated regime presets for {args.symbol} [{args.timeframe}]...")
        for preset_id, preset in REGIME_PRESETS.items():
            print(f"\nProcessing preset: {preset.name}...")
            _, profile = await builder.fetch_and_build_preset(
                preset=preset,
                symbol=args.symbol,
                timeframe=args.timeframe,
                output_dir=args.output_dir,
            )
            print_profile_summary(profile)
        print("\nAll presets downloaded and profiled successfully!\n")
        return

    # Case 2: Fetch a specific preset
    if args.preset:
        preset = get_regime_preset(args.preset)
        print(f"\nFetching preset '{preset.name}' for {args.symbol} [{args.timeframe}]...")
        _, profile = await builder.fetch_and_build_preset(
            preset=preset,
            symbol=args.symbol,
            timeframe=args.timeframe,
            output_dir=args.output_dir,
        )
        print_profile_summary(profile)
        return

    # Case 3: Custom date range
    if args.start:
        start_dt = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)

        end_dt = None
        if args.end:
            end_dt = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=UTC)

        dataset_name = args.name or f"custom_{args.symbol}_{start_dt.strftime('%Y%m%d')}"
        print(f"\nFetching custom range for {args.symbol} [{args.timeframe}]: {start_dt.isoformat()} -> {end_dt.isoformat() if end_dt else 'latest'}...")
        candles = await builder.fetch_candles(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=start_dt,
            end=end_dt,
            limit=args.limit,
        )
        print(f"Received {len(candles)} raw candles. Building features and classifying regimes...")
        _, profile = builder.build_and_profile(
            candles=candles,
            dataset_name=dataset_name,
            output_dir=args.output_dir,
        )
        print_profile_summary(profile)
        return

    # Default if no actionable arguments passed: show help
    print("No download target specified. Use --list to view presets, --preset <id> to fetch, or --help for options.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and profile large candlestick datasets across diverse market regimes."
    )
    parser.add_argument(
        "--list", action="store_true", help="List all curated market regime presets"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Identifier of the regime preset to fetch (e.g. bull_parabolic_2020_2021, bear_market_2022, multi_year_cycle_2021_2024)",
    )
    parser.add_argument(
        "--all-presets", action="store_true", help="Download all curated regime presets"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading symbol (e.g. BTCUSDT, ETHUSDT)"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (e.g. 1m, 5m, 15m, 1h, 4h, 1d; default 1h)",
    )
    parser.add_argument(
        "--start", type=str, default=None, help="Start ISO timestamp for custom range (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=None, help="End ISO timestamp for custom range (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum candle limit for custom range"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Custom dataset identifier name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets/market_regimes",
        help="Output directory for generated Parquet datasets",
    )

    args = parser.parse_args()
    asyncio.run(run_fetch(args))


if __name__ == "__main__":
    main()
