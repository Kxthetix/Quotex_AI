"""Command line script to calculate and inspect versioned market features (Phase 3).

Usage:
    python scripts/calculate_features.py --symbol BTC/USDT --count 300 --output data/features/btc_v1.parquet
"""

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from data.ingestion.models import Candle
from data.storage.features_store import save_features_to_parquet
from data.storage.parquet_store import load_candles_from_parquet
from features.config import FeatureEngineConfig
from features.engine import FeatureEngine
from features.registry import FeatureRegistry


def generate_synthetic_candles(count: int, symbol: str) -> list[Candle]:
    """Generate synthetic OHLCV candles strictly for smoke-testing and pipeline verification.

    CAUTION: Synthetic data must NEVER be used for strategy validation, historical
    performance claims, or profitability demonstrations. It serves strictly as a
    software verification fixture.
    """
    np.random.seed(42)
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    delta = timedelta(minutes=1)
    candles = []
    curr = 50000.0

    for i in range(count):
        s = start + i * delta
        e = s + delta
        step = np.sin(i / 10.0) * 15.0 + (np.random.rand() - 0.48) * 20.0
        open_p = curr
        close_p = open_p + step
        high_p = max(open_p, close_p) + abs(np.random.rand() * 10.0)
        low_p = min(open_p, close_p) - abs(np.random.rand() * 10.0)
        volume = float(np.random.randint(100, 500) + abs(np.sin(i / 5.0)) * 200.0)

        candles.append(
            Candle(
                symbol=symbol,
                timeframe="1m",
                open=round(open_p, 4),
                high=round(high_p, 4),
                low=round(low_p, 4),
                close=round(close_p, 4),
                volume=round(volume, 2),
                start_time=s,
                end_time=e,
                trades_count=100,
                is_closed=True,
            )
        )
        curr = close_p
    return candles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate versioned market features for historical data."
    )
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol to process")
    parser.add_argument(
        "--input-parquet", type=str, default=None, help="Input candles parquet file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/features/sample_features.parquet",
        help="Output features parquet file",
    )
    parser.add_argument(
        "--count", type=int, default=300, help="Number of synthetic candles if input not provided"
    )
    parser.add_argument(
        "--config", type=str, default="configs/features/v1.yaml", help="Feature config YAML"
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config = FeatureEngineConfig.load_from_yaml(config_path)
    else:
        config = FeatureEngineConfig()

    reg = FeatureRegistry()
    engine = FeatureEngine(config=config, registry=reg)

    if args.input_parquet and Path(args.input_parquet).exists():
        print(f"Loading candles from {args.input_parquet}...")
        candles = load_candles_from_parquet(args.input_parquet)
    else:
        print(f"Generating {args.count} synthetic candles for {args.symbol}...")
        candles = generate_synthetic_candles(count=args.count, symbol=args.symbol)

    print(f"Computing historical features using feature version '{config.version}'...")
    vectors = engine.compute_historical_features(candles)

    metrics = engine.get_metrics()
    print("\n" + "=" * 50)
    print("FEATURE ENGINE CALCULATION REPORT")
    print("=" * 50)
    print(f"Total Feature Vectors Calculated : {metrics['total_vectors_calculated']}")
    print(f"Valid Vectors Count             : {metrics['valid_vectors_count']}")
    print(f"Warming Up Vectors Count        : {metrics['warming_up_vectors_count']}")
    print(f"Invalid Vectors Count           : {metrics['invalid_vectors_count']}")
    print(f"Calculation Errors Count        : {metrics['calculation_errors_count']}")
    print(f"Total Registered Features       : {reg.count()}")

    if vectors:
        out_path = Path(args.output)
        save_features_to_parquet(vectors, out_path)
        print(f"\nPersisted {len(vectors)} vectors to Parquet at: {out_path}")

        latest = vectors[-1]
        print(f"\nSample Features at latest timestamp ({latest.timestamp}):")
        print(f"  Symbol: {latest.symbol} | Valid: {latest.is_valid}")
        for idx, (k, v) in enumerate(latest.values.items()):
            if idx < 12:
                status = latest.validity[k].value
                val_str = f"{v:.6f}" if v is not None else "None"
                print(f"  - {k:<28} : {val_str:<12} [{status}]")
        if len(latest.values) > 12:
            print(f"  ... ({len(latest.values) - 12} additional features)")
    print("=" * 50)


if __name__ == "__main__":
    main()
