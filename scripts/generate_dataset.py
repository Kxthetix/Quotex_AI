"""Command line script to generate versioned baseline datasets (Phase 4).

Usage:
    python scripts/generate_dataset.py --symbol BTC/USDT --candles 500 --output data/datasets/btc_1m_dataset.parquet
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learning.dataset import DatasetGenerator
from scripts.calculate_features import generate_synthetic_candles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate version-locked dataset pairing Features(T) with Labels(T+H)."
    )
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1m", help="Candle timeframe")
    parser.add_argument("--candles", type=int, default=500, help="Number of candles to process")
    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/baseline_dataset.parquet",
        help="Output parquet path",
    )
    args = parser.parse_args()

    print(f"Generating {args.candles} candles for {args.symbol} [{args.timeframe}]...")
    candles = generate_synthetic_candles(count=args.candles, symbol=args.symbol)

    generator = DatasetGenerator()
    df, metadata = generator.create_dataset(
        candles=candles,
        symbol=args.symbol,
        timeframe=args.timeframe,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print("\nDataset generation completed successfully!")
    print(f"  Dataset ID:       {metadata.dataset_id}")
    print(f"  Feature Version:  {metadata.feature_version}")
    print(f"  Label Version:    {metadata.label_version}")
    print(f"  Observations:     {len(df)}")
    print(f"  Features Count:   {len(metadata.feature_columns)}")
    print(f"  Target Columns:   {metadata.target_columns}")
    print(f"  Saved to:         {out_path.resolve()}")


if __name__ == "__main__":
    main()
