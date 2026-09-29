"""Command line script to run historical pattern matching (Phase 4 Section 8-14).

Usage:
    python scripts/run_pattern_search.py --top-k 5 --sequence-window 20
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.pattern.engine import HistoricalPatternEngine
from features.engine import FeatureEngine
from scripts.calculate_features import generate_synthetic_candles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search historical states for analog conditions and outcome distributions."
    )
    parser.add_argument(
        "--candles", type=int, default=300, help="Number of historical candles to generate"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of nearest historical analogs to retrieve"
    )
    parser.add_argument(
        "--sequence-window",
        type=int,
        default=20,
        choices=[20, 50, 100],
        help="Sequence-matching window length (candles)",
    )
    args = parser.parse_args()

    print(f"Generating {args.candles} candles...")
    candles = generate_synthetic_candles(count=args.candles, symbol="BTC/USDT")

    fe = FeatureEngine()
    vectors = fe.compute_historical_features(candles, symbol="BTC/USDT")

    # Use the latest feature vector as the query state at time T
    query_vector = vectors[-1]
    query_time = query_vector.timestamp

    print(f"\nQuerying Historical Pattern Engine at T = {query_time.isoformat()}...")
    engine = HistoricalPatternEngine()

    # 1. State vector similarity search
    result = engine.find_similar_states(
        query_vector=query_vector,
        historical_vectors=vectors,
        historical_candles=candles,
        top_k=args.top_k,
        horizon_bars=15,
    )

    print("\n=== Top Historical Matches (Vector Similarity) ===")
    for idx, match in enumerate(result.matches, 1):
        print(
            f"  [{idx}] Match Time: {match.matched_timestamp.isoformat()} | "
            f"Similarity: {match.similarity_score:.4f} | "
            f"Forward Return: {match.forward_return:+.4f} | "
            f"MFE: {match.mfe:+.4f} | "
            f"MAE: {match.mae:+.4f}"
        )

    dist = result.distribution
    print("\n=== Conditional Outcome Distribution ===")
    print(f"  Sample Matches:    {dist.match_count}")
    print(f"  Mean Return:       {dist.mean_forward_return:+.4f}")
    print(f"  Median Return:     {dist.median_forward_return:+.4f}")
    print(f"  Std Dev:           {dist.std_forward_return:.4f}")
    print(f"  Win Rate (>0):     {dist.positive_outcome_ratio * 100:.1f}%")
    print(f"  Mean MFE:          {dist.mean_mfe:+.4f}")
    print(f"  Mean MAE:          {dist.mean_mae:+.4f}")
    print(f"  Quantiles:         {dist.quantiles}")

    # 2. Sequence pattern matching
    print(f"\nRunning Sequence Pattern Matching ({args.sequence_window} candles)...")
    seq_result = engine.find_sequence_patterns(
        query_candles=candles,
        historical_candles=candles,
        window_size=args.sequence_window,
        top_k=args.top_k,
        horizon_bars=15,
    )
    print(f"Found {len(seq_result.matches)} sequence matches.")
    if seq_result.matches:
        top_seq = seq_result.matches[0]
        print(
            f"  Best Sequence Match: {top_seq.matched_timestamp.isoformat()} | "
            f"Similarity: {top_seq.similarity_score:.4f} | "
            f"Forward Return: {top_seq.forward_return:+.4f}"
        )


if __name__ == "__main__":
    main()
