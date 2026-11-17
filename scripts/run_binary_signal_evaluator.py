"""Synthetic candle rule inspector; not a trading or profitability test.

Evaluates experimental candle and momentum filters and prints a paper signal.

Usage:
    python scripts/run_binary_signal_evaluator.py --asset EUR/CHF --sentiment 96
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.storage.parquet_store import load_candles_from_parquet
from scripts.calculate_features import generate_synthetic_candles
from signals.binary_scalping_engine import BinaryAction, BinaryScalpingEngine


def run_evaluation(
    asset: str = "EUR/CHF",
    crowd_sentiment_bullish: float | None = None,
    data_file: str | None = None,
) -> None:
    print("=" * 75)
    if data_file:
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        candles = load_candles_from_parquet(data_path)
        if not candles:
            raise ValueError(f"No candles loaded from {data_file}")
        asset = candles[-1].symbol
        print("  AI-TRADER: BINARY STRATEGY EVALUATION (HISTORICAL COMPLETED BARS)")
        print(f"  Data Source      : {data_file} ({len(candles)} completed bars)")
    else:
        print("  [SMOKE-TEST ONLY - SYNTHETIC DATA FIXTURE]")
        print("  Candles below are synthetically generated strictly for rule smoke-testing.")
        print("  Synthetic results MUST NOT be used for strategy validation or profitability claims.")
        print("=" * 75)
        print(f"  AI-TRADER: BINARY MODEL RULE SMOKE-TEST (SYNTHETIC)")
        candles = generate_synthetic_candles(count=50, symbol=asset)

    sentiment_text = (
        f"{crowd_sentiment_bullish:.0f}% Bullish"
        if crowd_sentiment_bullish is not None
        else "not supplied"
    )
    print(f"  Asset: {asset} | Retail Crowd Sentiment: {sentiment_text}")
    print("=" * 75)

    engine = BinaryScalpingEngine(min_confidence=0.65, rsi_overbought=68.0, rsi_oversold=32.0)

    result = engine.evaluate(candles, asset=asset, crowd_sentiment_bullish_pct=crowd_sentiment_bullish)
    tech = result.technical_state

    if tech:
        print("\n[1] TECHNICAL INDICATOR CONFLUENCE:")
        print(f"  Current Price     : {tech.price:.5f}")
        print(f"  EMA-9 / EMA-21    : {tech.ema_9:.5f} / {tech.ema_21:.5f} ({tech.trend_state})")
        print(f"  RSI (14)          : {tech.rsi_14:.1f} ({tech.momentum_state})")
        print(f"  MACD Hist         : {tech.macd_hist:+.6f}")
        print(f"  Bollinger Bands   : [{tech.bb_lower:.5f} ... {tech.bb_middle:.5f} ... {tech.bb_upper:.5f}]")
        print(f"  Candle Geometry   : Body={tech.candle.body_pct*100:.0f}%, UpperWick={tech.candle.upper_wick:.5f}, LowerWick={tech.candle.lower_wick:.5f}")
        print(f"  Candle Pattern    : {'HAMMER' if tech.candle.is_hammer else ('SHOOTING STAR' if tech.candle.is_shooting_star else ('DOJI' if tech.candle.is_doji else 'NORMAL'))}")

    print("\n[2] REJECTION FLAGS & SAFEGUARDS:")
    if result.rejection_flags:
        for flag in result.rejection_flags:
            print(f"  [BLOCKED] {flag}")
    else:
        print("  None. All safety criteria satisfied.")

    print("\n[3] MODEL DECISION & AUDIT:")
    print(f"  Action            : {result.action.value}")
    print(f"  Rule Score        : {result.score:+.4f} (not a win probability)")
    print("  Diagnostic Log:")
    for reason in result.reasons:
        print(f"   * {reason}")

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Binary Scalping Model")
    parser.add_argument("--asset", type=str, default="EUR/CHF", help="Trading asset pair")
    parser.add_argument("--sentiment", type=float, default=None, help="Optional broker sentiment % bullish")
    parser.add_argument("--data-file", type=str, default=None, help="Optional path to historical completed candles parquet file")
    args = parser.parse_args()

    run_evaluation(asset=args.asset, crowd_sentiment_bullish=args.sentiment, data_file=args.data_file)


if __name__ == "__main__":
    main()
