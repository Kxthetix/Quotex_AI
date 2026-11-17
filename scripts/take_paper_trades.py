"""CLI runner to execute and inspect N simulated paper trades with longer horizons.

Simulates realistic order execution with exchange fees, bid/ask spread, and slippage.
Tracks individual trade entries, exits, holding periods, friction costs, and net P&L.

Usage:
    # Execute 5 trades on real 1-hour BTC dataset
    python scripts/take_paper_trades.py --trades 5 --data-file data/datasets/market_regimes/etf_bull_expansion_2023_2024_BTCUSDT_1h_ohlcv.parquet

    # Execute 5 trades on the 3-year multi-regime dataset
    python scripts/take_paper_trades.py --trades 5 --data-file data/datasets/market_regimes/multi_year_cycle_2021_2024_BTCUSDT_1h_ohlcv.parquet
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.engine import BacktestEngine
from backtesting.events import SignalEvent
from backtesting.models import BacktestConfig, ExitReason, PositionSide, SizingMethod, TradeRecord
from data.ingestion.models import Candle
from data.storage.parquet_store import load_candles_from_parquet
from scripts.calculate_features import generate_synthetic_candles


def build_longer_horizon_intelligence(
    trend_lookback_fast: int = 9,
    trend_lookback_slow: int = 21,
    min_holding_bars: int = 4,
):
    """Factory creating a higher-timeframe trend & momentum intelligence provider."""
    bars_in_trade = 0

    def intelligence_provider(candle: Candle, history: list[Candle]) -> SignalEvent | None:
        nonlocal bars_in_trade
        if len(history) < trend_lookback_slow + 5:
            return None

        closes = [c.close for c in history]
        sma_fast = sum(closes[-trend_lookback_fast:]) / float(trend_lookback_fast)
        sma_slow = sum(closes[-trend_lookback_slow:]) / float(trend_lookback_slow)

        # Simple RSI(14)
        deltas = [closes[i] - closes[i - 1] for i in range(-14, 0)]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        avg_gain = sum(gains) / 14.0 if gains else 0.0
        avg_loss = sum(losses) / 14.0 if losses else 1e-6
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        curr_price = candle.close

        # Signal logic:
        # Long when fast EMA crosses above slow EMA and RSI is healthy (40-65)
        # Exit or Short when fast EMA crosses below slow EMA or RSI extreme
        action = "HOLD"
        direction_prob = 0.50
        regime = "SIDEWAYS"

        if sma_fast > sma_slow * 1.002 and 40.0 <= rsi <= 65.0:
            action = "ENTER_LONG"
            direction_prob = 0.70
            regime = "TREND_UP"
            bars_in_trade = 0
        elif sma_fast < sma_slow * 0.998 and 35.0 <= rsi <= 60.0:
            action = "ENTER_SHORT"
            direction_prob = 0.30
            regime = "TREND_DOWN"
            bars_in_trade = 0
        else:
            bars_in_trade += 1
            # Respect minimum holding horizon (e.g. 4 bars / 4 hours) before exit on momentum decay
            if bars_in_trade >= min_holding_bars:
                if (sma_fast < sma_slow and rsi > 65.0) or (sma_fast > sma_slow and rsi < 35.0):
                    action = "EXIT"

        return SignalEvent(
            timestamp=candle.end_time,
            symbol=candle.symbol,
            direction_prob=direction_prob,
            expected_return=0.015 if action != "HOLD" else 0.0,
            expected_volatility=0.25,
            trade_quality_prob=0.68,
            regime=regime,
            suggested_action=action,
            strength=0.80,
        )

    return intelligence_provider


def run_paper_trades(
    candles: list[Candle],
    target_trades: int = 5,
    initial_capital: float = 10000.0,
    taker_fee: float = 0.0005,  # 5 bps
    spread_bps: float = 2.0,    # 2 bps
    slippage_bps: float = 3.0,  # 3 bps
    min_holding_bars: int = 4,  # e.g. 4-hour minimum trade horizon
) -> list[TradeRecord]:
    """Execute backtest simulation until exactly target_trades are completed."""
    config = BacktestConfig(
        symbol=candles[0].symbol if candles else "BTC/USDT",
        timeframe=candles[0].timeframe if candles else "1h",
        initial_capital=initial_capital,
        taker_fee=taker_fee,
        maker_fee=taker_fee * 0.4,
        spread_pct=spread_bps / 10000.0,
        slippage_pct=slippage_bps / 10000.0,
        sizing_method=SizingMethod.PERCENT_CAPITAL,
        sizing_value=0.20,  # 20% position size per trade
    )

    intelligence = build_longer_horizon_intelligence(min_holding_bars=min_holding_bars)
    engine = BacktestEngine(config=config, intelligence_provider=intelligence)

    # Stream candles sequentially and stop once target_trades completed
    completed_trades: list[TradeRecord] = []
    
    # We run the backtest over the full candle sequence
    result = engine.run(candles)
    all_trades = result.trade_ledger

    if len(all_trades) >= target_trades:
        selected_trades = all_trades[:target_trades]
    else:
        selected_trades = all_trades

    return selected_trades


def print_trade_details(trades: list[TradeRecord], initial_capital: float = 10000.0) -> None:
    """Print institutional-grade audit ledger for the executed trades."""
    print("\n" + "=" * 90)
    print(f"AI-TRADER: EXECUTED {len(trades)} COMPLETED PAPER TRADES (LONGER HORIZON)")
    print("=" * 90)

    running_equity = initial_capital
    total_gross_pnl = 0.0
    total_net_pnl = 0.0
    total_fees = 0.0
    total_slippage = 0.0
    total_spread = 0.0
    wins = 0

    for idx, t in enumerate(trades, 1):
        is_win = t.net_pnl > 0
        if is_win:
            wins += 1

        total_gross_pnl += t.gross_pnl
        total_net_pnl += t.net_pnl
        total_fees += t.fees
        total_slippage += t.slippage
        total_spread += t.spread
        running_equity += t.net_pnl

        holding_hours = round(t.holding_period_seconds / 3600.0, 1)
        side_str = "LONG (BUY)" if t.side == PositionSide.LONG else "SHORT (SELL)"
        status_tag = "[WIN]" if is_win else "[LOSS]"

        print(f"\nTRADE #{idx:02d} | {status_tag} {side_str} on {t.symbol}")
        print(f"  Entry Time:      {t.entry_time.strftime('%Y-%m-%d %H:%M UTC')} @ ${t.entry_price:,.2f}")
        print(f"  Exit Time:       {t.exit_time.strftime('%Y-%m-%d %H:%M UTC')} @ ${t.exit_price:,.2f}")
        print(f"  Trade Horizon:   {holding_hours} hours ({t.holding_period_seconds/60:.0f} mins) | Exit Reason: {t.exit_reason.value}")
        print(f"  Regime at Entry: {t.regime_at_entry.value if hasattr(t.regime_at_entry, 'value') else t.regime_at_entry}")
        print(f"  Quantity Traded: {t.quantity:.4f} {t.symbol[:3]} (Notional: ${t.quantity * t.entry_price:,.2f})")
        print(f"  Friction Costs:  Fees: ${t.fees:.2f} | Spread: ${t.spread:.2f} | Slippage: ${t.slippage:.2f} (Total: ${t.fees + t.spread + t.slippage:.2f})")
        print(f"  Gross P&L:       ${t.gross_pnl:+,.2f} ({t.return_pct:+.2f}%)")
        print(f"  Net P&L (Real):  ${t.net_pnl:+,.2f} (Net Return: {(t.net_pnl / (t.quantity * t.entry_price))*100:+.2f}%)")
        print(f"  Portfolio Cash:  ${running_equity:,.2f}")
        print("-" * 90)

    win_rate = (wins / len(trades)) * 100.0 if trades else 0.0
    profit_factor = (
        abs(sum(t.net_pnl for t in trades if t.net_pnl > 0)) / abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        if any(t.net_pnl < 0 for t in trades)
        else float("inf")
    )

    print("\n" + "=" * 90)
    print("SUMMARY PERFORMANCE REPORT (5 TRADES)")
    print("=" * 90)
    print(f"Initial Capital:         ${initial_capital:,.2f}")
    print(f"Final Capital:           ${running_equity:,.2f} ({((running_equity - initial_capital)/initial_capital)*100:+.2f}%)")
    print(f"Win Rate:                {win_rate:.1f}% ({wins}/{len(trades)} wins)")
    print(f"Profit Factor:           {profit_factor:.2f}" if profit_factor != float("inf") else "Profit Factor:           Inf (Zero losses)")
    print(f"Total Gross P&L:         ${total_gross_pnl:+,.2f}")
    print(f"Total Friction Deducted: ${total_fees + total_spread + total_slippage:,.2f} (Fees: ${total_fees:.2f}, Slippage: ${total_slippage:.2f}, Spread: ${total_spread:.2f})")
    print(f"Total Net P&L:           ${total_net_pnl:+,.2f}")
    print("=" * 90 + "\n")

    # Export markdown report
    rep_dir = Path("reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    md_path = rep_dir / "paper_trades_5.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Paper Trading Execution Audit — 5 Longer-Horizon Trades\n\n")
        f.write(f"- **Initial Capital**: ${initial_capital:,.2f}\n")
        f.write(f"- **Final Capital**: ${running_equity:,.2f} ({((running_equity - initial_capital)/initial_capital)*100:+.2f}%)\n")
        f.write(f"- **Win Rate**: {win_rate:.1f}% ({wins}/{len(trades)} wins)\n")
        f.write(f"- **Profit Factor**: {profit_factor:.2f}\n")
        f.write(f"- **Total Gross P&L**: ${total_gross_pnl:+,.2f}\n")
        f.write(f"- **Total Friction Deducted**: ${total_fees + total_spread + total_slippage:,.2f}\n")
        f.write(f"- **Net P&L**: ${total_net_pnl:+,.2f}\n\n")
        f.write("| # | Side | Entry Time | Entry Price | Exit Time | Exit Price | Horizon (h) | Gross P&L | Friction | Net P&L | Return % |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for idx, t in enumerate(trades, 1):
            side = "LONG" if t.side == PositionSide.LONG else "SHORT"
            fric = t.fees + t.spread + t.slippage
            f.write(f"| {idx:02d} | {side} | {t.entry_time.strftime('%Y-%m-%d %H:%M')} | ${t.entry_price:,.2f} | {t.exit_time.strftime('%Y-%m-%d %H:%M')} | ${t.exit_price:,.2f} | {t.holding_period_seconds/3600:.1f}h | ${t.gross_pnl:+,.2f} | ${fric:.2f} | ${t.net_pnl:+,.2f} | {(t.net_pnl / (t.quantity * t.entry_price))*100:+.2f}% |\n")
    print(f"Report exported to: {md_path.resolve()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Take N paper trades on real historical market data.")
    parser.add_argument("--trades", type=int, default=5, help="Number of completed trades to take")
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/datasets/market_regimes/etf_bull_expansion_2023_2024_BTCUSDT_1h_ohlcv.parquet",
        help="Path to Parquet dataset",
    )
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital in USD")
    parser.add_argument("--min-holding-bars", type=int, default=4, help="Minimum holding horizon in bars (e.g. 4 bars = 4h)")
    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        # Fallback to multi_year_cycle if etf_bull_expansion not found
        fallback = Path("data/datasets/market_regimes/multi_year_cycle_2021_2024_BTCUSDT_1h_ohlcv.parquet")
        if fallback.exists():
            data_path = fallback
        else:
            raise FileNotFoundError(f"Parquet dataset not found at {data_path}")

    candles = load_candles_from_parquet(data_path)
    trades = run_paper_trades(
        candles=candles,
        target_trades=args.trades,
        initial_capital=args.capital,
        min_holding_bars=args.min_holding_bars,
    )

    print_trade_details(trades, initial_capital=args.capital)


if __name__ == "__main__":
    main()
