"""Payout-aware, paper-only backtest for fixed-expiry binary signals."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from data.ingestion.models import Candle
from signals.binary_scalping_engine import BinaryAction, BinaryScalpingEngine


@dataclass(frozen=True)
class BinaryOptionTrade:
    """One settled fixed-expiry trade, including the exact bars used for entry and settlement."""

    asset: str
    direction: BinaryAction
    signal_time: str
    entry_time: str
    expiry_time: str
    entry_price: float
    settlement_price: float
    stake: float
    payout_rate: float
    score: float
    won: bool
    net_pnl: float


@dataclass(frozen=True)
class BinaryBacktestResult:
    """Payout-adjusted results from a chronological paper backtest."""

    trades: tuple[BinaryOptionTrade, ...]
    skipped_signals: int
    starting_balance: float
    ending_balance: float
    net_pnl: float
    win_rate: float
    break_even_win_rate: float
    profit_factor: float
    expectancy_per_trade: float
    max_drawdown: float

    @property
    def total_trades(self) -> int:
        """Total number of executed and settled trades."""
        return len(self.trades)

    @property
    def wins(self) -> int:
        """Number of winning trades."""
        return sum(1 for t in self.trades if t.won)

    @property
    def losses(self) -> int:
        """Number of losing trades (including ties)."""
        return sum(1 for t in self.trades if not t.won)


def run_binary_options_backtest(
    candles: list[Candle],
    engine: BinaryScalpingEngine,
    *,
    payout_rate: float,
    expiry_bars: int = 1,
    stake: float = 10.0,
    starting_balance: float = 1_000.0,
    max_stake_fraction: float = 0.01,
    first_signal_index: int = 49,
) -> BinaryBacktestResult:
    """Backtest a signal at a completed bar close, enter next bar open, and settle later.

    ``payout_rate`` is net profit on a winning stake (for example, 0.93 for a 93% payout).
    A tie is treated as a loss conservatively. Only one trade may be open at a time.
    This is a historical simulation, not a broker execution interface.
    """
    if not candles:
        raise ValueError("At least one candle is required.")
    if not 0.0 < payout_rate <= 1.0:
        raise ValueError("payout_rate must be greater than 0 and at most 1.0.")
    if expiry_bars < 1:
        raise ValueError("expiry_bars must be at least 1.")
    if not 0.0 < max_stake_fraction <= 1.0:
        raise ValueError("max_stake_fraction must be in the range (0, 1].")
    if stake <= 0.0 or starting_balance <= 0.0:
        raise ValueError("Stake and starting balance must be positive.")
    if stake > starting_balance * max_stake_fraction:
        raise ValueError(
            f"Stake exceeds the configured capital limit of {max_stake_fraction:.1%} "
            "of starting balance."
        )
    if first_signal_index < 0:
        raise ValueError("first_signal_index cannot be negative.")

    ordered = sorted(candles, key=lambda candle: candle.end_time)
    if any(not candle.is_closed for candle in ordered):
        raise ValueError("Backtest input must contain only completed candles.")
    if any(a.symbol != b.symbol or a.timeframe != b.timeframe for a, b in zip(ordered, ordered[1:])):
        raise ValueError("All candles must use the same symbol and timeframe.")
    if any(a.end_time > b.start_time for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Candles overlap; provide a clean chronological series.")

    trades: list[BinaryOptionTrade] = []
    skipped_signals = 0
    balance = starting_balance
    peak_balance = balance
    max_drawdown = 0.0
    index = max(first_signal_index, 0)

    while index + expiry_bars < len(ordered):
        signal_bar = ordered[index]
        result = engine.evaluate(ordered[: index + 1], asset=signal_bar.symbol)
        if result.action not in (BinaryAction.CALL, BinaryAction.PUT):
            skipped_signals += 1
            index += 1
            continue

        entry_index = index + 1
        settlement_index = index + expiry_bars
        entry_bar = ordered[entry_index]
        settlement_bar = ordered[settlement_index]

        # Timing verification: strictly prevent lookahead and ensure settlement is after entry
        if entry_bar.start_time < signal_bar.end_time:
            raise ValueError(
                f"Look-ahead timing violation: entry bar starts at {entry_bar.start_time} "
                f"before signal bar closes at {signal_bar.end_time}."
            )
        if settlement_bar.end_time <= entry_bar.start_time:
            raise ValueError(
                f"Invalid settlement timing: expiry {settlement_bar.end_time} must occur after entry {entry_bar.start_time}."
            )

        entry_price = entry_bar.open
        settlement_price = settlement_bar.close

        if balance < stake:
            break

        # Sizing rule: Stake is strictly fixed and capped. No martingale or progressive risk.
        # Equal settlement is a loss unless the target broker's contract explicitly refunds ties.
        won = (
            settlement_price > entry_price
            if result.action == BinaryAction.CALL
            else settlement_price < entry_price
        )
        net_pnl = stake * payout_rate if won else -stake
        balance += net_pnl
        peak_balance = max(peak_balance, balance)
        max_drawdown = max(max_drawdown, (peak_balance - balance) / peak_balance)
        trades.append(
            BinaryOptionTrade(
                asset=signal_bar.symbol,
                direction=result.action,
                signal_time=signal_bar.end_time.isoformat(),
                entry_time=entry_bar.start_time.isoformat(),
                expiry_time=settlement_bar.end_time.isoformat(),
                entry_price=entry_price,
                settlement_price=settlement_price,
                stake=stake,
                payout_rate=payout_rate,
                score=result.score,
                won=won,
                net_pnl=net_pnl,
            )
        )
        index = settlement_index

    winners = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
    losers = [trade.net_pnl for trade in trades if trade.net_pnl < 0]
    total_trades = len(trades)
    net_pnl = balance - starting_balance
    return BinaryBacktestResult(
        trades=tuple(trades),
        skipped_signals=skipped_signals,
        starting_balance=starting_balance,
        ending_balance=balance,
        net_pnl=net_pnl,
        win_rate=(len(winners) / total_trades) if total_trades else 0.0,
        break_even_win_rate=1.0 / (1.0 + payout_rate),
        profit_factor=(sum(winners) / abs(sum(losers))) if losers else (inf if winners else 0.0),
        expectancy_per_trade=(net_pnl / total_trades) if total_trades else 0.0,
        max_drawdown=max_drawdown,
    )
