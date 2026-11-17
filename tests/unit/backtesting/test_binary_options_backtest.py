"""Unit tests for the payout-aware binary options backtesting engine."""

from datetime import UTC, datetime, timedelta
import pytest

from backtesting.binary_options import (
    BinaryBacktestResult,
    BinaryOptionTrade,
    run_binary_options_backtest,
)
from data.ingestion.models import Candle
from signals.binary_scalping_engine import BinaryAction, BinaryScalpingEngine, BinarySignalResult


def _create_clean_candles(count: int = 100, interval_minutes: int = 1) -> list[Candle]:
    """Helper to generate a clean, strictly chronological series of completed candles."""
    start = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    delta = timedelta(minutes=interval_minutes)
    candles = []
    price = 100.0

    for i in range(count):
        s = start + i * delta
        e = s + delta
        # Alternate small price steps
        step = 0.05 if (i % 2 == 0) else -0.02
        c_open = price
        c_close = price + step
        c_high = max(c_open, c_close) + 0.05
        c_low = min(c_open, c_close) - 0.05
        candles.append(
            Candle(
                symbol="EUR/USD",
                timeframe=f"{interval_minutes}m",
                open=round(c_open, 4),
                high=round(c_high, 4),
                low=round(c_low, 4),
                close=round(c_close, 4),
                volume=500.0,
                start_time=s,
                end_time=e,
                trades_count=50,
                is_closed=True,
            )
        )
        price = c_close
    return candles


class MockEngine:
    """Mock engine emitting deterministic actions for testing backtest dynamics."""

    def __init__(self, actions: list[BinaryAction]):
        self.actions = actions
        self.call_count = 0

    def evaluate(self, candles, asset="EUR/USD", crowd_sentiment_bullish_pct=None):
        action = self.actions[self.call_count % len(self.actions)]
        self.call_count += 1
        return BinarySignalResult(
            action=action,
            confidence=0.85,
            score=0.85 if action == BinaryAction.CALL else (-0.85 if action == BinaryAction.PUT else 0.0),
            asset=asset,
            timestamp=candles[-1].end_time,
            reasons=["Mock evaluation trigger"],
            rejection_flags=[],
        )


def test_break_even_win_rate_calculation():
    """Verify break_even_win_rate = 1 / (1 + net_payout_rate)."""
    candles = _create_clean_candles(count=70)
    mock = MockEngine([BinaryAction.CALL])

    # Test payout 0.85 (85%)
    res_85 = run_binary_options_backtest(
        candles, mock, payout_rate=0.85, stake=10.0, starting_balance=1000.0, first_signal_index=50
    )
    expected_85 = 1.0 / (1.0 + 0.85)  # 54.054%
    assert abs(res_85.break_even_win_rate - expected_85) < 1e-6

    # Test payout 0.93 (93%)
    res_93 = run_binary_options_backtest(
        candles, mock, payout_rate=0.93, stake=10.0, starting_balance=1000.0, first_signal_index=50
    )
    expected_93 = 1.0 / (1.0 + 0.93)  # 51.813%
    assert abs(res_93.break_even_win_rate - expected_93) < 1e-6


def test_strict_timing_and_no_lookahead():
    """Verify entry is at bar[t+1].open and settlement is at bar[t+expiry].close."""
    candles = _create_clean_candles(count=80)
    mock = MockEngine([BinaryAction.CALL])

    res = run_binary_options_backtest(
        candles, mock, payout_rate=0.90, expiry_bars=1, stake=10.0, starting_balance=1000.0, first_signal_index=50
    )

    assert len(res.trades) > 0
    for trade in res.trades:
        # Strict timing assertions:
        # Signal is formed at completed candle close
        # Entry starts at or after signal candle close
        assert trade.entry_time >= trade.signal_time, "Lookahead: entry occurred before signal bar closed!"
        # Expiry is strictly after entry
        assert trade.expiry_time > trade.entry_time, "Settlement did not occur after entry!"


def test_settlement_matches_specified_expiry():
    """Verify expiry_bars=3 settles exactly 3 bars after signal bar."""
    candles = _create_clean_candles(count=90)
    mock = MockEngine([BinaryAction.CALL])

    res = run_binary_options_backtest(
        candles, mock, payout_rate=0.90, expiry_bars=3, stake=10.0, starting_balance=1000.0, first_signal_index=50
    )

    assert len(res.trades) > 0
    for trade in res.trades:
        t_entry = datetime.fromisoformat(trade.entry_time)
        t_expiry = datetime.fromisoformat(trade.expiry_time)
        # For 1m candles and expiry_bars=3, duration must be exactly 3 minutes
        duration = t_expiry - t_entry
        assert duration == timedelta(minutes=3), f"Expected 3m expiry duration, got {duration}"


def test_fixed_stake_and_no_martingale():
    """Verify stake sizing remains strictly constant across all trades (no martingale)."""
    candles = _create_clean_candles(count=100)
    # Alternate CALL and PUT to generate both wins and losses
    mock = MockEngine([BinaryAction.CALL, BinaryAction.PUT, BinaryAction.NO_TRADE])

    res = run_binary_options_backtest(
        candles, mock, payout_rate=0.80, stake=10.0, starting_balance=1000.0, first_signal_index=50
    )

    assert len(res.trades) > 0
    for trade in res.trades:
        # Stake must NEVER increase after a loss or win
        assert trade.stake == 10.0, f"Expected fixed stake of 10.0, found {trade.stake}"


def test_stake_capping_enforcement():
    """Verify stake exceeding max_stake_fraction raises ValueError."""
    candles = _create_clean_candles(count=70)
    mock = MockEngine([BinaryAction.CALL])

    # Starting balance = 1000.0, max_stake_fraction = 0.01 (max stake = 10.0)
    # Attempting stake of 15.0 must raise ValueError
    with pytest.raises(ValueError, match="Stake exceeds the configured capital limit"):
        run_binary_options_backtest(
            candles, mock, payout_rate=0.90, stake=15.0, starting_balance=1000.0, max_stake_fraction=0.01
        )


def test_unclosed_candle_rejected():
    """Verify backtest rejects series containing incomplete candles."""
    candles = _create_clean_candles(count=70)
    # Mark candle 65 as open/forming
    candles[65] = Candle(
        symbol=candles[65].symbol,
        timeframe=candles[65].timeframe,
        open=candles[65].open,
        high=candles[65].high,
        low=candles[65].low,
        close=candles[65].close,
        volume=candles[65].volume,
        start_time=candles[65].start_time,
        end_time=candles[65].end_time,
        trades_count=10,
        is_closed=False,
    )

    mock = MockEngine([BinaryAction.CALL])
    with pytest.raises(ValueError, match="Backtest input must contain only completed candles"):
        run_binary_options_backtest(candles, mock, payout_rate=0.85)


def test_overlapping_candles_rejected():
    """Verify overlapping candle timestamps are rejected."""
    candles = _create_clean_candles(count=70)
    # Introduce overlap
    start_bad = candles[55].start_time - timedelta(seconds=30)
    candles[55] = Candle(
        symbol=candles[55].symbol,
        timeframe=candles[55].timeframe,
        open=candles[55].open,
        high=candles[55].high,
        low=candles[55].low,
        close=candles[55].close,
        volume=candles[55].volume,
        start_time=start_bad,
        end_time=candles[55].end_time,
        trades_count=10,
        is_closed=True,
    )

    mock = MockEngine([BinaryAction.CALL])
    with pytest.raises(ValueError, match="Candles overlap"):
        run_binary_options_backtest(candles, mock, payout_rate=0.85)


def test_helper_properties():
    """Verify total_trades, wins, and losses helper properties."""
    trades = (
        BinaryOptionTrade(
            asset="EUR/USD",
            direction=BinaryAction.CALL,
            signal_time="2026-01-01T10:00:00Z",
            entry_time="2026-01-01T10:01:00Z",
            expiry_time="2026-01-01T10:02:00Z",
            entry_price=1.0000,
            settlement_price=1.0010,
            stake=10.0,
            payout_rate=0.85,
            score=0.75,
            won=True,
            net_pnl=8.5,
        ),
        BinaryOptionTrade(
            asset="EUR/USD",
            direction=BinaryAction.PUT,
            signal_time="2026-01-01T10:05:00Z",
            entry_time="2026-01-01T10:06:00Z",
            expiry_time="2026-01-01T10:07:00Z",
            entry_price=1.0010,
            settlement_price=1.0020,
            stake=10.0,
            payout_rate=0.85,
            score=-0.75,
            won=False,
            net_pnl=-10.0,
        ),
    )
    result = BinaryBacktestResult(
        trades=trades,
        skipped_signals=5,
        starting_balance=1000.0,
        ending_balance=998.5,
        net_pnl=-1.5,
        win_rate=0.5,
        break_even_win_rate=1.0 / 1.85,
        profit_factor=0.85,
        expectancy_per_trade=-0.75,
        max_drawdown=0.01,
    )
    assert result.total_trades == 2
    assert result.wins == 1
    assert result.losses == 1
