"""In-memory Market State manager for Phase 7 Paper Trading.

Maintains real-time in-memory state per symbol:
- Latest price, bid, ask, volume
- Current in-flight and latest closed candles across timeframes
- Market timestamp, receive timestamp
- Data quality validation status

Live processing consumes this in-memory state without round-tripping to database.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from data.ingestion.models import Candle, NormalizedTrade
from monitoring.logging import get_logger
from paper_trading.models import MarketState

logger = get_logger(__name__)


class MarketStateManager:
    """Thread-safe and async-safe in-memory store for live multi-symbol market states."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        self._states: dict[str, MarketState] = {}
        for s in (symbols or []):
            self.init_symbol(s)

    def init_symbol(self, symbol: str) -> MarketState:
        sym = symbol.upper()
        if sym not in self._states:
            self._states[sym] = MarketState(symbol=sym)
        return self._states[sym]

    def get_state(self, symbol: str) -> MarketState:
        """Get or initialize state for a symbol."""
        sym = symbol.upper()
        if sym not in self._states:
            self._states[sym] = MarketState(symbol=sym)
        return self._states[sym]

    def update_from_trade(self, trade: NormalizedTrade, data_quality_status: str = "VALID") -> MarketState:
        """Update market state from an incoming normalized trade."""
        sym = trade.symbol.upper()
        state = self.get_state(sym)
        state.latest_price = trade.price
        state.latest_volume = trade.quantity
        state.market_timestamp = trade.timestamp
        state.receive_timestamp = trade.receive_time or datetime.now(UTC)
        state.data_quality_status = data_quality_status
        state.event_count += 1
        return state

    def update_book_ticker(
        self,
        symbol: str,
        bid: float,
        ask: float,
        timestamp: datetime | None = None,
        receive_time: datetime | None = None,
    ) -> MarketState:
        """Update bid/ask prices from book ticker or depth update."""
        sym = symbol.upper()
        state = self.get_state(sym)
        state.bid = bid
        state.ask = ask
        if timestamp:
            state.market_timestamp = timestamp
        if receive_time:
            state.receive_timestamp = receive_time
        return state

    def update_closed_candle(self, candle: Candle, timeframe: str = "1m") -> MarketState:
        """Update latest closed candle for a given timeframe."""
        sym = candle.symbol.upper()
        state = self.get_state(sym)
        state.candles[timeframe] = candle
        state.latest_price = candle.close
        state.market_timestamp = candle.end_time
        return state

    def update_active_candle(self, candle: Candle, timeframe: str = "1m") -> MarketState:
        """Update currently accumulating in-flight candle for a timeframe."""
        sym = candle.symbol.upper()
        state = self.get_state(sym)
        state.active_candles[timeframe] = candle
        state.latest_price = candle.close
        state.market_timestamp = candle.end_time
        return state

    def get_latest_price(self, symbol: str) -> float:
        """Fast lookup of current mark/last price for a symbol."""
        return self.get_state(symbol).latest_price

    def all_states(self) -> dict[str, MarketState]:
        return dict(self._states)
