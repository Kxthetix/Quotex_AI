"""Event normalization engine.

Converts provider-specific raw market events into canonical domain models
(NormalizedTrade and Candle) following a strictly uniform schema.
"""

from datetime import UTC, datetime
from typing import Any

from data.ingestion.models import Candle, MarketEvent, NormalizedTrade
from monitoring.logging import get_logger

logger = get_logger(__name__)


class EventNormalizer:
    """Enterprise normalizer mapping raw provider payloads to uniform entities."""

    def __init__(self, provider: str = "binance") -> None:
        self.provider = provider.lower()

    def normalize_trade(self, event: MarketEvent) -> NormalizedTrade:
        """Normalize an incoming trade MarketEvent."""
        payload = event.payload

        if self.provider == "binance":
            return self._normalize_binance_trade(event.symbol, payload)
        else:
            # Generic fallback normalizer
            return self._normalize_generic_trade(event.symbol, payload)

    def normalize_kline(self, event: MarketEvent) -> Candle:
        """Normalize an incoming kline/candlestick MarketEvent."""
        payload = event.payload

        if self.provider == "binance":
            return self._normalize_binance_kline(event.symbol, payload)
        else:
            return self._normalize_generic_kline(event.symbol, payload)

    def _normalize_binance_trade(self, symbol: str, payload: dict[str, Any]) -> NormalizedTrade:
        """Parse Binance @trade or @aggTrade WebSocket message."""
        # Binance trade message fields:
        # e: event type, E: event time, s: symbol, t/a: trade id, p: price, q: quantity, T: trade time, m: buyer is maker
        trade_id = str(payload.get("t") or payload.get("a") or payload.get("trade_id") or "")
        price = float(payload.get("p") or payload.get("price") or 0.0)
        quantity = float(payload.get("q") or payload.get("quantity") or 0.0)

        # Trade timestamp in ms
        raw_ts = payload.get("T") or payload.get("E")
        if raw_ts is not None:
            ts = datetime.fromtimestamp(float(raw_ts) / 1000.0, tz=UTC)
        else:
            ts = datetime.now(UTC)

        # In Binance: "m" is True if buyer is maker (meaning the taker was a seller -> Sell)
        is_buyer_maker = payload.get("m", False)
        side = "sell" if is_buyer_maker else "buy"

        return NormalizedTrade(
            trade_id=trade_id,
            source="binance",
            symbol=symbol,
            price=price,
            quantity=quantity,
            side=side,
            timestamp=ts,
        )

    def _normalize_binance_kline(self, symbol: str, payload: dict[str, Any]) -> Candle:
        """Parse Binance @kline_<interval> WebSocket message."""
        k = payload.get("k", payload)
        open_price = float(k.get("o") or k.get("open") or 0.0)
        high_price = float(k.get("h") or k.get("high") or 0.0)
        low_price = float(k.get("l") or k.get("low") or 0.0)
        close_price = float(k.get("c") or k.get("close") or 0.0)
        volume = float(k.get("v") or k.get("volume") or 0.0)
        trades_count = int(k.get("n") or k.get("trades") or 0)
        is_closed = bool(k.get("x", True))

        start_ts = k.get("t") or k.get("start_time")
        end_ts = k.get("T") or k.get("end_time")

        start_time = (
            datetime.fromtimestamp(float(start_ts) / 1000.0, tz=UTC)
            if start_ts is not None
            else datetime.now(UTC)
        )
        end_time = (
            datetime.fromtimestamp(float(end_ts) / 1000.0, tz=UTC)
            if end_ts is not None
            else datetime.now(UTC)
        )
        interval = str(k.get("i") or "1m")

        return Candle(
            symbol=symbol,
            timeframe=interval,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            start_time=start_time,
            end_time=end_time,
            trades_count=trades_count,
            is_closed=is_closed,
        )

    def _normalize_generic_trade(self, symbol: str, payload: dict[str, Any]) -> NormalizedTrade:
        """Fallback generic normalizer."""
        return NormalizedTrade(
            trade_id=str(payload.get("trade_id", "")),
            source=self.provider,
            symbol=symbol,
            price=float(payload.get("price", 0.0)),
            quantity=float(payload.get("quantity", 0.0)),
            side=str(payload.get("side", "unknown")).lower(),
            timestamp=datetime.now(UTC),
        )

    def _normalize_generic_kline(self, symbol: str, payload: dict[str, Any]) -> Candle:
        """Fallback generic kline normalizer."""
        return Candle(
            symbol=symbol,
            timeframe=str(payload.get("timeframe", "1m")),
            open=float(payload.get("open", 0.0)),
            high=float(payload.get("high", 0.0)),
            low=float(payload.get("low", 0.0)),
            close=float(payload.get("close", 0.0)),
            volume=float(payload.get("volume", 0.0)),
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            trades_count=int(payload.get("trades_count", 0)),
            is_closed=bool(payload.get("is_closed", True)),
        )
