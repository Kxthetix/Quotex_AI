"""Unit tests for EventNormalizer."""

from datetime import UTC, datetime

from data.ingestion.models import MarketEvent
from data.normalization.normalizer import EventNormalizer


def test_normalize_binance_trade():
    normalizer = EventNormalizer(provider="binance")

    raw_event = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="trade",
        event_time=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        payload={
            "e": "trade",
            "E": 1767268800000,
            "s": "BTCUSDT",
            "t": 1234567,
            "p": "65123.45",
            "q": "0.15000000",
            "T": 1767268800000,
            "m": True,  # buyer is maker -> seller was taker -> "sell"
        },
    )

    trade = normalizer.normalize_trade(raw_event)
    assert trade.symbol == "BTCUSDT"
    assert trade.source == "binance"
    assert trade.price == 65123.45
    assert trade.quantity == 0.15
    assert trade.side == "sell"
    assert trade.trade_id == "1234567"


def test_normalize_binance_kline():
    normalizer = EventNormalizer(provider="binance")

    raw_event = MarketEvent(
        source="binance",
        symbol="ETHUSDT",
        event_type="kline",
        event_time=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        payload={
            "e": "kline",
            "E": 1767268800000,
            "s": "ETHUSDT",
            "k": {
                "t": 1767268800000,
                "T": 1767268859999,
                "s": "ETHUSDT",
                "i": "1m",
                "o": "3450.00",
                "c": "3455.50",
                "h": "3460.00",
                "l": "3448.00",
                "v": "120.500",
                "n": 450,
                "x": True,
            },
        },
    )

    candle = normalizer.normalize_kline(raw_event)
    assert candle.symbol == "ETHUSDT"
    assert candle.timeframe == "1m"
    assert candle.open == 3450.00
    assert candle.high == 3460.00
    assert candle.low == 3448.00
    assert candle.close == 3455.50
    assert candle.volume == 120.5
    assert candle.trades_count == 450
    assert candle.is_closed is True
