"""Unit tests for Event Normalizer (Test Group C).

Verifies provider-neutral normalization:
- Binance trade / aggTrade payloads -> Canonical NormalizedTrade
- Binance kline payloads -> Canonical Candle
- Millisecond timestamp conversion into timezone-aware UTC datetime
- Numeric float conversions with precision preservation
- Symbol normalization
"""

from datetime import UTC, datetime

from data.ingestion.models import MarketEvent
from data.normalization.normalizer import EventNormalizer


def test_normalize_agg_trade():
    """Verify Binance aggTrade payload converts into canonical NormalizedTrade."""
    normalizer = EventNormalizer(provider="binance")

    now = datetime.now(UTC)
    raw_event = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="aggTrade",
        event_time=now,
        receive_time=now,
        payload={
            "e": "aggTrade",
            "E": 1704110400000,
            "s": "btcusdt",
            "a": 987654321,
            "p": "67450.25",
            "q": "1.45000000",
            "f": 1000,
            "l": 1005,
            "T": 1704110400000,
            "m": True,  # buyer is maker -> sell side
        },
    )

    trade = normalizer.normalize_trade(raw_event)
    assert trade.trade_id == "987654321"
    assert trade.source == "binance"
    assert trade.symbol == "BTCUSDT"
    assert trade.price == 67450.25
    assert trade.quantity == 1.45
    assert trade.side == "sell"
    assert trade.timestamp.tzinfo == UTC
    assert trade.timestamp.year == 2024


def test_normalize_individual_trade():
    """Verify Binance standard trade payload converts with correct buyer/maker side."""
    normalizer = EventNormalizer(provider="binance")

    now = datetime.now(UTC)
    raw_event = MarketEvent(
        source="binance",
        symbol="ETHUSDT",
        event_type="trade",
        event_time=now,
        receive_time=now,
        payload={
            "e": "trade",
            "E": 1704110400000,
            "s": "ethusdt",
            "t": 11223344,
            "p": "3450.75",
            "q": "10.00000000",
            "T": 1704110400000,
            "m": False,  # buyer is taker -> buy side
        },
    )

    trade = normalizer.normalize_trade(raw_event)
    assert trade.trade_id == "11223344"
    assert trade.symbol == "ETHUSDT"
    assert trade.price == 3450.75
    assert trade.quantity == 10.0
    assert trade.side == "buy"


def test_normalize_kline():
    """Verify Binance kline payload converts into canonical Candle."""
    normalizer = EventNormalizer(provider="binance")

    now = datetime.now(UTC)
    raw_event = MarketEvent(
        source="binance",
        symbol="BTCUSDT",
        event_type="kline",
        event_time=now,
        receive_time=now,
        payload={
            "e": "kline",
            "E": 1704110460000,
            "s": "btcusdt",
            "k": {
                "t": 1704110400000,
                "T": 1704110459999,
                "s": "BTCUSDT",
                "i": "1m",
                "o": "65000.00",
                "c": "65200.00",
                "h": "65250.00",
                "l": "64950.00",
                "v": "100.5",
                "n": 500,
                "x": True,  # closed
            },
        },
    )

    candle = normalizer.normalize_kline(raw_event)
    assert candle.symbol == "BTCUSDT"
    assert candle.timeframe == "1m"
    assert candle.open == 65000.00
    assert candle.high == 65250.00
    assert candle.low == 64950.00
    assert candle.close == 65200.00
    assert candle.volume == 100.5
    assert candle.trades_count == 500
    assert candle.is_closed is True
    assert candle.start_time.tzinfo == UTC
    assert candle.end_time.tzinfo == UTC
