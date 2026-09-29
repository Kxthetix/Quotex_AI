"""Unit tests for WebSocketMarketDataProvider / BinanceWebSocketConnector (Test Group A).

Verifies connection lifecycle, subscriptions, message parsing, malformed payload resilience,
and health metrics isolation.
"""

import json
from unittest.mock import AsyncMock

import pytest

from data.ingestion.binance_connector import BinanceWebSocketConnector
from data.ingestion.market_data_interface import (
    HistoricalMarketDataProvider,
    MarketDataInterface,
    WebSocketMarketDataProvider,
)


@pytest.mark.asyncio
async def test_provider_interface_conformance():
    """Verify BinanceWebSocketConnector conforms to both WebSocket and Historical interfaces."""
    connector = BinanceWebSocketConnector(
        ws_url="wss://stream.binance.com:9443/ws",
        rest_url="https://api.binance.com",
    )
    assert isinstance(connector, WebSocketMarketDataProvider)
    assert isinstance(connector, HistoricalMarketDataProvider)
    assert isinstance(connector, MarketDataInterface)
    assert connector.provider_name == "binance"
    assert not connector.is_connected


@pytest.mark.asyncio
async def test_provider_subscriptions():
    """Verify topic generation and subscription messaging."""
    from websockets.protocol import State

    connector = BinanceWebSocketConnector(
        ws_url="wss://stream.binance.com:9443/ws",
        rest_url="https://api.binance.com",
    )

    # Mock the underlying WebSocket
    mock_ws = AsyncMock()
    mock_ws.state = State.OPEN
    connector._is_connected = True
    connector._ws = mock_ws

    await connector.subscribe_trades(["BTCUSDT", "ETHUSDT"])
    assert "btcusdt@trade" in connector._subscribed_streams
    assert "ethusdt@trade" in connector._subscribed_streams

    await connector.subscribe_candles(["BTCUSDT"], timeframe="1m")
    assert "btcusdt@kline_1m" in connector._subscribed_streams

    # Check that subscription payload was sent over ws
    assert mock_ws.send.called
    sent_payload = json.loads(mock_ws.send.call_args[0][0])
    assert sent_payload["method"] == "SUBSCRIBE"
    assert "btcusdt@kline_1m" in sent_payload["params"]


@pytest.mark.asyncio
async def test_provider_stream_events_parsing():
    """Verify stream_events correctly parses raw messages into canonical MarketEvents."""
    from websockets.protocol import State

    connector = BinanceWebSocketConnector(
        ws_url="wss://stream.binance.com:9443/ws",
        rest_url="https://api.binance.com",
    )

    # Simulated Binance messages
    trade_raw = json.dumps(
        {
            "e": "aggTrade",
            "E": 1704110400000,
            "s": "BTCUSDT",
            "a": 12345,
            "p": "65000.50",
            "q": "0.15000000",
            "f": 100,
            "l": 105,
            "T": 1704110400000,
            "m": True,
        }
    )
    kline_raw = json.dumps(
        {
            "e": "kline",
            "E": 1704110460000,
            "s": "BTCUSDT",
            "k": {
                "t": 1704110400000,
                "T": 1704110459999,
                "s": "BTCUSDT",
                "i": "1m",
                "o": "65000.0",
                "c": "65050.0",
                "h": "65100.0",
                "l": "64990.0",
                "v": "12.5",
                "n": 42,
                "x": True,
            },
        }
    )
    # Malformed message (should be gracefully handled/ignored without crashing)
    malformed_raw = "THIS IS NOT JSON"

    messages = [trade_raw, malformed_raw, kline_raw]

    mock_ws = AsyncMock()
    mock_ws.state = State.OPEN
    mock_ws.recv = AsyncMock(side_effect=messages)
    connector._is_connected = True
    connector._ws = mock_ws

    events = []
    async for event in connector.stream_events():
        events.append(event)
        if len(events) == 2:
            break

    assert len(events) == 2
    # Check trade event
    e_trade = events[0]
    assert e_trade.source == "binance"
    assert e_trade.symbol == "BTCUSDT"
    assert e_trade.event_type == "aggTrade"
    assert e_trade.payload["p"] == "65000.50"

    # Check kline event
    e_kline = events[1]
    assert e_kline.source == "binance"
    assert e_kline.symbol == "BTCUSDT"
    assert e_kline.event_type == "kline"
    assert e_kline.payload["k"]["c"] == "65050.0"


@pytest.mark.asyncio
async def test_provider_health_metrics():
    """Verify provider health dictionary reports status and counters."""
    connector = BinanceWebSocketConnector(
        ws_url="wss://stream.binance.com:9443/ws",
        rest_url="https://api.binance.com",
    )
    health = connector.get_health()
    assert health["provider"] == "binance"
    assert health["is_connected"] is False
    assert health["events_received"] == 0
