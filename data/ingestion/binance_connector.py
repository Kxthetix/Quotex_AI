"""Binance Market Data Connector.

Implements MarketDataInterface using Binance public WebSockets and REST API.
Does not require any API keys or paid subscriptions.
"""

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import aiohttp
import websockets

try:
    from websockets.asyncio.client import ClientConnection as WSConnection
except ImportError:
    WSConnection = Any  # type: ignore

from data.ingestion.market_data_interface import MarketDataInterface
from data.ingestion.models import Candle, MarketEvent
from monitoring.logging import get_logger

logger = get_logger(__name__)


class BinanceWebSocketConnector(MarketDataInterface):
    """Binance public market data connector via WebSocket and REST."""

    def __init__(
        self,
        ws_url: str = "wss://stream.binance.com:9443/ws",
        rest_url: str = "https://api.binance.com",
    ) -> None:
        self._ws_url = ws_url
        self._rest_url = rest_url
        self._ws: WSConnection | None = None
        self._subscribed_streams: set[str] = set()
        self._is_connected = False
        self._reconnect_count = 0
        self._events_received = 0
        self._last_event_time: datetime | None = None
        self._request_id = 0

    @property
    def provider_name(self) -> str:
        return "binance"

    @property
    def is_connected(self) -> bool:
        if not self._is_connected or self._ws is None:
            return False
        if hasattr(self._ws, "state"):
            from websockets.protocol import State

            return self._ws.state == State.OPEN
        return not getattr(self._ws, "closed", True)

    async def connect(self) -> None:
        """Connect to Binance public WebSocket endpoint."""
        logger.info("connecting_to_binance_websocket", url=self._ws_url)
        try:
            self._ws = await websockets.connect(
                self._ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            self._is_connected = True
            logger.info("binance_websocket_connected")

            # Resubscribe if reconnecting
            if self._subscribed_streams:
                await self._send_subscription(list(self._subscribed_streams))
        except Exception as e:
            self._is_connected = False
            logger.error("binance_connection_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Gracefully disconnect from Binance WebSocket."""
        self._is_connected = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning("binance_ws_close_error", error=str(e))
            finally:
                self._ws = None
        logger.info("binance_websocket_disconnected")

    async def subscribe_trades(self, symbols: list[str]) -> None:
        """Subscribe to @trade streams for specified symbols."""
        streams = [f"{s.lower()}@trade" for s in symbols]
        for s in streams:
            self._subscribed_streams.add(s)
        if self.is_connected:
            await self._send_subscription(streams)

    async def subscribe_candles(self, symbols: list[str], timeframe: str = "1m") -> None:
        """Subscribe to @kline streams for specified symbols."""
        streams = [f"{s.lower()}@kline_{timeframe}" for s in symbols]
        for s in streams:
            self._subscribed_streams.add(s)
        if self.is_connected:
            await self._send_subscription(streams)

    async def _send_subscription(self, streams: list[str]) -> None:
        """Send a subscription command frame to Binance WebSocket."""
        if not self.is_connected or not self._ws:
            return
        self._request_id += 1
        payload = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": self._request_id,
        }
        await self._ws.send(json.dumps(payload))
        logger.info("binance_subscription_sent", streams=streams, id=self._request_id)

    async def stream_events(self) -> AsyncIterator[MarketEvent]:
        """Asynchronously yield raw MarketEvent objects parsed from incoming frames."""
        if not self.is_connected or not self._ws:
            raise ConnectionError("Cannot stream events: WebSocket is not connected.")

        while self.is_connected and self._ws:
            try:
                raw_msg = await self._ws.recv()
                self._events_received += 1
                self._last_event_time = datetime.now(UTC)

                data = json.loads(raw_msg)
                # Ignore subscription ack responses
                if "result" in data and "id" in data:
                    continue

                # Multi-stream wrapper handling
                payload = data.get("data", data)
                event_type = payload.get("e", "unknown")
                symbol = str(payload.get("s", "")).upper()

                raw_ts = payload.get("E") or payload.get("T")
                if raw_ts is not None:
                    event_time = datetime.fromtimestamp(float(raw_ts) / 1000.0, tz=UTC)
                else:
                    event_time = datetime.now(UTC)

                yield MarketEvent(
                    source=self.provider_name,
                    symbol=symbol,
                    event_type=event_type,
                    event_time=event_time,
                    receive_time=datetime.now(UTC),
                    payload=payload,
                )
            except json.JSONDecodeError as e:
                logger.warning("binance_malformed_json_frame_ignored", error=str(e))
                continue
            except websockets.exceptions.ConnectionClosed as e:
                self._is_connected = False
                logger.warning("binance_connection_closed", code=e.code, reason=e.reason)
                break
            except Exception as e:
                logger.exception("binance_stream_error", error=str(e))
                break

    async def fetch_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        """Fetch historical klines via Binance public REST API."""
        url = f"{self._rest_url}/api/v3/klines"
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": min(limit, 1000),
        }
        if start:
            params["startTime"] = int(start.timestamp() * 1000)
        if end:
            params["endTime"] = int(end.timestamp() * 1000)

        async with (
            aiohttp.ClientSession() as session,
            session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp,
        ):
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"Failed to fetch Binance historical klines: {resp.status} - {text}"
                )
            data = await resp.json()

        candles: list[Candle] = []
        for k in data:
            # Binance kline format:
            # 0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume, 6: Close time, ...
            start_time = datetime.fromtimestamp(float(k[0]) / 1000.0, tz=UTC)
            end_time = datetime.fromtimestamp(float(k[6]) / 1000.0, tz=UTC)
            candles.append(
                Candle(
                    symbol=symbol.upper(),
                    timeframe=timeframe,
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    start_time=start_time,
                    end_time=end_time,
                    trades_count=int(k[8]),
                    is_closed=True,
                )
            )
        logger.info("historical_candles_fetched", symbol=symbol, count=len(candles))
        return candles

    def get_health(self) -> dict[str, Any]:
        """Return connector operational metrics."""
        return {
            "provider": self.provider_name,
            "is_connected": self.is_connected,
            "subscribed_streams": list(self._subscribed_streams),
            "events_received": self._events_received,
            "last_event_time": self._last_event_time.isoformat() if self._last_event_time else None,
        }
