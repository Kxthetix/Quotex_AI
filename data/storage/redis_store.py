"""Redis and in-memory fast market state store."""

import asyncio
import contextlib
from typing import Any

import redis.asyncio as aioredis

from configs.config import get_settings
from data.ingestion.models import Candle, NormalizedTrade
from monitoring.logging import get_logger

logger = get_logger(__name__)


class MarketStateStore:
    """Fast market state store with Redis backing and in-memory fallback."""

    def __init__(
        self,
        redis_host: str | None = None,
        redis_port: int | None = None,
        redis_db: int | None = None,
        redis_password: str | None = None,
    ) -> None:
        settings = get_settings()
        self.redis_host = redis_host or settings.redis_host
        self.redis_port = redis_port or settings.redis_port
        self.redis_db = redis_db or settings.redis_db
        self.redis_password = redis_password or settings.redis_password

        # Local in-memory cache (guarantees operation even if Redis container is not running)
        self._memory_trades: dict[str, NormalizedTrade] = {}
        self._memory_candles: dict[str, Candle] = {}
        self._memory_metadata: dict[str, Any] = {}

        self._redis_client: aioredis.Redis | None = None
        self._is_redis_available = False

    async def connect(self) -> None:
        """Attempt connection to Redis with short timeout, falling back to memory."""
        try:
            self._redis_client = aioredis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            )
            await asyncio.wait_for(self._redis_client.ping(), timeout=0.3)
            self._is_redis_available = True
            logger.info("redis_connected_successfully", host=self.redis_host, port=self.redis_port)
        except Exception as e:
            self._is_redis_available = False
            if self._redis_client:
                with contextlib.suppress(Exception):
                    await self._redis_client.aclose()
                self._redis_client = None
            logger.info("redis_offline_using_in_memory_state", info=str(e))

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.aclose()  # type: ignore[attr-defined]

    async def set_latest_trade(self, trade: NormalizedTrade) -> None:
        """Store the most recent trade tick for a symbol."""
        symbol = trade.symbol.upper()
        self._memory_trades[symbol] = trade

        if self._is_redis_available and self._redis_client:
            try:
                key = f"market:trade:{symbol}"
                await self._redis_client.set(key, trade.model_dump_json(), ex=300)
            except Exception as e:
                logger.warning("redis_set_trade_failed", error=str(e))
                self._is_redis_available = False

    async def get_latest_trade(self, symbol: str) -> NormalizedTrade | None:
        """Get the most recent trade tick."""
        symbol = symbol.upper()
        if symbol in self._memory_trades:
            return self._memory_trades[symbol]

        if self._is_redis_available and self._redis_client:
            try:
                raw = await self._redis_client.get(f"market:trade:{symbol}")
                if raw:
                    return NormalizedTrade.model_validate_json(raw)
            except Exception:
                pass
        return None

    async def set_latest_candle(self, candle: Candle) -> None:
        """Store the most recent candle for a symbol and timeframe."""
        key = f"{candle.symbol.upper()}:{candle.timeframe}"
        self._memory_candles[key] = candle

        if self._is_redis_available and self._redis_client:
            try:
                redis_key = f"market:candle:{candle.symbol.upper()}:{candle.timeframe}"
                await self._redis_client.set(redis_key, candle.model_dump_json(), ex=600)
            except Exception as e:
                logger.warning("redis_set_candle_failed", error=str(e))
                self._is_redis_available = False

    async def get_latest_candle(self, symbol: str, timeframe: str = "1m") -> Candle | None:
        """Get the most recent candle."""
        key = f"{symbol.upper()}:{timeframe}"
        if key in self._memory_candles:
            return self._memory_candles[key]

        if self._is_redis_available and self._redis_client:
            try:
                raw = await self._redis_client.get(f"market:candle:{symbol.upper()}:{timeframe}")
                if raw:
                    return Candle.model_validate_json(raw)
            except Exception:
                pass
        return None

    def get_health(self) -> dict[str, Any]:
        """Return cache health status."""
        return {
            "redis_available": self._is_redis_available,
            "cached_symbols_count": len(self._memory_trades),
            "cached_candles_count": len(self._memory_candles),
        }
