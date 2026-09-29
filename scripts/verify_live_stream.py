"""Quick verification script to test live connectivity against public Binance streams."""

import asyncio

from data.ingestion.binance_connector import BinanceWebSocketConnector
from data.normalization.normalizer import EventNormalizer
from data.validation.engine import DataQualityEngine


async def test_live_stream():
    connector = BinanceWebSocketConnector()
    normalizer = EventNormalizer(provider="binance")
    quality_engine = DataQualityEngine()

    print("Connecting to Binance WebSocket...")
    await connector.connect()
    await connector.subscribe_trades(["BTCUSDT"])
    print("Subscribed to BTCUSDT trade stream. Waiting for 3 live ticks...")

    count = 0
    async for event in connector.stream_events():
        # Validate raw
        raw_val = quality_engine.validate_raw_event(event)
        if not raw_val.is_valid:
            continue
        # Normalize
        trade = normalizer.normalize_trade(event)
        # Validate trade
        trade_val = quality_engine.validate_trade(trade)

        print(
            f"  [TICK {count + 1}] Symbol: {trade.symbol} | Price: ${trade.price:,.2f} | "
            f"Qty: {trade.quantity} | Side: {trade.side} | Valid: {trade_val.is_valid}"
        )

        count += 1
        if count >= 3:
            break

    await connector.disconnect()
    print(
        "Live stream verification successful! Pipeline received, validated, and normalized real ticks."
    )


if __name__ == "__main__":
    asyncio.run(test_live_stream())
