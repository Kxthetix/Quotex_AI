"""CLI tool to fetch, validate, and store historical market data in Parquet and TimescaleDB."""

import argparse
import asyncio
from datetime import UTC, datetime

from configs.config import get_settings
from data.ingestion.binance_connector import BinanceWebSocketConnector
from data.storage.parquet_store import save_candles_to_parquet
from data.validation.engine import DataQualityEngine
from monitoring.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def import_historical_data(symbol: str, timeframe: str = "1m", limit: int = 500) -> None:
    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)

    clean_symbol = symbol.upper()
    print(f"Fetching {limit} historical {timeframe} candles for {clean_symbol}...")

    connector = BinanceWebSocketConnector(rest_url=settings.rest_base_url)
    candles = await connector.fetch_historical_candles(
        symbol=clean_symbol,
        timeframe=timeframe,
        limit=limit,
    )

    if not candles:
        print("No candles returned.")
        return

    print(f"Received {len(candles)} candles. Running data quality engine validation...")
    quality_engine = DataQualityEngine()

    valid_candles = []
    for c in candles:
        res = quality_engine.validate_candle(c)
        if res.is_valid:
            valid_candles.append(c)
        else:
            print(f"Candle at {c.start_time} had issues: {[i.message for i in res.issues]}")

    print(f"Validation complete: {len(valid_candles)}/{len(candles)} valid candles.")

    # Save to Parquet
    today_str = datetime.now(UTC).strftime("%Y%m%d")
    parquet_path = (
        settings.parquet_storage_path
        / clean_symbol
        / f"{clean_symbol}_{timeframe}_{today_str}.parquet"
    )
    save_candles_to_parquet(valid_candles, parquet_path)
    print(f"Dataset successfully persisted to Parquet: {parquet_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import historical market data")
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading symbol (e.g. BTCUSDT)"
    )
    parser.add_argument("--timeframe", type=str, default="1m", help="Candle timeframe (e.g. 1m)")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles to fetch")

    args = parser.parse_args()
    asyncio.run(
        import_historical_data(symbol=args.symbol, timeframe=args.timeframe, limit=args.limit)
    )


if __name__ == "__main__":
    main()
