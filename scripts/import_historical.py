"""CLI tool to fetch, validate, and store historical market data in Parquet and TimescaleDB."""

import argparse
import asyncio
from datetime import UTC, datetime

from configs.config import get_settings
from data.ingestion.binance_connector import BinanceWebSocketConnector
from data.ingestion.historical_importer import HistoricalDataImporter
from monitoring.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def import_historical_data(
    symbol: str,
    timeframe: str = "1m",
    limit: int | None = None,
    start_str: str | None = None,
    end_str: str | None = None,
    force: bool = False,
) -> None:
    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)

    clean_symbol = symbol.upper()

    start_dt = None
    if start_str:
        start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)

    end_dt = None
    if end_str:
        end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=UTC)

    print(f"Initializing HistoricalDataImporter for {clean_symbol} ({timeframe})...")
    connector = BinanceWebSocketConnector(rest_url=settings.rest_base_url)
    importer = HistoricalDataImporter(provider=connector)

    run = await importer.import_candles(
        symbol=clean_symbol,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
        limit=limit,
        force_reimport=force,
    )

    print("\n" + "=" * 55)
    print("HISTORICAL INGESTION RUN SUMMARY")
    print("=" * 55)
    print(f"Run ID:            {run.run_id}")
    print(f"Provider:          {run.provider}")
    print(f"Symbol:            {run.symbol}")
    print(f"Timeframe:         {run.timeframe}")
    print(f"Status:            {run.status.value}")
    print(f"Records Received:  {run.records_received}")
    print(f"Records Accepted:  {run.records_accepted}")
    print(f"Records Rejected:  {run.records_rejected}")
    print(f"Duplicates:        {run.duplicates}")
    if run.actual_start and run.actual_end:
        print(f"Date Range:        {run.actual_start.isoformat()} -> {run.actual_end.isoformat()}")
    if run.errors:
        print(f"Errors ({len(run.errors)}):")
        for err in run.errors[:5]:
            print(f"  - {err}")
    print("=" * 55 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import historical market data with audit tracking"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading symbol (e.g. BTCUSDT)"
    )
    parser.add_argument("--timeframe", type=str, default="1m", help="Candle timeframe (e.g. 1m, 1h, 1d)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of candles to fetch (defaults to 500 if no start date, or unlimited if start date given)",
    )
    parser.add_argument(
        "--start", type=str, default=None, help="Start ISO timestamp (YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--end", type=str, default=None, help="End ISO timestamp (YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-import even if already completed"
    )

    args = parser.parse_args()
    asyncio.run(
        import_historical_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.limit,
            start_str=args.start,
            end_str=args.end,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
