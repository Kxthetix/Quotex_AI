"""Executable runner for live market data ingestion pipeline."""

import asyncio
import contextlib
import signal
import sys

from configs.config import get_settings
from data.ingestion.pipeline import IngestionPipeline
from monitoring.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)

    print("=" * 60)
    print("TRADING INTELLIGENCE SYSTEM — LIVE DATA PIPELINE (V1)")
    print(f"Provider: {settings.data_provider}")
    print(f"Symbols:  {', '.join(settings.market_symbols)}")
    print(f"WS URL:   {settings.ws_base_url}")
    print("=" * 60)

    pipeline = IngestionPipeline(settings=settings)

    # Attach live candle reporter
    pipeline.on_candle_closed(
        lambda c: print(
            f"[{c.end_time.strftime('%H:%M:%S')}] 1m CANDLE CLOSED: {c.symbol} | "
            f"O={c.open:.2f} H={c.high:.2f} L={c.low:.2f} C={c.close:.2f} "
            f"V={c.volume:.4f} T={c.trades_count}"
        )
    )

    # Start pipeline
    await pipeline.start()

    stop_event = asyncio.Event()

    def handle_exit(*args):
        print("\nStopping live ingestion pipeline gracefully...")
        stop_event.set()

    # Handle Ctrl+C on Windows
    with contextlib.suppress(Exception):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, handle_exit)

    # Telemetry loop
    try:
        while not stop_event.is_set():
            await asyncio.sleep(10.0)
            health = pipeline.get_health()
            metrics = health.latency
            print(
                f"[TELEMETRY] Status: {health.status} | "
                f"Latency: p50={metrics.p50_ms:.2f}ms, p95={metrics.p95_ms:.2f}ms, p99={metrics.p99_ms:.2f}ms "
                f"({metrics.sample_count} samples) | Breaker Tripped: {health.breaker_tripped}"
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nShutdown requested.")
    finally:
        await pipeline.stop()
        print("Pipeline safely stopped. Exiting.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
