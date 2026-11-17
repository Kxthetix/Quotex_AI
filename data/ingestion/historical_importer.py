"""Historical Market Data Importer and Backfill Manager.

Implements the audited historical ingestion flow:
Historical Source -> Historical Downloader -> Raw Historical Storage -> Validator -> Normalizer -> Clean Historical Storage.
Guarantees idempotent imports, prevents duplicate records, and maintains an IngestionRun audit ledger.
"""

from datetime import UTC, datetime

from configs.config import Settings, get_settings
from data.ingestion.market_data_interface import HistoricalMarketDataProvider
from data.ingestion.models import Candle, IngestionRun, IngestionRunStatus
from data.storage.parquet_store import save_candles_to_parquet
from data.storage.postgres import AsyncPersistenceWorker, DatabaseManager, IngestionRunRepository
from data.validation.engine import DataQualityEngine
from data.validation.missing_data import MissingDataDetector
from monitoring.logging import get_logger

logger = get_logger(__name__)


class HistoricalDataImporter:
    """Enterprise historical market data importer with run tracking and idempotency."""

    def __init__(
        self,
        provider: HistoricalMarketDataProvider,
        settings: Settings | None = None,
        quality_engine: DataQualityEngine | None = None,
        missing_data_detector: MissingDataDetector | None = None,
        db_manager: DatabaseManager | None = None,
        run_repository: IngestionRunRepository | None = None,
        persistence_worker: AsyncPersistenceWorker | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings or get_settings()
        self.quality_engine = quality_engine or DataQualityEngine()
        self.missing_data_detector = missing_data_detector or MissingDataDetector(
            is_24_7_market=(self.provider.provider_name == "binance")
        )
        self.db_manager = db_manager
        self.run_repository = run_repository or IngestionRunRepository(db_manager=self.db_manager)
        self.persistence_worker = persistence_worker

    async def import_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        force_reimport: bool = False,
    ) -> IngestionRun:
        """Execute historical candlestick import with audit tracking and idempotency."""
        clean_symbol = symbol.upper()
        now = datetime.now(UTC)

        # 1. Idempotency Check: check if identical completed run already exists
        if not force_reimport and start and end:
            existing = await self.run_repository.find_completed_run(
                provider=self.provider.provider_name,
                symbol=clean_symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            if existing:
                logger.info(
                    "historical_import_already_completed_idempotent_skip",
                    run_id=existing.run_id,
                    symbol=clean_symbol,
                    start=start.isoformat(),
                    end=end.isoformat(),
                )
                return existing

        # 2. Initialize IngestionRun ledger record
        run = IngestionRun(
            provider=self.provider.provider_name,
            symbol=clean_symbol,
            timeframe=timeframe,
            requested_start=start,
            requested_end=end,
            status=IngestionRunStatus.IN_PROGRESS,
            created_at=now,
        )
        await self.run_repository.create_run(run)

        logger.info(
            "starting_historical_candle_import",
            run_id=run.run_id,
            provider=self.provider.provider_name,
            symbol=clean_symbol,
            timeframe=timeframe,
            limit=limit,
        )

        try:
            # 3. Source fetch
            raw_candles: list[Candle] = await self.provider.fetch_historical_candles(
                symbol=clean_symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit,
            )
            run.records_received = len(raw_candles)

            if not raw_candles:
                run.status = IngestionRunStatus.COMPLETED
                run.completion_time = datetime.now(UTC)
                await self.run_repository.update_run(run)
                return run

            # Record actual start and end
            run.actual_start = min(c.start_time for c in raw_candles)
            run.actual_end = max(c.end_time for c in raw_candles)

            # 4. Raw persistence (Preserve raw downloaded records immutably)
            self._persist_raw_historical(run.run_id, clean_symbol, raw_candles)

            # 5. Validation & Deduplication
            valid_candles: list[Candle] = []
            seen_intervals: set[str] = set()

            for c in raw_candles:
                interval_key = f"{c.symbol}:{c.timeframe}:{c.start_time.isoformat()}"
                if interval_key in seen_intervals:
                    run.duplicates += 1
                    continue
                seen_intervals.add(interval_key)

                val_res = self.quality_engine.validate_candle(c)
                if val_res.is_valid:
                    valid_candles.append(c)
                    run.records_accepted += 1
                else:
                    run.records_rejected += 1
                    err_msg = f"Invalid candle at {c.start_time.isoformat()}: {[i.message for i in val_res.issues]}"
                    run.errors.append(err_msg)

            # 6. Missing Data & Gap Detection
            gaps = self.missing_data_detector.detect_gaps(valid_candles)
            if gaps:
                logger.info(
                    "historical_import_gaps_identified",
                    run_id=run.run_id,
                    gaps_count=len(gaps),
                    total_missing=sum(g.missing_candles_count for g in gaps),
                )

            # 7. Clean Persistence (Parquet & Database)
            today_str = datetime.now(UTC).strftime("%Y%m%d")
            parquet_dir = self.settings.parquet_storage_path / clean_symbol
            parquet_file = (
                parquet_dir / f"{clean_symbol}_{timeframe}_{today_str}_{run.run_id[:8]}.parquet"
            )
            save_candles_to_parquet(valid_candles, parquet_file)

            if self.persistence_worker:
                for c in valid_candles:
                    self.persistence_worker.enqueue_candle(c)

            # 8. Mark Run Completed
            run.status = (
                IngestionRunStatus.COMPLETED
                if run.records_rejected == 0
                else IngestionRunStatus.PARTIAL
            )
            run.completion_time = datetime.now(UTC)
            await self.run_repository.update_run(run)

            logger.info(
                "historical_candle_import_finished",
                run_id=run.run_id,
                status=run.status.value,
                accepted=run.records_accepted,
                rejected=run.records_rejected,
                duplicates=run.duplicates,
            )
            return run

        except Exception as e:
            run.status = IngestionRunStatus.FAILED
            run.errors.append(str(e))
            run.completion_time = datetime.now(UTC)
            await self.run_repository.update_run(run)
            logger.exception("historical_import_failed", run_id=run.run_id, error=str(e))
            raise

    def _persist_raw_historical(self, run_id: str, symbol: str, candles: list[Candle]) -> None:
        """Write raw immutable historical dump to disk."""
        raw_path = self.settings.raw_storage_path / "historical"
        raw_path.mkdir(parents=True, exist_ok=True)
        dump_file = raw_path / f"raw_hist_{symbol}_{run_id}.jsonl"
        with open(dump_file, "w", encoding="utf-8") as f:
            for c in candles:
                f.write(c.model_dump_json() + "\n")
