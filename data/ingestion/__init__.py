"""Data ingestion package."""

from data.ingestion.binance_connector import BinanceWebSocketConnector
from data.ingestion.candle_builder import CandleBuilder
from data.ingestion.connection_manager import ConnectionManager, ConnectionStatus
from data.ingestion.historical_importer import HistoricalDataImporter
from data.ingestion.market_data_interface import (
    HistoricalMarketDataProvider,
    MarketDataInterface,
    WebSocketMarketDataProvider,
)
from data.ingestion.models import (
    Candle,
    CandleCorrectionRecord,
    CandleGapReason,
    CandleGapRecord,
    IngestionRun,
    IngestionRunStatus,
    LatencyMetrics,
    MarketEvent,
    NormalizedTrade,
)
from data.ingestion.pipeline import IngestionPipeline

__all__ = [
    "MarketEvent",
    "NormalizedTrade",
    "Candle",
    "LatencyMetrics",
    "IngestionRun",
    "IngestionRunStatus",
    "CandleCorrectionRecord",
    "CandleGapReason",
    "CandleGapRecord",
    "MarketDataInterface",
    "WebSocketMarketDataProvider",
    "HistoricalMarketDataProvider",
    "BinanceWebSocketConnector",
    "CandleBuilder",
    "ConnectionManager",
    "ConnectionStatus",
    "IngestionPipeline",
    "HistoricalDataImporter",
]
