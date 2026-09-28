"""Data ingestion package."""

from data.ingestion.binance_connector import BinanceWebSocketConnector
from data.ingestion.candle_builder import CandleBuilder
from data.ingestion.connection_manager import ConnectionManager, ConnectionStatus
from data.ingestion.market_data_interface import MarketDataInterface
from data.ingestion.models import Candle, LatencyMetrics, MarketEvent, NormalizedTrade
from data.ingestion.pipeline import IngestionPipeline

__all__ = [
    "MarketEvent",
    "NormalizedTrade",
    "Candle",
    "LatencyMetrics",
    "MarketDataInterface",
    "BinanceWebSocketConnector",
    "CandleBuilder",
    "ConnectionManager",
    "ConnectionStatus",
    "IngestionPipeline",
]
