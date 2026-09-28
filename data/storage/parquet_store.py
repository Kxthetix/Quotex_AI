"""Parquet offline and research dataset storage.

Provides fast, columnar, compressed storage for historical candles and features,
ensuring identical schema representation between offline research and live execution.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.ingestion.models import Candle
from monitoring.logging import get_logger

logger = get_logger(__name__)

CANDLE_PYARROW_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("timeframe", pa.string()),
        ("start_time", pa.timestamp("us", tz="UTC")),
        ("end_time", pa.timestamp("us", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("trades_count", pa.int64()),
        ("is_closed", pa.bool_()),
    ]
)


def candles_to_dataframe(candles: list[Candle]) -> pd.DataFrame:
    """Convert a list of Candles into a pandas DataFrame matching schema."""
    if not candles:
        return pd.DataFrame(columns=[f.name for f in CANDLE_PYARROW_SCHEMA])

    data = [
        {
            "symbol": c.symbol,
            "timeframe": c.timeframe,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
            "trades_count": int(c.trades_count),
            "is_closed": bool(c.is_closed),
        }
        for c in candles
    ]
    df = pd.DataFrame(data)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    return df


def dataframe_to_candles(df: pd.DataFrame) -> list[Candle]:
    """Convert pandas DataFrame back into domain Candle models."""
    candles: list[Candle] = []
    for _, row in df.iterrows():
        candles.append(
            Candle(
                symbol=str(row["symbol"]),
                timeframe=str(row["timeframe"]),
                start_time=row["start_time"].to_pydatetime()
                if hasattr(row["start_time"], "to_pydatetime")
                else row["start_time"],
                end_time=row["end_time"].to_pydatetime()
                if hasattr(row["end_time"], "to_pydatetime")
                else row["end_time"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                trades_count=int(row["trades_count"]),
                is_closed=bool(row["is_closed"]),
            )
        )
    return candles


def save_candles_to_parquet(candles: list[Candle], output_path: Path) -> Path:
    """Write candles to a standardized Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = candles_to_dataframe(candles)
    table = pa.Table.from_pandas(df, schema=CANDLE_PYARROW_SCHEMA, preserve_index=False)
    pq.write_table(table, output_path, compression="zstd")
    logger.info("candles_saved_to_parquet", path=str(output_path), count=len(candles))
    return output_path


def load_candles_from_parquet(input_path: Path) -> list[Candle]:
    """Read candles from a standardized Parquet file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {input_path}")
    table = pq.read_table(input_path)
    df = table.to_pandas()
    return dataframe_to_candles(df)
