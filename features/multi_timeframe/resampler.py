"""Candlestick resampling utilities for multi-timeframe feature calculations.

Provides deterministic aggregation of lower-timeframe candles (e.g. 1m) into
higher-timeframe candles (5m, 15m, 1h, 4h, 1d) with strict boundary tracking
and incomplete-candle discarding to prevent data leakage.
"""

import re
from datetime import timedelta

import pandas as pd

TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}

TIMEFRAME_PANDAS_FREQS: dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
    "1w": "1W",
}


def parse_timeframe_to_timedelta(tf: str) -> timedelta:
    """Convert timeframe string (e.g. '5m', '1h', '1d') to timedelta."""
    clean = tf.strip().lower()
    if clean in TIMEFRAME_DELTAS:
        return TIMEFRAME_DELTAS[clean]

    match = re.match(r"^(\d+)([mhdwn])$", clean)
    if not match:
        raise ValueError(f"Unsupported timeframe format: '{tf}'")

    num = int(match.group(1))
    unit = match.group(2)
    if unit in ("m", "min"):
        return timedelta(minutes=num)
    elif unit == "h":
        return timedelta(hours=num)
    elif unit == "d":
        return timedelta(days=num)
    elif unit == "w":
        return timedelta(weeks=num)
    raise ValueError(f"Unknown timeframe unit in '{tf}'")


def parse_timeframe_to_freq(tf: str) -> str:
    """Convert timeframe string to pandas frequency string."""
    clean = tf.strip().lower()
    if clean in TIMEFRAME_PANDAS_FREQS:
        return TIMEFRAME_PANDAS_FREQS[clean]

    match = re.match(r"^(\d+)([mhdwn])$", clean)
    if not match:
        raise ValueError(f"Unsupported timeframe format: '{tf}'")

    num = int(match.group(1))
    unit = match.group(2)
    if unit in ("m", "min"):
        return f"{num}min"
    elif unit == "h":
        return f"{num}h"
    elif unit == "d":
        return f"{num}D"
    elif unit == "w":
        return f"{num}W"
    raise ValueError(f"Unknown timeframe unit in '{tf}'")


def resample_candles_to_timeframe(
    df: pd.DataFrame,
    target_timeframe: str,
    base_timeframe: str = "1m",
) -> pd.DataFrame:
    """Resample lower-timeframe OHLCV candles to higher-timeframe candles.

    Strict Invariants:
    1. Only complete candles are returned. If the latest higher-timeframe candle is incomplete
       (i.e. its theoretical end_time > max available base end_time), it is discarded.
    2. The resulting DataFrame has columns: [open, high, low, close, volume, start_time, end_time, available_at].
    3. `available_at` equals `end_time`, marking the earliest timestamp this completed candle can be observed.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "start_time",
                "end_time",
                "available_at",
            ]
        )

    work_df = df.copy()

    # Determine time column
    time_col = None
    if "start_time" in work_df.columns:
        time_col = "start_time"
    elif "timestamp" in work_df.columns:
        time_col = "timestamp"
    elif isinstance(work_df.index, pd.DatetimeIndex):
        work_df = work_df.reset_index()
        time_col = work_df.columns[0]
    else:
        raise ValueError("DataFrame must contain 'start_time', 'timestamp', or a DatetimeIndex.")

    work_df[time_col] = pd.to_datetime(work_df[time_col], utc=True)

    # Determine base candle delta
    base_delta = parse_timeframe_to_timedelta(base_timeframe)
    target_delta = parse_timeframe_to_timedelta(target_timeframe)

    if target_delta <= base_delta:
        # Same or smaller timeframe, return copy with standard columns
        if "end_time" not in work_df.columns:
            work_df["end_time"] = work_df[time_col] + base_delta
        work_df["available_at"] = work_df["end_time"]
        work_df["start_time"] = work_df[time_col]
        return work_df[
            ["open", "high", "low", "close", "volume", "start_time", "end_time", "available_at"]
        ]

    # Calculate latest available base end time
    if "end_time" in work_df.columns:
        max_base_end = pd.to_datetime(work_df["end_time"], utc=True).max()
    else:
        max_base_end = work_df[time_col].max() + base_delta

    freq = parse_timeframe_to_freq(target_timeframe)
    indexed_df = work_df.set_index(time_col).sort_index()

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in indexed_df.columns:
        agg_rules["volume"] = "sum"

    resampled = indexed_df.resample(freq, closed="left", label="left").agg(agg_rules)
    # Drop empty buckets
    resampled = resampled.dropna(subset=["close"]).reset_index()
    resampled = resampled.rename(columns={time_col: "start_time"})

    resampled["end_time"] = resampled["start_time"] + target_delta
    resampled["available_at"] = resampled["end_time"]

    # Filter out incomplete candles where end_time > max_base_end
    # (i.e. candle is still in-progress at the end of the input stream)
    completed_mask = resampled["end_time"] <= max_base_end
    completed = resampled[completed_mask].copy()

    if "volume" not in completed.columns:
        completed["volume"] = 0.0

    return completed[
        ["open", "high", "low", "close", "volume", "start_time", "end_time", "available_at"]
    ].reset_index(drop=True)
