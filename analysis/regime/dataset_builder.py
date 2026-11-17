"""Dataset builder and regime profiler for large historical datasets.

Orchestrates downloading, validation, feature extraction, and regime classification
for multi-month and multi-year market data. Computes regime distribution metrics,
volatility profiles, and exports version-locked Parquet datasets.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.regime.catalog import RegimePreset
from analysis.regime.engine import MarketRegimeEngine
from analysis.regime.models import MarketRegime
from configs.config import get_settings
from data.ingestion.binance_connector import BinanceWebSocketConnector
from data.ingestion.models import Candle
from data.storage.parquet_store import save_candles_to_parquet
from data.validation.engine import DataQualityEngine
from features.engine import FeatureEngine, candles_to_dataframe
from monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeProfile:
    """Summary statistics and regime breakdown of a historical dataset."""

    dataset_name: str
    symbol: str
    timeframe: str
    start_time: str
    end_time: str
    total_candles: int
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    total_return_pct: float
    max_drawdown_pct: float
    regime_distribution: dict[str, dict[str, float]]
    average_regime_streak_bars: dict[str, float]
    output_parquet_path: str
    output_annotated_parquet_path: str
    generated_at: str


class RegimeDatasetBuilder:
    """Orchestrates ingestion, feature enrichment, and regime profiling."""

    def __init__(
        self,
        connector: BinanceWebSocketConnector | None = None,
        feature_engine: FeatureEngine | None = None,
        regime_engine: MarketRegimeEngine | None = None,
        quality_engine: DataQualityEngine | None = None,
    ) -> None:
        settings = get_settings()
        self.connector = connector or BinanceWebSocketConnector(rest_url=settings.rest_base_url)
        self.feature_engine = feature_engine or FeatureEngine()
        self.regime_engine = regime_engine or MarketRegimeEngine()
        self.quality_engine = quality_engine or DataQualityEngine()

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Candle]:
        """Fetch historical candles with automatic pagination."""
        logger.info(
            "fetching_candles_for_regime_dataset",
            symbol=symbol,
            timeframe=timeframe,
            start=start.isoformat() if start else None,
            end=end.isoformat() if end else None,
            limit=limit,
        )
        candles = await self.connector.fetch_historical_candles(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
        )
        return candles

    def build_and_profile(
        self,
        candles: list[Candle],
        dataset_name: str,
        output_dir: Path | str = "data/datasets/market_regimes",
    ) -> tuple[pd.DataFrame, RegimeProfile]:
        """Validate, extract features, classify regimes, profile, and save dataset."""
        if not candles:
            raise ValueError("Candles list cannot be empty.")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        symbol = candles[0].symbol
        timeframe = candles[0].timeframe

        # 1. Validation
        valid_candles: list[Candle] = []
        for c in candles:
            res = self.quality_engine.validate_candle(c)
            if res.is_valid:
                valid_candles.append(c)

        if not valid_candles:
            raise ValueError("No valid candles remained after data quality filtering.")

        # Save standard clean OHLCV parquet
        raw_parquet_path = out_path / f"{dataset_name}_{symbol}_{timeframe}_ohlcv.parquet"
        save_candles_to_parquet(valid_candles, raw_parquet_path)

        # 2. DataFrame conversion and Feature Calculation
        df_candles = candles_to_dataframe(valid_candles)
        features_df = self.feature_engine.compute_features_dataframe(df_candles)

        # 3. Classify Regimes for each bar
        feature_vectors = self.feature_engine.compute_historical_features(
            df_candles, symbol=symbol
        )
        regimes: list[str] = []
        confidences: list[float] = []

        for vec in feature_vectors:
            assessment = self.regime_engine.classify(vec)
            regimes.append(assessment.regime.value)
            confidences.append(assessment.confidence)

        # Pad with SIDEWAYS / 0.5 for warm-up rows if lengths differ
        if len(regimes) < len(df_candles):
            diff = len(df_candles) - len(regimes)
            regimes = [MarketRegime.SIDEWAYS.value] * diff + regimes
            confidences = [0.5] * diff + confidences

        # 4. Enriched DataFrame
        enriched_df = df_candles.copy()
        enriched_df["regime"] = regimes[: len(enriched_df)]
        enriched_df["regime_confidence"] = confidences[: len(enriched_df)]

        # Add key features
        for key in ["rsi_14", "adx_14", "ema_20", "ema_50", "rolling_vol_20", "atr_14"]:
            if key in features_df.columns:
                enriched_df[key] = features_df[key].values

        annotated_parquet_path = (
            out_path / f"{dataset_name}_{symbol}_{timeframe}_annotated.parquet"
        )
        enriched_df.to_parquet(annotated_parquet_path, index=False)

        # 5. Profile Calculation
        open_price = float(df_candles["open"].iloc[0])
        close_price = float(df_candles["close"].iloc[-1])
        high_price = float(df_candles["high"].max())
        low_price = float(df_candles["low"].min())
        total_return_pct = round(((close_price - open_price) / open_price) * 100.0, 2)

        # Max drawdown
        cummax = df_candles["close"].cummax()
        drawdowns = (df_candles["close"] - cummax) / cummax
        max_drawdown_pct = round(abs(float(drawdowns.min())) * 100.0, 2)

        # Regime distributions and streaks
        regime_counts = enriched_df["regime"].value_counts()
        total_bars = len(enriched_df)
        distribution: dict[str, dict[str, float]] = {}
        for r_name, cnt in regime_counts.items():
            distribution[str(r_name)] = {
                "count": int(cnt),
                "pct": round((float(cnt) / total_bars) * 100.0, 2),
            }

        # Calculate streak lengths
        regime_series = enriched_df["regime"]
        streak_lengths: dict[str, list[int]] = {}
        cur_regime = regime_series.iloc[0]
        cur_len = 1
        for r in regime_series.iloc[1:]:
            if r == cur_regime:
                cur_len += 1
            else:
                streak_lengths.setdefault(cur_regime, []).append(cur_len)
                cur_regime = r
                cur_len = 1
        streak_lengths.setdefault(cur_regime, []).append(cur_len)

        avg_streaks = {
            r: round(float(np.mean(lens)), 1) for r, lens in streak_lengths.items() if lens
        }

        profile = RegimeProfile(
            dataset_name=dataset_name,
            symbol=symbol,
            timeframe=timeframe,
            start_time=df_candles["start_time"].iloc[0].isoformat(),
            end_time=df_candles["end_time"].iloc[-1].isoformat(),
            total_candles=total_bars,
            open_price=open_price,
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            regime_distribution=distribution,
            average_regime_streak_bars=avg_streaks,
            output_parquet_path=str(raw_parquet_path),
            output_annotated_parquet_path=str(annotated_parquet_path),
            generated_at=datetime.now(UTC).isoformat(),
        )

        manifest_path = out_path / f"{dataset_name}_{symbol}_{timeframe}_profile.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, indent=2)

        return enriched_df, profile

    async def fetch_and_build_preset(
        self,
        preset: RegimePreset,
        symbol: str | None = None,
        timeframe: str = "1h",
        output_dir: Path | str = "data/datasets/market_regimes",
    ) -> tuple[pd.DataFrame, RegimeProfile]:
        """Fetch and build a dataset directly from a curated RegimePreset."""
        target_symbol = symbol or preset.default_symbol
        candles = await self.fetch_candles(
            symbol=target_symbol,
            timeframe=timeframe,
            start=preset.start,
            end=preset.end,
        )
        return self.build_and_profile(
            candles=candles,
            dataset_name=preset.preset_id,
            output_dir=output_dir,
        )
