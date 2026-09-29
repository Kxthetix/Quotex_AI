"""Historical Pattern Engine for Phase 4.

Searches historical market states and multi-candle sequence patterns for similar conditions,
and calculates the conditional distribution of subsequent forward outcomes (returns, MFE, MAE).
Strict anti-leakage: At timestamp T, only historical states whose forward evaluation window
concluded on or before T are queried.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from analysis.pattern.models import HistoricalMatch, PatternDistribution, PatternSearchResult
from analysis.regime.engine import MarketRegimeEngine
from analysis.regime.models import MarketRegime
from data.ingestion.models import Candle
from features.engine import candles_to_dataframe
from features.models import FeatureVector

# Canonical features used for state pattern similarity matching
DEFAULT_SIMILARITY_FEATURES: list[str] = [
    "rsi_14",
    "adx_14",
    "return_5",
    "range_compression_20",
    "relative_vol_20",
    "slope_10",
]


class HistoricalPatternEngine:
    """Finds historical analogous market states and calculates empirical outcome distributions."""

    def __init__(
        self,
        similarity_features: list[str] | None = None,
        regime_engine: MarketRegimeEngine | None = None,
        feature_version: str = "v1.0.0",
    ) -> None:
        self.similarity_features = similarity_features or DEFAULT_SIMILARITY_FEATURES
        self.regime_engine = regime_engine or MarketRegimeEngine()
        self.feature_version = feature_version

    def find_similar_states(
        self,
        query_vector: FeatureVector,
        historical_vectors: Sequence[FeatureVector],
        historical_candles: Sequence[Candle] | pd.DataFrame,
        top_k: int = 10,
        horizon_bars: int = 15,
        target_regime: MarketRegime | None = None,
    ) -> PatternSearchResult:
        """Find the top-K most similar historical feature states and calculate their subsequent outcomes.

        Anti-Leakage Guard:
        Only candidates with candidate_time <= query_vector.timestamp - horizon_delta
        can be selected, ensuring the outcome was fully realized in the past relative to T.
        """
        # Convert candles to DataFrame for rapid forward price indexing
        if isinstance(historical_candles, pd.DataFrame):
            df_candles = historical_candles.copy()
        else:
            df_candles = candles_to_dataframe(historical_candles)

        if "end_time" in df_candles.columns:
            df_candles["time"] = pd.to_datetime(df_candles["end_time"], utc=True)
        else:
            df_candles["time"] = pd.to_datetime(df_candles["start_time"], utc=True)
        df_candles = df_candles.sort_values("time").reset_index(drop=True)
        close_series = df_candles["close"].values
        time_series = df_candles["time"].tolist()
        time_to_idx = {ts: idx for idx, ts in enumerate(time_series)}

        query_ts = query_vector.timestamp
        query_regime = (
            target_regime
            if target_regime is not None
            else self.regime_engine.classify(query_vector).regime
        )

        # Extract query vector values
        q_vals = np.array(
            [float(query_vector.values.get(f) or 0.0) for f in self.similarity_features],
            dtype=np.float64,
        )

        candidates: list[tuple[float, HistoricalMatch]] = []

        for h_vec in historical_vectors:
            h_ts = h_vec.timestamp

            # Strict anti-leakage check: candidate must be strictly before query timestamp
            if h_ts >= query_ts:
                continue

            # Candidate index in candle DataFrame
            h_idx = time_to_idx.get(h_ts)
            if h_idx is None:
                continue

            # Ensure complete forward horizon exists before query_ts
            if h_idx + horizon_bars >= len(df_candles):
                continue
            outcome_end_time = time_series[h_idx + horizon_bars]
            if outcome_end_time > query_ts:
                # Outcome would use data after query_ts! Strictly skip.
                continue

            # Match features
            h_vals = np.array(
                [float(h_vec.values.get(f) or 0.0) for f in self.similarity_features],
                dtype=np.float64,
            )

            # Normalized Euclidean distance
            # Scale differences by standard feature variances for equal weighting
            diff = q_vals - h_vals
            dist = float(np.sqrt(np.mean(diff**2)))
            similarity = float(1.0 / (1.0 + dist))

            # Calculate future forward outcome over horizon
            entry_price = close_series[h_idx]
            forward_prices = close_series[h_idx + 1 : h_idx + horizon_bars + 1]

            forward_return = float((forward_prices[-1] - entry_price) / entry_price)
            mfe = float(np.max(forward_prices - entry_price) / entry_price)
            mae = float(np.min(forward_prices - entry_price) / entry_price)

            h_regime = self.regime_engine.classify(h_vec).regime

            match = HistoricalMatch(
                matched_timestamp=h_ts,
                similarity_score=round(similarity, 4),
                distance=round(dist, 4),
                regime=h_regime,
                forward_return=round(forward_return, 6),
                mfe=round(mfe, 6),
                mae=round(mae, 6),
                horizon_bars=horizon_bars,
                supporting_features={
                    f: float(h_vec.values.get(f) or 0.0) for f in self.similarity_features
                },
            )
            candidates.append((dist, match))

        # Sort candidates by distance ascending (closest first)
        candidates.sort(key=lambda x: x[0])
        top_matches = [m for _, m in candidates[:top_k]]

        distribution = self._calculate_distribution(top_matches, horizon_bars)

        return PatternSearchResult(
            symbol=query_vector.symbol,
            timestamp=query_vector.timestamp,
            timeframe=query_vector.timeframe,
            feature_version=self.feature_version,
            regime=query_regime,
            matches=top_matches,
            distribution=distribution,
            metadata={
                "top_k": top_k,
                "horizon_bars": horizon_bars,
                "total_candidates_scanned": len(candidates),
            },
        )

    def find_sequence_patterns(
        self,
        query_candles: Sequence[Candle],
        historical_candles: Sequence[Candle],
        window_size: int = 20,
        top_k: int = 5,
        horizon_bars: int = 10,
    ) -> PatternSearchResult:
        """Find historical multi-candle sequence patterns matching a trajectory window (20, 50, 100)."""
        if len(query_candles) < window_size:
            raise ValueError(
                f"Query candles length {len(query_candles)} is less than window_size {window_size}."
            )

        query_df = candles_to_dataframe(query_candles).iloc[-window_size:]
        hist_df = (
            candles_to_dataframe(historical_candles)
            .sort_values("start_time")
            .reset_index(drop=True)
        )

        query_prices = query_df["close"].values
        # Normalize query trajectory by initial price of window
        q_norm = query_prices / query_prices[0] - 1.0

        query_ts = pd.to_datetime(query_df["end_time"].iloc[-1], utc=True).to_pydatetime()
        hist_closes = hist_df["close"].values
        hist_times = pd.to_datetime(hist_df["end_time"], utc=True).tolist()

        matches: list[tuple[float, HistoricalMatch]] = []
        n_hist = len(hist_df)

        for i in range(n_hist - window_size - horizon_bars):
            cand_end_time = hist_times[i + window_size - 1].to_pydatetime()
            outcome_end_time = hist_times[i + window_size + horizon_bars - 1].to_pydatetime()

            # Anti-leakage: whole outcome must be <= query timestamp
            if outcome_end_time > query_ts:
                continue

            window_prices = hist_closes[i : i + window_size]
            w_norm = window_prices / window_prices[0] - 1.0

            dist = float(np.sqrt(np.mean((q_norm - w_norm) ** 2)))
            sim = float(1.0 / (1.0 + dist))

            entry_p = window_prices[-1]
            future_p = hist_closes[i + window_size : i + window_size + horizon_bars]

            f_ret = float((future_p[-1] - entry_p) / entry_p)
            mfe = float(np.max(future_p - entry_p) / entry_p)
            mae = float(np.min(future_p - entry_p) / entry_p)

            match = HistoricalMatch(
                matched_timestamp=cand_end_time,
                similarity_score=round(sim, 4),
                distance=round(dist, 4),
                regime=MarketRegime.SIDEWAYS,
                forward_return=round(f_ret, 6),
                mfe=round(mfe, 6),
                mae=round(mae, 6),
                horizon_bars=horizon_bars,
            )
            matches.append((dist, match))

        matches.sort(key=lambda x: x[0])
        top_matches = [m for _, m in matches[:top_k]]
        distrib = self._calculate_distribution(top_matches, horizon_bars)

        return PatternSearchResult(
            symbol=query_candles[-1].symbol,
            timestamp=query_ts,
            timeframe=query_candles[-1].timeframe,
            feature_version=self.feature_version,
            regime=MarketRegime.SIDEWAYS,
            matches=top_matches,
            distribution=distrib,
            metadata={"window_size": window_size, "top_k": top_k, "horizon_bars": horizon_bars},
        )

    def _calculate_distribution(
        self, matches: list[HistoricalMatch], horizon_bars: int
    ) -> PatternDistribution:
        """Compute summary statistics across historical forward outcomes."""
        if not matches:
            return PatternDistribution(
                match_count=0,
                horizon_bars=horizon_bars,
                mean_forward_return=0.0,
                median_forward_return=0.0,
                std_forward_return=0.0,
                positive_outcome_ratio=0.0,
                negative_outcome_ratio=0.0,
                quantiles={},
                mean_mfe=0.0,
                mean_mae=0.0,
            )

        returns = np.array([m.forward_return for m in matches])
        mfes = np.array([m.mfe for m in matches])
        maes = np.array([m.mae for m in matches])

        pos_count = int(np.sum(returns > 0))
        neg_count = int(np.sum(returns < 0))
        total = len(returns)

        return PatternDistribution(
            match_count=total,
            horizon_bars=horizon_bars,
            mean_forward_return=float(np.mean(returns)),
            median_forward_return=float(np.median(returns)),
            std_forward_return=float(np.std(returns)) if total > 1 else 0.0,
            positive_outcome_ratio=round(pos_count / total, 4),
            negative_outcome_ratio=round(neg_count / total, 4),
            quantiles={
                "p10": float(np.percentile(returns, 10)),
                "p25": float(np.percentile(returns, 25)),
                "p50": float(np.percentile(returns, 50)),
                "p75": float(np.percentile(returns, 75)),
                "p90": float(np.percentile(returns, 90)),
            },
            mean_mfe=float(np.mean(mfes)),
            mean_mae=float(np.mean(maes)),
        )
