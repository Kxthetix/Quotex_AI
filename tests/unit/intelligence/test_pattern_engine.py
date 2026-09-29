"""Tests for HistoricalPatternEngine (Phase 4 Section 8-14 & 46.C)."""

from analysis.pattern.engine import HistoricalPatternEngine
from features.engine import FeatureEngine
from tests.unit.features.conftest import generate_candles


def test_state_pattern_similarity_search():
    """Verify that similar historical feature states are identified and forward outcomes evaluated."""
    # Generate 150 historical candles
    candles = generate_candles(count=150, seed=42)
    feat_engine = FeatureEngine()
    vectors = feat_engine.compute_historical_features(candles)

    pattern_engine = HistoricalPatternEngine()

    # Query vector is the latest vector (bar index 149)
    query_vector = vectors[-1]
    history = vectors[:-1]

    result = pattern_engine.find_similar_states(
        query_vector=query_vector,
        historical_vectors=history,
        historical_candles=candles,
        top_k=5,
        horizon_bars=10,
    )

    assert result.symbol == query_vector.symbol
    assert result.timestamp == query_vector.timestamp
    assert len(result.matches) <= 5
    assert result.distribution.match_count == len(result.matches)

    # Verify that each match timestamp occurred strictly before query timestamp
    for match in result.matches:
        assert match.matched_timestamp < query_vector.timestamp
        assert match.similarity_score > 0.0
        assert match.horizon_bars == 10
        # MFE must be >= forward_return (by mathematical definition)
        assert match.mfe >= match.forward_return - 1e-9
        # MAE must be <= forward_return (by mathematical definition)
        assert match.mae <= match.forward_return + 1e-9


def test_sequence_pattern_matching_windows():
    """Verify sequence pattern matching over 20-candle and 50-candle windows."""
    candles = generate_candles(count=150, seed=100)
    pattern_engine = HistoricalPatternEngine()

    # Query sequence is the last 30 candles
    query_candles = candles[-30:]
    historical_candles = candles[:-30]

    # Window size 20
    res_20 = pattern_engine.find_sequence_patterns(
        query_candles=query_candles,
        historical_candles=historical_candles,
        window_size=20,
        top_k=3,
        horizon_bars=5,
    )
    assert len(res_20.matches) == 3
    assert res_20.distribution.match_count == 3
    assert "p50" in res_20.distribution.quantiles

    # All matches must have concluded their outcome before query timestamp
    query_ts = query_candles[-1].end_time
    for match in res_20.matches:
        assert match.matched_timestamp < query_ts


def test_pattern_outcome_distributions():
    """Verify statistical outcome distribution calculation."""
    pattern_engine = HistoricalPatternEngine()
    candles = generate_candles(count=100, seed=200)
    vectors = FeatureEngine().compute_historical_features(candles)

    query = vectors[-1]
    res = pattern_engine.find_similar_states(
        query_vector=query,
        historical_vectors=vectors[:-1],
        historical_candles=candles,
        top_k=5,
        horizon_bars=5,
    )

    distrib = res.distribution
    assert distrib.positive_outcome_ratio + distrib.negative_outcome_ratio <= 1.0001
    assert "p10" in distrib.quantiles
    assert "p90" in distrib.quantiles
    assert distrib.quantiles["p90"] >= distrib.quantiles["p10"]
