"""Historical pattern matching and outcome distribution package (Phase 4)."""

from analysis.pattern.engine import DEFAULT_SIMILARITY_FEATURES, HistoricalPatternEngine
from analysis.pattern.models import (
    HistoricalMatch,
    PatternDistribution,
    PatternSearchResult,
)

__all__ = [
    "DEFAULT_SIMILARITY_FEATURES",
    "HistoricalMatch",
    "HistoricalPatternEngine",
    "PatternDistribution",
    "PatternSearchResult",
]
