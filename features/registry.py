"""Feature Registry for Phase 3.

Maintains an immutable, versioned catalog of all measurable market features,
including their formulas, required lookbacks, parameters, and calculation categories.
"""

from features.models import FeatureDefinition
from monitoring.logging import get_logger

logger = get_logger(__name__)


class FeatureRegistry:
    """Central registry holding metadata for all computable features."""

    def __init__(self) -> None:
        self._definitions: dict[str, FeatureDefinition] = {}

    def register(self, definition: FeatureDefinition) -> None:
        """Register a feature definition into the catalog."""
        key = definition.name.lower()
        if key in self._definitions:
            existing = self._definitions[key]
            if existing.version == definition.version and existing != definition:
                raise ValueError(
                    f"Conflicting definition for feature '{key}' under version {definition.version}. "
                    "Feature definitions must be immutable within the same version."
                )
        self._definitions[key] = definition

    def get(self, name: str) -> FeatureDefinition | None:
        """Retrieve definition for a feature name."""
        return self._definitions.get(name.lower())

    def list_features(
        self,
        category: str | None = None,
        version: str | None = None,
        timeframe: str | None = None,
    ) -> list[FeatureDefinition]:
        """List registered features matching optional filters."""
        results = list(self._definitions.values())
        if category:
            results = [f for f in results if f.category == category]
        if version:
            results = [f for f in results if f.version == version]
        if timeframe:
            results = [f for f in results if f.timeframe == timeframe]
        return results

    def get_max_lookback(self, feature_names: list[str]) -> int:
        """Calculate the maximum lookback required across a list of features."""
        max_lb = 1
        for name in feature_names:
            defn = self.get(name)
            if defn and defn.lookback > max_lb:
                max_lb = defn.lookback
        return max_lb

    def get_catalog(self) -> list[dict]:
        """Export comprehensive feature catalog for documentation and governance."""
        return [defn.model_dump() for defn in self._definitions.values()]

    def count(self) -> int:
        """Total number of registered features."""
        return len(self._definitions)


# Global default registry instance
default_registry = FeatureRegistry()
