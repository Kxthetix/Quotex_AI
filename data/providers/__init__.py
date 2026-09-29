"""Market Data Providers and Failover Subsystem."""

from data.providers.adapters import SyntheticProviderAdapter
from data.providers.base import MarketDataProvider
from data.providers.failover import (
    CrossProviderConsistencyChecker,
    FailoverEvent,
    ProviderFailoverManager,
    ProviderPriority,
)

__all__ = [
    "CrossProviderConsistencyChecker",
    "FailoverEvent",
    "MarketDataProvider",
    "ProviderFailoverManager",
    "ProviderPriority",
    "SyntheticProviderAdapter",
]
