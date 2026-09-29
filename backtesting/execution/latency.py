"""Latency models for execution simulation (Phase 5).

Models network transmission and exchange queuing delays between signal decision time
and actual order fill time.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta


class LatencyModel(ABC):
    """Abstract base class for execution latency."""

    @abstractmethod
    def calculate_execution_time(self, submission_time: datetime) -> datetime:
        """Return the effective execution timestamp given an order submission timestamp."""
        pass


class ZeroLatencyModel(LatencyModel):
    """Executes with zero time delay (synchronous/immediate execution)."""

    def calculate_execution_time(self, submission_time: datetime) -> datetime:
        return submission_time


class FixedLatencyModel(LatencyModel):
    """Simulates a fixed network & processing delay in seconds or milliseconds."""

    def __init__(self, latency_seconds: float = 0.5) -> None:
        self.latency_seconds = max(0.0, latency_seconds)
        self.delta = timedelta(seconds=self.latency_seconds)

    def calculate_execution_time(self, submission_time: datetime) -> datetime:
        return submission_time + self.delta
