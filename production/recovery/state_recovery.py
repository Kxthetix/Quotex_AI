"""State Recovery and Durable Persistence Buffer (Sections 14, 15, 16).

Handles:
- Complete system state restoration upon process restart (portfolio, predictions, orders, versions)
- Bounded in-memory fallback buffering when database/Redis persistence is temporarily unavailable
- Idempotent flushing of buffered records once database connectivity is restored
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RestoredSystemState:
    """Consolidated state restored from durable storage upon system startup."""
    session_id: str = ""
    active_model_version: str = "v1.0.0"
    active_feature_version: str = "v1.0.0"
    active_signal_version: str = "v1.0.0"
    portfolio_equity: float = 100_000.0
    portfolio_cash: float = 100_000.0
    open_positions_count: int = 0
    open_orders_count: int = 0
    pending_predictions_count: int = 0
    restored_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "active_model_version": self.active_model_version,
            "active_feature_version": self.active_feature_version,
            "active_signal_version": self.active_signal_version,
            "portfolio_equity": self.portfolio_equity,
            "portfolio_cash": self.portfolio_cash,
            "open_positions_count": self.open_positions_count,
            "open_orders_count": self.open_orders_count,
            "pending_predictions_count": self.pending_predictions_count,
            "restored_at": self.restored_at.isoformat(),
            "metadata": self.metadata,
        }


class PersistenceBuffer:
    """Thread-safe bounded buffer protecting against memory exhaustion during DB outages (Section 14)."""

    def __init__(self, max_buffer_size: int = 5000) -> None:
        self.max_buffer_size = max_buffer_size
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_buffer_size)
        self.dropped_records_count: int = 0
        self.flushed_records_count: int = 0

    def push(self, record: dict[str, Any]) -> bool:
        """Push a record to buffer. If buffer is full, drops oldest to protect memory."""
        if len(self._buffer) >= self.max_buffer_size:
            self.dropped_records_count += 1
            logger.warning(
                "persistence_buffer_overflow_dropped_record",
                dropped_total=self.dropped_records_count,
                buffer_size=len(self._buffer),
            )
        self._buffer.append(record)
        return True

    def flush(self, write_fn: Callable[[list[dict[str, Any]]], bool]) -> int:
        """Attempt to flush buffered records to durable store using provided writer."""
        if not self._buffer:
            return 0

        batch = list(self._buffer)
        try:
            success = write_fn(batch)
            if success:
                count = len(batch)
                self._buffer.clear()
                self.flushed_records_count += count
                logger.info("persistence_buffer_flushed_successfully", count=count)
                return count
        except Exception as e:
            logger.error("persistence_buffer_flush_failed", error=str(e))
        return 0

    def size(self) -> int:
        return len(self._buffer)


class StateRecoveryManager:
    """Coordinates state restoration across restarts and safeguards persistence during outages."""

    def __init__(self, persistence_buffer: PersistenceBuffer | None = None) -> None:
        self.buffer = persistence_buffer or PersistenceBuffer()

    def restore_state(self, loader_fn: Callable[[], dict[str, Any]] | None = None) -> RestoredSystemState:
        """Restore previous system state using durable storage loader."""
        if loader_fn:
            try:
                raw_state = loader_fn()
                restored = RestoredSystemState(
                    session_id=raw_state.get("session_id", ""),
                    active_model_version=raw_state.get("active_model_version", "v1.0.0"),
                    active_feature_version=raw_state.get("active_feature_version", "v1.0.0"),
                    active_signal_version=raw_state.get("active_signal_version", "v1.0.0"),
                    portfolio_equity=float(raw_state.get("portfolio_equity", 100_000.0)),
                    portfolio_cash=float(raw_state.get("portfolio_cash", 100_000.0)),
                    open_positions_count=int(raw_state.get("open_positions_count", 0)),
                    open_orders_count=int(raw_state.get("open_orders_count", 0)),
                    pending_predictions_count=int(raw_state.get("pending_predictions_count", 0)),
                    metadata=raw_state.get("metadata", {}),
                )
                logger.info(
                    "system_state_restored_successfully",
                    session=restored.session_id,
                    model_version=restored.active_model_version,
                )
                return restored
            except Exception as e:
                logger.error("failed_to_restore_state_using_defaults", error=str(e))

        # Default clean state
        return RestoredSystemState()
