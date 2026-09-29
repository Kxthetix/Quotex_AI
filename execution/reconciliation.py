"""Execution and Position Reconciliation Engine (Sections 33, 34, 58).

Compares internal ledger states against provider reported orders and positions.
If discrepancies exist, triggers POSITION_RECONCILIATION_REQUIRED and blocks unsafe new executions.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from execution.models import Order, Position

if TYPE_CHECKING:
    from production.audit.audit_logger import AuditLogger


@dataclass
class PositionDiscrepancy:
    symbol: str
    internal_quantity: float
    provider_quantity: float
    delta: float
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["detected_at"] = self.detected_at.isoformat()
        return data


class ExecutionReconciliationEngine:
    """Detects and handles order and position state discrepancies."""

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        self.audit_logger = audit_logger
        self.reconciliation_required = False
        self._active_discrepancies: dict[str, PositionDiscrepancy] = {}
        self._lock = threading.Lock()

    def reconcile_positions(
        self,
        internal_positions: dict[str, float],
        provider_positions: dict[str, float],
        tolerance: float = 1e-5,
    ) -> tuple[bool, list[PositionDiscrepancy]]:
        """Section 34, 58: Compare internal vs broker position quantities.
        
        If mismatch is found, marks reconciliation_required = True and logs audit event.
        """
        with self._lock:
            all_symbols = set(internal_positions.keys()) | set(provider_positions.keys())
            found_discrepancies: list[PositionDiscrepancy] = []

            for sym in all_symbols:
                int_qty = internal_positions.get(sym, 0.0)
                prov_qty = provider_positions.get(sym, 0.0)
                delta = abs(int_qty - prov_qty)

                if delta > tolerance:
                    disc = PositionDiscrepancy(
                        symbol=sym,
                        internal_quantity=int_qty,
                        provider_quantity=prov_qty,
                        delta=delta,
                    )
                    found_discrepancies.append(disc)
                    self._active_discrepancies[sym] = disc

            if found_discrepancies:
                self.reconciliation_required = True
                if self.audit_logger:
                    self.audit_logger.record_action(
                        actor="ExecutionReconciliationEngine",
                        action="POSITION_RECONCILIATION_REQUIRED",
                        resource="positions",
                        reason=f"Position mismatch detected on {len(found_discrepancies)} symbols",
                        previous_state={"status": "IN_SYNC"},
                        new_state={
                            "status": "RECONCILIATION_REQUIRED",
                            "discrepancies": [d.to_dict() for d in found_discrepancies],
                        },
                    )
            else:
                self.reconciliation_required = False
                self._active_discrepancies.clear()

            is_in_sync = len(found_discrepancies) == 0
            return is_in_sync, found_discrepancies

    def can_safely_execute(self, symbol: str) -> bool:
        """Returns False if the symbol or portfolio is currently under reconciliation block."""
        with self._lock:
            if self.reconciliation_required:
                return symbol not in self._active_discrepancies
            return True

    def resolve_discrepancy(self, symbol: str, actor: str, resolution_notes: str) -> bool:
        """Manually clear a position discrepancy after audit review."""
        with self._lock:
            if symbol in self._active_discrepancies:
                del self._active_discrepancies[symbol]
                if not self._active_discrepancies:
                    self.reconciliation_required = False

                if self.audit_logger:
                    self.audit_logger.record_action(
                        actor=actor,
                        action="POSITION_DISCREPANCY_RESOLVED",
                        resource=f"position:{symbol}",
                        reason=resolution_notes,
                    )
                return True
            return False
