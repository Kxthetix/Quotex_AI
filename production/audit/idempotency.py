"""Idempotency Management (Section 17).

Guarantees that retries of critical state-changing operations
(event processing, predictions, orders, fills, outcomes, backups)
do not produce duplicate mutations or double-counting.
"""

from __future__ import annotations

import collections
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class IdempotencyError(Exception):
    """Raised on idempotency violation or concurrent duplicate execution."""
    pass


@dataclass
class IdempotencyRecord:
    key: str
    action_type: str
    created_at: float = field(default_factory=time.time)
    status: str = "COMPLETED"  # IN_PROGRESS, COMPLETED, FAILED
    result_data: Any = None


class IdempotencyManager:
    """Thread-safe idempotency manager with bounded LRU memory and TTL expiration."""

    def __init__(self, max_records: int = 10000, default_ttl_sec: float = 86400.0) -> None:
        self.max_records = max_records
        self.default_ttl_sec = default_ttl_sec
        self._records: collections.OrderedDict[str, IdempotencyRecord] = collections.OrderedDict()
        self._lock = threading.Lock()

    def generate_key(self, domain: str, *args: Any, **kwargs: Any) -> str:
        """Create a deterministic SHA-256 idempotency key from arguments."""
        payload = {
            "domain": domain,
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        }
        serialized = json.dumps(payload, sort_keys=True)
        return f"{domain}:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]}"

    def is_processed(self, key: str) -> bool:
        """Check if an operation key has already completed within its TTL."""
        with self._lock:
            record = self._records.get(key)
            if not record:
                return False
            if time.time() - record.created_at > self.default_ttl_sec:
                del self._records[key]
                return False
            return record.status == "COMPLETED"

    def get_result(self, key: str) -> Any | None:
        """Get cached execution result if previously completed."""
        with self._lock:
            record = self._records.get(key)
            if record and record.status == "COMPLETED":
                return record.result_data
            return None

    def mark_in_progress(self, key: str, action_type: str) -> None:
        """Mark operation as in-flight; prevents concurrent identical runs."""
        with self._lock:
            record = self._records.get(key)
            if record:
                if time.time() - record.created_at <= self.default_ttl_sec:
                    if record.status == "IN_PROGRESS":
                        raise IdempotencyError(f"Operation with key '{key}' is already in progress.")
                    if record.status == "COMPLETED":
                        raise IdempotencyError(f"Operation with key '{key}' has already completed.")

            self._records[key] = IdempotencyRecord(key=key, action_type=action_type, status="IN_PROGRESS")
            self._prune_if_needed()

    def mark_completed(self, key: str, action_type: str, result: Any = None) -> None:
        """Mark operation as completed and store result for replay."""
        with self._lock:
            self._records[key] = IdempotencyRecord(
                key=key,
                action_type=action_type,
                status="COMPLETED",
                result_data=result,
            )
            self._records.move_to_end(key)
            self._prune_if_needed()

    def mark_failed(self, key: str) -> None:
        """Remove or mark failed so retries can proceed."""
        with self._lock:
            if key in self._records:
                del self._records[key]

    def execute_idempotent(
        self,
        key: str,
        action_type: str,
        func: Callable[[], T],
    ) -> tuple[T, bool]:
        """Execute a function idempotently.
        
        Returns:
            tuple[result, was_executed]: was_executed is True if freshly run,
            False if returned from cache.
        """
        with self._lock:
            if key in self._records:
                rec = self._records[key]
                if time.time() - rec.created_at <= self.default_ttl_sec:
                    if rec.status == "COMPLETED":
                        return rec.result_data, False
                    if rec.status == "IN_PROGRESS":
                        raise IdempotencyError(f"Operation {key} is already in progress.")
            
            # Mark in progress
            self._records[key] = IdempotencyRecord(key=key, action_type=action_type, status="IN_PROGRESS")
            self._prune_if_needed()

        try:
            result = func()
        except Exception:
            self.mark_failed(key)
            raise

        self.mark_completed(key, action_type, result)
        return result, True

    def _prune_if_needed(self) -> None:
        """Evict oldest entries beyond max_records or expired TTL."""
        now = time.time()
        # Evict expired
        keys_to_remove = [k for k, v in self._records.items() if now - v.created_at > self.default_ttl_sec]
        for k in keys_to_remove:
            del self._records[k]

        # Evict LRU if over capacity
        while len(self._records) > self.max_records:
            self._records.popitem(last=False)
