"""Audit Trail and Structured Logging (Sections 19, 21, 22).

Provides an immutable, tamper-evident audit trail with SHA-256 hash chaining
and structured JSON logging with automatic secret redaction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from production.audit.correlation import CorrelationTracker

# Regex patterns for detecting and masking sensitive tokens in logs/audit records
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token|auth|credential)['\":\s=]+([a-zA-Z0-9_\-\.]{8,})"),
    re.compile(r"(?i)(bearer\s+)([a-zA-Z0-9_\-\.]{8,})"),
]


def redact_sensitive_str(text: str) -> str:
    """Mask credentials and sensitive tokens from text strings."""
    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(r"\1=***REDACTED***", redacted)
    return redacted


def redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact dictionary keys and string values."""
    cleaned: dict[str, Any] = {}
    sensitive_keys = {"api_key", "secret", "password", "token", "auth", "credential", "private_key"}
    for k, v in data.items():
        if any(sk in k.lower() for sk in sensitive_keys):
            cleaned[k] = "***REDACTED***"
        elif isinstance(v, dict):
            cleaned[k] = redact_dict(v)
        elif isinstance(v, list):
            cleaned[k] = [
                redact_dict(item) if isinstance(item, dict)
                else redact_sensitive_str(str(item)) if isinstance(item, str)
                else item
                for item in v
            ]
        elif isinstance(v, str):
            cleaned[k] = redact_sensitive_str(v)
        else:
            cleaned[k] = v
    return cleaned


@dataclass
class AuditRecord:
    """Immutable audit record representing a security or critical operational action."""
    record_id: str
    timestamp: str
    actor: str
    action: str
    resource: str
    previous_state: dict[str, Any] | None
    new_state: dict[str, Any] | None
    reason: str
    correlation_id: str
    request_id: str | None = None
    prev_hash: str = "GENESIS"
    record_hash: str = ""

    def calculate_hash(self) -> str:
        """Compute SHA-256 hash chaining to previous record."""
        payload = {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "reason": self.reason,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "prev_hash": self.prev_hash,
        }
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """Manages append-only audit trail with tamper detection and hash verification."""

    def __init__(self, log_path: Path | str = "data/audit/audit_trail.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._get_last_hash()

    def _get_last_hash(self) -> str:
        """Scan existing audit file to retrieve the latest record hash."""
        if not self.log_path.exists() or os.path.getsize(self.log_path) == 0:
            return "GENESIS"

        last_hash = "GENESIS"
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record_dict = json.loads(line)
                        last_hash = record_dict.get("record_hash", last_hash)
                    except json.JSONDecodeError:
                        continue
        return last_hash

    def record_action(
        self,
        actor: str,
        action: str,
        resource: str,
        reason: str,
        previous_state: dict[str, Any] | None = None,
        new_state: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        request_id: str | None = None,
    ) -> AuditRecord:
        """Append an immutable audit entry."""
        if correlation_id is None:
            correlation_id = CorrelationTracker.get_current_chain().correlation_id

        # Redact any accidental secrets
        safe_prev = redact_dict(previous_state) if previous_state else None
        safe_new = redact_dict(new_state) if new_state else None
        safe_reason = redact_sensitive_str(reason)

        record_id = f"aud_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        timestamp = datetime.now(UTC).isoformat()

        record = AuditRecord(
            record_id=record_id,
            timestamp=timestamp,
            actor=actor,
            action=action,
            resource=resource,
            previous_state=safe_prev,
            new_state=safe_new,
            reason=safe_reason,
            correlation_id=correlation_id,
            request_id=request_id,
            prev_hash=self._last_hash,
        )
        record.record_hash = record.calculate_hash()
        self._last_hash = record.record_hash

        # Atomic append to file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        return record

    def verify_integrity(self) -> tuple[bool, str]:
        """Verify the complete hash chain of the audit trail."""
        if not self.log_path.exists():
            return True, "No audit trail file exists yet."

        expected_prev_hash = "GENESIS"
        line_num = 0

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    return False, f"Corrupted JSON at line {line_num}"

                record = AuditRecord(**data)
                if record.prev_hash != expected_prev_hash:
                    return False, f"Broken hash chain at line {line_num}: expected prev {expected_prev_hash}, got {record.prev_hash}"

                computed = record.calculate_hash()
                if record.record_hash != computed:
                    return False, f"Tampered record at line {line_num}: recorded {record.record_hash}, calculated {computed}"

                expected_prev_hash = record.record_hash

        return True, f"Audit trail verified ({line_num} records valid)."


class StructuredLogger:
    """Production JSON logger conforming to Sections 21 & 22."""

    def __init__(self, service: str = "trading_system", environment: str = "production") -> None:
        self.service = service
        self.environment = environment
        self.logger = logging.getLogger(service)

    def log(
        self,
        level: int,
        component: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Output structured log payload with redaction."""
        if correlation_id is None:
            correlation_id = CorrelationTracker.get_current_chain().correlation_id

        safe_metadata = redact_dict(metadata) if metadata else {}
        safe_message = redact_sensitive_str(message)

        log_payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": logging.getLevelName(level),
            "service": self.service,
            "environment": self.environment,
            "component": component,
            "event_type": event_type,
            "correlation_id": correlation_id,
            "message": safe_message,
            "metadata": safe_metadata,
        }

        self.logger.log(level, json.dumps(log_payload))
        return log_payload

    def info(self, component: str, event_type: str, message: str, **kwargs: Any) -> dict[str, Any]:
        return self.log(logging.INFO, component, event_type, message, metadata=kwargs)

    def warning(self, component: str, event_type: str, message: str, **kwargs: Any) -> dict[str, Any]:
        return self.log(logging.WARNING, component, event_type, message, metadata=kwargs)

    def error(self, component: str, event_type: str, message: str, **kwargs: Any) -> dict[str, Any]:
        return self.log(logging.ERROR, component, event_type, message, metadata=kwargs)

    def critical(self, component: str, event_type: str, message: str, **kwargs: Any) -> dict[str, Any]:
        return self.log(logging.CRITICAL, component, event_type, message, metadata=kwargs)
