"""Failure Classification Engine for Phase 9 (Section 7).

Classifies runtime errors and exceptions into:
- TRANSIENT: Temporary timeouts or transient blips (safe to retry immediately or with short delay).
- RECOVERABLE: Connection drops, socket disconnects (reconnect protocol, backfill gaps).
- DEGRADED: Telemetry lag or non-critical worker offline (continue trading in degraded mode).
- CRITICAL: Stale data, model mismatch, circuit breaker trip (halt new trading decisions immediately).
- FATAL: Persistent state corruption, unrecoverable schema mismatch (graceful shutdown).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from monitoring.logging import get_logger

logger = get_logger(__name__)


class FailureSeverity(StrEnum):
    """Explicit failure classification (Section 7)."""
    TRANSIENT = "TRANSIENT"
    RECOVERABLE = "RECOVERABLE"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"


# Alias for compatibility
FailureCategory = FailureSeverity


class RecoveryAction(StrEnum):
    """Recommended immediate action following failure detection (Section 8)."""
    RETRY = "RETRY"
    RECONNECT = "RECONNECT"
    DEGRADE_MODE = "DEGRADE_MODE"
    HALT_DECISIONS = "HALT_DECISIONS"
    SAFE_SHUTDOWN = "SAFE_SHUTDOWN"
    # Additional aliases matching Section 8
    RETRY_WITH_BACKOFF = "RETRY"
    RECONNECT_AND_RESUBSCRIBE = "RECONNECT"
    BLOCK_DECISION_MAKING = "HALT_DECISIONS"


@dataclass
class FailureAssessment:
    """Detailed diagnosis and recommended recovery strategy for a failure event."""
    severity: FailureSeverity
    recommended_action: RecoveryAction
    error_message: str
    error_type: str
    component: str = "unknown"
    can_continue_trading: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": str(self.severity),
            "recommended_action": str(self.recommended_action),
            "error_message": self.error_message,
            "error_type": self.error_type,
            "component": self.component,
            "can_continue_trading": self.can_continue_trading,
            "metadata": self.metadata,
        }


class FailureClassifier:
    """Classifies operational failures and determines safety policy."""

    def __init__(self) -> None:
        self._custom_mappings: dict[type[BaseException], tuple[FailureSeverity, RecoveryAction]] = {}

    def register_mapping(
        self,
        exc_type: type[BaseException],
        severity: FailureSeverity,
        action: RecoveryAction,
    ) -> None:
        """Register custom exception mapping."""
        self._custom_mappings[exc_type] = (severity, action)

    def classify(self, error: BaseException, component: str = "system") -> FailureAssessment:
        """Classify an exception and determine immediate recovery action."""
        err_type = type(error).__name__
        err_msg = str(error)

        # 1. Custom mappings
        for exc_cls, (sev, act) in self._custom_mappings.items():
            if isinstance(error, exc_cls):
                can_trade = sev in (FailureSeverity.TRANSIENT, FailureSeverity.DEGRADED)
                return FailureAssessment(
                    severity=sev,
                    recommended_action=act,
                    error_message=err_msg,
                    error_type=err_type,
                    component=component,
                    can_continue_trading=can_trade,
                )

        # 2. Built-in heuristics
        err_lower = err_msg.lower()

        # Transient network errors
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)) or "timeout" in err_lower or "timed out" in err_lower:
            return FailureAssessment(
                severity=FailureSeverity.TRANSIENT,
                recommended_action=RecoveryAction.RETRY,
                error_message=err_msg,
                error_type=err_type,
                component=component,
                can_continue_trading=True,
            )

        # Recoverable connection drops
        if isinstance(error, ConnectionError) or "connection reset" in err_lower or "disconnect" in err_lower:
            return FailureAssessment(
                severity=FailureSeverity.RECOVERABLE,
                recommended_action=RecoveryAction.RECONNECT,
                error_message=err_msg,
                error_type=err_type,
                component=component,
                can_continue_trading=False,
            )

        # Critical data, model, or risk errors
        if "stale" in err_lower or "circuit breaker" in err_lower or "incompatible" in err_lower:
            return FailureAssessment(
                severity=FailureSeverity.CRITICAL,
                recommended_action=RecoveryAction.HALT_DECISIONS,
                error_message=err_msg,
                error_type=err_type,
                component=component,
                can_continue_trading=False,
            )

        # Fatal corruption or schema mismatches
        if "corrupt" in err_lower or "integrity violation" in err_lower:
            return FailureAssessment(
                severity=FailureSeverity.FATAL,
                recommended_action=RecoveryAction.SAFE_SHUTDOWN,
                error_message=err_msg,
                error_type=err_type,
                component=component,
                can_continue_trading=False,
            )

        # Default fallback
        logger.warning("unclassified_failure", error_type=err_type, error=err_msg, component=component)
        return FailureAssessment(
            severity=FailureSeverity.CRITICAL,
            recommended_action=RecoveryAction.HALT_DECISIONS,
            error_message=err_msg,
            error_type=err_type,
            component=component,
            can_continue_trading=False,
        )
