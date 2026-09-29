"""Execution Modes and Safety Guard (Section 29, 30).

Enforces explicit execution mode configuration:
BACKTEST, PAPER, SHADOW, SIMULATED, LIVE.
Strictly prevents accidental transition into LIVE execution.
"""

from __future__ import annotations

import os
from enum import StrEnum


class ExecutionMode(StrEnum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    SIMULATED = "SIMULATED"
    LIVE = "LIVE"


class ModeGuard:
    """Validates execution mode and prevents accidental live trade routing."""

    @staticmethod
    def validate_mode(mode: ExecutionMode) -> None:
        """Verify that LIVE execution mode has been deliberately enabled with explicit env flags."""
        if mode == ExecutionMode.LIVE:
            allow_live = os.environ.get("ALLOW_LIVE_REAL_MONEY_TRADING", "false").lower() == "true"
            if not allow_live:
                raise PermissionError(
                    "CRITICAL SAFETY VIOLATION: LIVE execution mode blocked! "
                    "ALLOW_LIVE_REAL_MONEY_TRADING environment variable must be explicitly 'true'."
                )
