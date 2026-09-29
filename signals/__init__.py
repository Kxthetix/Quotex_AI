"""Signals package for Phase 6 — Signal Engine.

Public API:
- TradingSignal — The canonical signal output.
- SignalAction — BUY / SELL / NO_TRADE enum.
- Evidence, EvidenceType — Evidence records.
- SignalEngine, IntelligenceSnapshot — Main orchestrator.
- FusionEngine, FusionConfig — Evidence fusion.
- ConfidenceScorer, ConfidenceConfig — Post-fusion confidence.
"""

from signals.engine import IntelligenceSnapshot, SignalEngine, SignalEngineConfig
from signals.fusion.engine import FusionConfig, FusionEngine
from signals.models import Evidence, EvidenceType, SignalAction, TradingSignal

__all__ = [
    "ConfidenceConfig",
    "Evidence",
    "EvidenceType",
    "FusionConfig",
    "FusionEngine",
    "IntelligenceSnapshot",
    "SignalAction",
    "SignalEngine",
    "SignalEngineConfig",
    "TradingSignal",
]
