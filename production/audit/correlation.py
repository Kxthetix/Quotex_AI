"""Correlation ID and Event Chain Propagation (Section 20).

Links operational event causality across all stages of the pipeline:
market_event_id -> prediction_id -> signal_id -> risk_decision_id -> order_id -> fill_id -> trade_id
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Context variable for thread-safe and async-task propagation
_current_correlation_chain: contextvars.ContextVar[CorrelationChain | None] = contextvars.ContextVar(
    "current_correlation_chain", default=None
)


@dataclass
class CorrelationChain:
    """Immutable chain linking a trading action from market tick to trade outcome."""
    correlation_id: str = field(default_factory=lambda: f"corr_{uuid.uuid4().hex[:12]}")
    market_event_id: str | None = None
    prediction_id: str | None = None
    signal_id: str | None = None
    risk_decision_id: str | None = None
    order_id: str | None = None
    fill_id: str | None = None
    trade_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "market_event_id": self.market_event_id,
            "prediction_id": self.prediction_id,
            "signal_id": self.signal_id,
            "risk_decision_id": self.risk_decision_id,
            "order_id": self.order_id,
            "fill_id": self.fill_id,
            "trade_id": self.trade_id,
            "created_at": self.created_at.isoformat(),
        }


class CorrelationTracker:
    """Manages active correlation chains for execution tracing."""

    @staticmethod
    def create_chain(market_event_id: str | None = None) -> CorrelationChain:
        """Initialize a new correlation chain for an incoming market event."""
        chain = CorrelationChain(market_event_id=market_event_id)
        _current_correlation_chain.set(chain)
        return chain

    @staticmethod
    def get_current_chain() -> CorrelationChain:
        """Retrieve current correlation chain or generate a default one."""
        chain = _current_correlation_chain.get()
        if chain is None:
            chain = CorrelationChain()
            _current_correlation_chain.set(chain)
        return chain

    @staticmethod
    def set_current_chain(chain: CorrelationChain) -> None:
        _current_correlation_chain.set(chain)

    @classmethod
    def link_prediction(cls, prediction_id: str) -> None:
        chain = cls.get_current_chain()
        chain.prediction_id = prediction_id

    @classmethod
    def link_signal(cls, signal_id: str) -> None:
        chain = cls.get_current_chain()
        chain.signal_id = signal_id

    @classmethod
    def link_risk_decision(cls, risk_decision_id: str) -> None:
        chain = cls.get_current_chain()
        chain.risk_decision_id = risk_decision_id

    @classmethod
    def link_order(cls, order_id: str) -> None:
        chain = cls.get_current_chain()
        chain.order_id = order_id

    @classmethod
    def link_fill(cls, fill_id: str) -> None:
        chain = cls.get_current_chain()
        chain.fill_id = fill_id

    @classmethod
    def link_trade(cls, trade_id: str) -> None:
        chain = cls.get_current_chain()
        chain.trade_id = trade_id
