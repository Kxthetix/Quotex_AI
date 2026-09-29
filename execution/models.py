"""Execution Domain Models and State Hierarchy (Sections 28, 31, 32, 35).

Defines Order, Position, AccountState, and the discrete OrderState lifecycle.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"


class OrderState(StrEnum):
    """Discrete states of the execution state machine (Section 32)."""
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"


@dataclass
class Order:
    """Canonical Order entity representing an intent to transact."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    order_id: str = field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:12]}")
    client_order_id: str = ""
    provider_order_id: str | None = None
    asset_id: str = ""
    state: OrderState = OrderState.CREATED
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    slippage_bps: float = 0.0
    signal_id: str | None = None
    risk_decision_id: str | None = None
    correlation_id: str | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.client_order_id:
            self.client_order_id = self.order_id

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["side"] = str(self.side)
        data["order_type"] = str(self.order_type)
        data["state"] = str(self.state)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data


@dataclass
class Position:
    """Canonical Position entity tracking portfolio holding in an asset."""
    symbol: str
    asset_id: str
    quantity: float  # Positive for long, negative for short, 0 for flat
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["updated_at"] = self.updated_at.isoformat()
        return data


@dataclass
class AccountState:
    """Canonical broker/provider account balance snapshot."""
    account_id: str
    total_equity: float
    cash: float
    used_margin: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "total_equity": self.total_equity,
            "cash": self.cash,
            "used_margin": self.used_margin,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "updated_at": self.updated_at.isoformat(),
        }
