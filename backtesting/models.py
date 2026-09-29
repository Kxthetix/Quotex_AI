"""Core data models, enumerations, and configuration schemas for Phase 5 Backtesting Engine.

Defines orders, fills, positions, trade records, equity tracking, and configuration.
Strict invariant: All temporal models record exact timestamps for auditability and leakage prevention.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from analysis.regime.models import MarketRegime


class OrderType(StrEnum):
    """Supported order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(StrEnum):
    """Order side: BUY or SELL."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    """Lifecycle status of an order."""

    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(StrEnum):
    """Side of an open or closed position."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class ExitReason(StrEnum):
    """Trigger reason for closing or reducing a position."""

    SIGNAL = "SIGNAL"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TIME_LIMIT = "TIME_LIMIT"
    END_OF_BACKTEST = "END_OF_BACKTEST"
    RISK_LIMIT = "RISK_LIMIT"


class SizingMethod(StrEnum):
    """Position sizing methodology."""

    FIXED_QUANTITY = "FIXED_QUANTITY"
    FIXED_NOTIONAL = "FIXED_NOTIONAL"
    PERCENT_CAPITAL = "PERCENT_CAPITAL"
    VOLATILITY_ADJUSTED = "VOLATILITY_ADJUSTED"


class EndOfTestAction(StrEnum):
    """Action taken on remaining open positions at the conclusion of backtesting."""

    FORCE_CLOSE = "FORCE_CLOSE"
    MARK_TO_MARKET = "MARK_TO_MARKET"


@dataclass
class BacktestConfig:
    """Versioned configuration for reproducible backtest execution."""

    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    initial_capital: float = 10000.0
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Fee model
    maker_fee: float = 0.0002  # 2 bps (0.02%)
    taker_fee: float = 0.0005  # 5 bps (0.05%)
    fixed_fee: float = 0.0

    # Spread & Slippage models
    spread_pct: float = 0.0002  # 2 bps default spread (0.02%)
    slippage_pct: float = 0.0003  # 3 bps default slippage (0.03%)
    slippage_vol_factor: float = 0.0  # Optional volatility-dependent multiplier

    # Latency model
    latency_seconds: float = 0.0  # 0.0 = immediate next-event execution

    # Liquidity / Partial fills
    max_bar_volume_pct: float = 0.10  # Max 10% of bar volume per fill
    allow_partial_fills: bool = True

    # Position sizing & limits
    sizing_method: SizingMethod = SizingMethod.PERCENT_CAPITAL
    sizing_value: float = 0.10  # e.g., 10% of portfolio capital per trade
    max_position_size: float = 5.0  # max 5 BTC
    max_notional: float = 50000.0  # max $50k notional exposure
    max_leverage: float = 1.0  # 1.0 = cash only / no borrowing

    # End-of-backtest behavior
    end_of_test_action: EndOfTestAction = EndOfTestAction.FORCE_CLOSE

    # Versioning & Reproducibility
    feature_version: str = "v1.0.0"
    model_version: str = "v1.0.0"
    random_seed: int = 42
    config_version: str = "v1.0.0"


@dataclass
class Order:
    """Explicit representation of a trading order."""

    order_id: str = field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:10]}")
    symbol: str = "BTC/USDT"
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.CREATED

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    submitted_at: datetime | None = None
    executed_at: datetime | None = None

    filled_qty: float = 0.0
    remaining_qty: float = 0.0
    avg_fill_price: float = 0.0
    cumulative_fees: float = 0.0
    cumulative_slippage: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.remaining_qty == 0.0 and self.quantity > 0.0:
            self.remaining_qty = self.quantity


@dataclass
class Fill:
    """Execution fill event representing a completed or partial order fill."""

    fill_id: str = field(default_factory=lambda: f"fill_{uuid.uuid4().hex[:10]}")
    order_id: str = ""
    symbol: str = "BTC/USDT"
    side: OrderSide = OrderSide.BUY
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    price: float = 0.0
    quantity: float = 0.0
    fee: float = 0.0
    slippage_cost: float = 0.0
    spread_cost: float = 0.0
    is_partial: bool = False


@dataclass
class Position:
    """Active open position tracking quantity, entry price, and valuation."""

    symbol: str = "BTC/USDT"
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    entry_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_update_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    cumulative_fees: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0.0 or self.side == PositionSide.FLAT:
            return 0.0
        if self.side == PositionSide.LONG:
            return self.quantity * (self.current_price - self.avg_entry_price)
        else:
            return self.quantity * (self.avg_entry_price - self.current_price)


@dataclass
class TradeRecord:
    """Immutable record of a completed trade cycle from entry to full exit."""

    trade_id: str
    symbol: str
    side: PositionSide
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    slippage: float
    spread: float
    net_pnl: float
    return_pct: float
    holding_period_seconds: float
    exit_reason: ExitReason
    model_version: str
    feature_version: str
    regime_at_entry: MarketRegime = MarketRegime.SIDEWAYS


@dataclass
class EquityPoint:
    """Time-series snapshot of portfolio equity and exposure."""

    timestamp: datetime
    equity: float
    cash: float
    market_value: float
    gross_exposure: float
    net_exposure: float
    realized_pnl: float
    unrealized_pnl: float
    cumulative_fees: float
    drawdown: float = 0.0


@dataclass
class BacktestResult:
    """Comprehensive result container for a completed backtest execution."""

    backtest_id: str
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    symbols: list[str]
    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    expectancy: float
    trade_count: int
    avg_trade_pnl: float
    total_fees: float
    total_slippage: float
    total_spread: float
    equity_curve: list[EquityPoint]
    trade_ledger: list[TradeRecord]
    metrics: Any  # PerformanceMetrics
    audit_log: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    code_version: str = "v1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
