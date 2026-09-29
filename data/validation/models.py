"""Validation models and result types for the Data Quality Engine."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ValidationSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationCategory(StrEnum):
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_VOLUME = "INVALID_VOLUME"
    UNKNOWN_EVENT_TYPE = "UNKNOWN_EVENT_TYPE"
    MALFORMED_PAYLOAD = "MALFORMED_PAYLOAD"
    DUPLICATE_EVENT = "DUPLICATE_EVENT"
    OUT_OF_ORDER_EVENT = "OUT_OF_ORDER_EVENT"
    PRICE_OUTLIER = "PRICE_OUTLIER"
    CANDLE_OHLC_INCONSISTENCY = "CANDLE_OHLC_INCONSISTENCY"
    MISSING_INTERVAL = "MISSING_INTERVAL"
    STALE_DATA = "STALE_DATA"


class ValidationIssue(BaseModel):
    """An individual data quality or validation failure."""

    check_name: str
    category: ValidationCategory = ValidationCategory.MALFORMED_PAYLOAD
    severity: ValidationSeverity
    message: str
    symbol: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Comprehensive validation outcome."""

    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)

    @property
    def has_error(self) -> bool:
        return any(
            issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for issue in self.issues
        )


class DataQualityStats(BaseModel):
    """Aggregated data quality telemetry and metrics."""

    provider: str = "binance"
    symbol: str = "ALL"
    total_events_received: int = 0
    valid_events: int = 0
    invalid_events: int = 0
    duplicate_events: int = 0
    out_of_order_events: int = 0
    missing_intervals: int = 0
    stale_data_incidents: int = 0
    reconnection_count: int = 0
    last_successful_event: datetime | None = None
    last_successful_candle: datetime | None = None
    processing_failures: int = 0
    category_counts: dict[str, int] = Field(default_factory=dict)
