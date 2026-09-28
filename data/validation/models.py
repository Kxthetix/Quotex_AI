"""Validation models and result types for the Data Quality Engine."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ValidationSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationIssue(BaseModel):
    """An individual data quality or validation failure."""

    check_name: str
    severity: ValidationSeverity
    message: str
    symbol: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict = Field(default_factory=dict)


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
