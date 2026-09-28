"""Data Quality Engine.

Implements rigorous checks:
- Timestamp sanity (clock drift / future / excessive age)
- Duplicate event/trade detection
- OHLC consistency and non-negative volume
- Outlier price jump detection
- Missing candle gap detection
- Stale data detection
- Data-Quality Circuit Breaker (halts live decisions upon corruption or stale data)
"""

from collections import deque
from datetime import UTC, datetime

from data.ingestion.models import Candle, MarketEvent, NormalizedTrade
from data.validation.models import ValidationIssue, ValidationResult, ValidationSeverity
from monitoring.logging import get_logger

logger = get_logger(__name__)


class DataQualityEngine:
    """Enterprise Data Quality Engine with automated circuit breakers."""

    def __init__(
        self,
        max_clock_drift_seconds: float = 5.0,
        max_price_change_pct: float = 20.0,
        stale_data_timeout_seconds: float = 15.0,
        dedup_cache_size: int = 100_000,
        max_consecutive_errors: int = 5,
    ) -> None:
        self.max_clock_drift_seconds = max_clock_drift_seconds
        self.max_price_change_pct = max_price_change_pct
        self.stale_data_timeout_seconds = stale_data_timeout_seconds
        self.max_consecutive_errors = max_consecutive_errors

        # Deduplication caches: symbol -> set of IDs + FIFO queue for eviction
        self._dedup_set: set[str] = set()
        self._dedup_queue: deque[str] = deque(maxlen=dedup_cache_size)

        # Market state tracking
        self._last_price: dict[str, float] = {}
        self._last_event_time: dict[str, datetime] = {}
        self._last_candle_end_time: dict[str, datetime] = {}

        # Circuit breaker state
        self._consecutive_errors: int = 0
        self._is_breaker_tripped: bool = False
        self._breaker_reason: str | None = None

    @property
    def is_breaker_tripped(self) -> bool:
        """Returns True if data quality breaker is active, halting trading."""
        return self._is_breaker_tripped

    @property
    def breaker_reason(self) -> str | None:
        return self._breaker_reason

    def reset_breaker(self) -> None:
        """Manually or programmatically reset the circuit breaker after recovery."""
        self._is_breaker_tripped = False
        self._breaker_reason = None
        self._consecutive_errors = 0
        logger.info("data_quality_breaker_reset")

    def trip_breaker(self, reason: str) -> None:
        """Trip circuit breaker to halt downstream intelligence."""
        self._is_breaker_tripped = True
        self._breaker_reason = reason
        logger.error("data_quality_breaker_tripped", reason=reason)

    def validate_raw_event(self, event: MarketEvent) -> ValidationResult:
        """Validate raw incoming market event."""
        issues: list[ValidationIssue] = []
        now = datetime.now(UTC)

        # 1. Timestamp validation (Check for future timestamps / clock drift)
        drift = (event.event_time - now).total_seconds()
        if drift > self.max_clock_drift_seconds:
            issues.append(
                ValidationIssue(
                    check_name="timestamp_future_drift",
                    severity=ValidationSeverity.ERROR,
                    message=f"Event time {event.event_time} is {drift:.2f}s in the future (max allowed {self.max_clock_drift_seconds}s)",
                    symbol=event.symbol,
                    details={"drift_seconds": drift},
                )
            )

        # 2. Payload presence
        if not event.payload:
            issues.append(
                ValidationIssue(
                    check_name="empty_payload",
                    severity=ValidationSeverity.CRITICAL,
                    message="MarketEvent payload is empty",
                    symbol=event.symbol,
                )
            )

        return self._evaluate_issues(issues, symbol=event.symbol, event_time=event.receive_time)

    def validate_trade(self, trade: NormalizedTrade) -> ValidationResult:
        """Validate normalized trade tick."""
        issues: list[ValidationIssue] = []

        # 1. Deduplication check
        dedup_key = f"{trade.source}:{trade.symbol}:{trade.trade_id}"
        if dedup_key in self._dedup_set:
            issues.append(
                ValidationIssue(
                    check_name="duplicate_trade",
                    severity=ValidationSeverity.WARNING,
                    message=f"Duplicate trade detected: {trade.trade_id}",
                    symbol=trade.symbol,
                    details={"trade_id": trade.trade_id},
                )
            )
        else:
            if len(self._dedup_queue) >= self._dedup_queue.maxlen:  # type: ignore[operator]
                evicted = self._dedup_queue.popleft()
                self._dedup_set.discard(evicted)
            self._dedup_queue.append(dedup_key)
            self._dedup_set.add(dedup_key)

        # 2. Sanity on values
        if trade.price <= 0:
            issues.append(
                ValidationIssue(
                    check_name="non_positive_price",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Invalid trade price: {trade.price}",
                    symbol=trade.symbol,
                )
            )

        if trade.quantity <= 0:
            issues.append(
                ValidationIssue(
                    check_name="non_positive_quantity",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Invalid trade quantity: {trade.quantity}",
                    symbol=trade.symbol,
                )
            )

        # 3. Outlier price jump check
        if trade.symbol in self._last_price and trade.price > 0:
            last = self._last_price[trade.symbol]
            pct_change = abs(trade.price - last) / last * 100.0
            if pct_change > self.max_price_change_pct:
                issues.append(
                    ValidationIssue(
                        check_name="price_outlier_jump",
                        severity=ValidationSeverity.ERROR,
                        message=f"Price jumped by {pct_change:.2f}% from {last} to {trade.price}",
                        symbol=trade.symbol,
                        details={
                            "pct_change": pct_change,
                            "previous_price": last,
                            "new_price": trade.price,
                        },
                    )
                )

        # Update last state on valid price
        if not any(i.severity == ValidationSeverity.CRITICAL for i in issues):
            self._last_price[trade.symbol] = trade.price
            self._last_event_time[trade.symbol] = trade.timestamp

        return self._evaluate_issues(issues, symbol=trade.symbol, event_time=trade.timestamp)

    def validate_candle(self, candle: Candle) -> ValidationResult:
        """Validate candle OHLC consistency, volume, and continuity."""
        issues: list[ValidationIssue] = []

        # 1. OHLC consistency
        if candle.high < candle.open:
            issues.append(
                ValidationIssue(
                    check_name="ohlc_high_less_than_open",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"High ({candle.high}) is less than Open ({candle.open})",
                    symbol=candle.symbol,
                )
            )
        if candle.high < candle.close:
            issues.append(
                ValidationIssue(
                    check_name="ohlc_high_less_than_close",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"High ({candle.high}) is less than Close ({candle.close})",
                    symbol=candle.symbol,
                )
            )
        if candle.low > candle.open:
            issues.append(
                ValidationIssue(
                    check_name="ohlc_low_greater_than_open",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Low ({candle.low}) is greater than Open ({candle.open})",
                    symbol=candle.symbol,
                )
            )
        if candle.low > candle.close:
            issues.append(
                ValidationIssue(
                    check_name="ohlc_low_greater_than_close",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Low ({candle.low}) is greater than Close ({candle.close})",
                    symbol=candle.symbol,
                )
            )
        if candle.low > candle.high:
            issues.append(
                ValidationIssue(
                    check_name="ohlc_low_greater_than_high",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Low ({candle.low}) is greater than High ({candle.high})",
                    symbol=candle.symbol,
                )
            )

        # 2. Volume validation
        if candle.volume < 0:
            issues.append(
                ValidationIssue(
                    check_name="negative_volume",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Negative volume: {candle.volume}",
                    symbol=candle.symbol,
                )
            )

        # 3. Time boundaries
        if candle.end_time <= candle.start_time:
            issues.append(
                ValidationIssue(
                    check_name="invalid_candle_time_range",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"end_time ({candle.end_time}) <= start_time ({candle.start_time})",
                    symbol=candle.symbol,
                )
            )

        # 4. Missing candle / gap detection
        candle_key = f"{candle.symbol}:{candle.timeframe}"
        if candle_key in self._last_candle_end_time:
            expected_start = self._last_candle_end_time[candle_key]
            if candle.start_time > expected_start:
                gap_seconds = (candle.start_time - expected_start).total_seconds()
                issues.append(
                    ValidationIssue(
                        check_name="missing_candle_gap",
                        severity=ValidationSeverity.WARNING,
                        message=f"Candle sequence gap detected: {gap_seconds:.1f}s missing between {expected_start} and {candle.start_time}",
                        symbol=candle.symbol,
                        details={
                            "expected_start": expected_start.isoformat(),
                            "gap_seconds": gap_seconds,
                        },
                    )
                )

        if not any(i.severity == ValidationSeverity.CRITICAL for i in issues):
            self._last_candle_end_time[candle_key] = candle.end_time

        return self._evaluate_issues(issues, symbol=candle.symbol, event_time=candle.end_time)

    def check_stale_data(self, symbol: str) -> bool:
        """Check if market data for a symbol is stale."""
        now = datetime.now(UTC)
        last_time = self._last_event_time.get(symbol)
        if last_time is None:
            return False  # No data received yet, not yet marked stale

        elapsed = (now - last_time).total_seconds()
        if elapsed > self.stale_data_timeout_seconds:
            self.trip_breaker(f"Market data stale for {symbol}: no events for {elapsed:.1f}s")
            return True
        return False

    def _evaluate_issues(
        self,
        issues: list[ValidationIssue],
        symbol: str | None = None,
        event_time: datetime | None = None,
    ) -> ValidationResult:
        has_error = any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) for i in issues
        )

        if has_error:
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.max_consecutive_errors:
                self.trip_breaker(
                    f"Exceeded max consecutive data quality errors ({self._consecutive_errors})"
                )
        else:
            self._consecutive_errors = max(0, self._consecutive_errors - 1)

        return ValidationResult(is_valid=not has_error, issues=issues)
