"""Data Quality Engine.

Implements rigorous checks:
- Timestamp sanity (clock drift / future / excessive age) -> INVALID_TIMESTAMP
- Duplicate event/trade detection -> DUPLICATE_EVENT
- OHLC consistency and non-negative volume -> CANDLE_OHLC_INCONSISTENCY, INVALID_VOLUME
- Outlier price jump detection -> PRICE_OUTLIER
- Out-of-order event arrival -> OUT_OF_ORDER_EVENT
- Missing candle gap detection -> MISSING_INTERVAL
- Stale data detection -> STALE_DATA
- Data-Quality Circuit Breaker (halts live decisions upon corruption or stale data)
- Comprehensive statistics and incident auditing
"""

from collections import defaultdict, deque
from datetime import UTC, datetime

from data.ingestion.models import Candle, MarketEvent, NormalizedTrade
from data.validation.models import (
    DataQualityStats,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from monitoring.logging import get_logger

logger = get_logger(__name__)

KNOWN_EVENT_TYPES = {
    "trade",
    "aggTrade",
    "kline",
    "depth",
    "depthUpdate",
    "bookTicker",
    "ticker",
    "24hrTicker",
    "markPriceUpdate",
}


class DataQualityEngine:
    """Enterprise Data Quality Engine with automated circuit breakers and statistics tracking."""

    def __init__(
        self,
        max_clock_drift_seconds: float = 5.0,
        max_price_change_pct: float = 20.0,
        stale_data_timeout_seconds: float = 15.0,
        dedup_cache_size: int = 100_000,
        max_consecutive_errors: int = 5,
        max_incident_history: int = 1_000,
    ) -> None:
        self.max_clock_drift_seconds = max_clock_drift_seconds
        self.max_price_change_pct = max_price_change_pct
        self.stale_data_timeout_seconds = stale_data_timeout_seconds
        self.max_consecutive_errors = max_consecutive_errors

        # Deduplication caches: separate for raw events and normalized trades
        self._raw_dedup_set: set[str] = set()
        self._raw_dedup_queue: deque[str] = deque(maxlen=dedup_cache_size)
        self._dedup_set: set[str] = set()
        self._dedup_queue: deque[str] = deque(maxlen=dedup_cache_size)

        # Market state tracking
        self._last_price: dict[str, float] = {}
        self._last_event_time: dict[str, datetime] = {}
        self._last_receive_time: dict[str, datetime] = {}
        self._last_candle_end_time: dict[str, datetime] = {}
        self._last_successful_candle_time: dict[str, datetime] = {}

        # Circuit breaker state
        self._consecutive_errors: int = 0
        self._is_breaker_tripped: bool = False
        self._breaker_reason: str | None = None

        # Telemetry & Auditable Incident Log
        self._recent_incidents: deque[ValidationIssue] = deque(maxlen=max_incident_history)
        self._reconnection_counts: dict[str, int] = defaultdict(int)
        self._stats: dict[str, DataQualityStats] = defaultdict(DataQualityStats)

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

    def record_reconnection(self, symbol: str = "ALL") -> None:
        """Record an ingestion reconnection incident."""
        self._reconnection_counts[symbol] += 1
        stats = self._get_or_create_stats(symbol)
        stats.reconnection_count += 1

    def validate_raw_event(self, event: MarketEvent) -> ValidationResult:
        """Validate raw incoming market event."""
        issues: list[ValidationIssue] = []
        now = datetime.now(UTC)
        symbol = event.symbol.upper()

        stats = self._get_or_create_stats(symbol)
        stats.total_events_received += 1

        # 1. Symbol validation
        if not symbol or not symbol.strip():
            issues.append(
                ValidationIssue(
                    check_name="missing_symbol",
                    category=ValidationCategory.MISSING_FIELD,
                    severity=ValidationSeverity.CRITICAL,
                    message="MarketEvent is missing a valid symbol",
                    symbol=symbol,
                )
            )

        # 2. Payload presence & structure
        if not event.payload or not isinstance(event.payload, dict):
            issues.append(
                ValidationIssue(
                    check_name="empty_or_malformed_payload",
                    category=ValidationCategory.MALFORMED_PAYLOAD,
                    severity=ValidationSeverity.CRITICAL,
                    message="MarketEvent payload is empty or not a dictionary",
                    symbol=symbol,
                )
            )
        else:
            # 3. Payload specific field checks (for trade/aggTrade)
            # Price check if price key is present
            if "p" in event.payload or "price" in event.payload:
                raw_p = (
                    event.payload.get("p") if "p" in event.payload else event.payload.get("price")
                )
                try:
                    p_val = float(raw_p)
                    if p_val <= 0:
                        issues.append(
                            ValidationIssue(
                                check_name="invalid_price_value",
                                category=ValidationCategory.INVALID_PRICE,
                                severity=ValidationSeverity.ERROR,
                                message=f"Non-positive price: {raw_p}",
                                symbol=symbol,
                            )
                        )
                except (ValueError, TypeError):
                    issues.append(
                        ValidationIssue(
                            check_name="non_numeric_price",
                            category=ValidationCategory.INVALID_PRICE,
                            severity=ValidationSeverity.ERROR,
                            message=f"Non-numeric price: {raw_p}",
                            symbol=symbol,
                        )
                    )

            # Volume check if volume key is present
            raw_q = (
                event.payload.get("q")
                if "q" in event.payload
                else (event.payload.get("qty") or event.payload.get("quantity"))
            )
            if raw_q is not None:
                try:
                    q_val = float(raw_q)
                    if q_val <= 0:
                        issues.append(
                            ValidationIssue(
                                check_name="invalid_volume_value",
                                category=ValidationCategory.INVALID_VOLUME,
                                severity=ValidationSeverity.ERROR,
                                message=f"Non-positive volume: {raw_q}",
                                symbol=symbol,
                            )
                        )
                except (ValueError, TypeError):
                    issues.append(
                        ValidationIssue(
                            check_name="non_numeric_volume",
                            category=ValidationCategory.INVALID_VOLUME,
                            severity=ValidationSeverity.ERROR,
                            message=f"Non-numeric volume: {raw_q}",
                            symbol=symbol,
                        )
                    )

            # Duplicate raw event check using _raw_dedup_set
            trade_id = str(
                event.payload.get("t")
                or event.payload.get("a")
                or event.payload.get("trade_id")
                or ""
            )
            if trade_id:
                raw_key = f"{event.source}:{symbol}:{trade_id}"
                if raw_key in self._raw_dedup_set:
                    issues.append(
                        ValidationIssue(
                            check_name="duplicate_raw_event",
                            category=ValidationCategory.DUPLICATE_EVENT,
                            severity=ValidationSeverity.ERROR,
                            message=f"Duplicate raw event ID: {trade_id}",
                            symbol=symbol,
                            details={"trade_id": trade_id},
                        )
                    )
                    stats.duplicate_events += 1
                else:
                    if len(self._raw_dedup_queue) >= self._raw_dedup_queue.maxlen:  # type: ignore[operator]
                        evicted = self._raw_dedup_queue.popleft()
                        self._raw_dedup_set.discard(evicted)
                    self._raw_dedup_queue.append(raw_key)
                    self._raw_dedup_set.add(raw_key)

        # 4. Event type recognition
        if event.event_type not in KNOWN_EVENT_TYPES and not event.event_type.startswith("kline"):
            issues.append(
                ValidationIssue(
                    check_name="unknown_event_type",
                    category=ValidationCategory.UNKNOWN_EVENT_TYPE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Event type '{event.event_type}' is unrecognized",
                    symbol=symbol,
                    details={"event_type": event.event_type},
                )
            )

        # 5. Timestamp validation (Check for future timestamps / clock drift)
        drift = (event.event_time - now).total_seconds()
        if drift > self.max_clock_drift_seconds:
            issues.append(
                ValidationIssue(
                    check_name="timestamp_future_drift",
                    category=ValidationCategory.INVALID_TIMESTAMP,
                    severity=ValidationSeverity.ERROR,
                    message=f"Event time {event.event_time} is {drift:.2f}s in the future (max allowed {self.max_clock_drift_seconds}s)",
                    symbol=symbol,
                    details={"drift_seconds": drift},
                )
            )

        # Update last receive time
        self._last_receive_time[symbol] = event.receive_time

        return self._evaluate_and_record(issues, symbol=symbol, event_time=event.event_time)

    def validate_trade(self, trade: NormalizedTrade) -> ValidationResult:
        """Validate normalized trade tick."""
        issues: list[ValidationIssue] = []
        symbol = trade.symbol.upper()
        stats = self._get_or_create_stats(symbol)

        # 1. Required trade ID
        if not trade.trade_id or not trade.trade_id.strip():
            issues.append(
                ValidationIssue(
                    check_name="missing_trade_id",
                    category=ValidationCategory.MISSING_FIELD,
                    severity=ValidationSeverity.ERROR,
                    message="Trade ID is missing or empty",
                    symbol=symbol,
                )
            )

        # 2. Deduplication check
        dedup_key = f"{trade.source}:{symbol}:{trade.trade_id}"
        if dedup_key in self._dedup_set:
            issues.append(
                ValidationIssue(
                    check_name="duplicate_trade",
                    category=ValidationCategory.DUPLICATE_EVENT,
                    severity=ValidationSeverity.ERROR,
                    message=f"Duplicate trade detected: {trade.trade_id}",
                    symbol=symbol,
                    details={"trade_id": trade.trade_id},
                )
            )
            stats.duplicate_events += 1
        else:
            if len(self._dedup_queue) >= self._dedup_queue.maxlen:  # type: ignore[operator]
                evicted = self._dedup_queue.popleft()
                self._dedup_set.discard(evicted)
            self._dedup_queue.append(dedup_key)
            self._dedup_set.add(dedup_key)

        # 3. Sanity on values
        if trade.price <= 0:
            issues.append(
                ValidationIssue(
                    check_name="non_positive_price",
                    category=ValidationCategory.INVALID_PRICE,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Invalid trade price: {trade.price}",
                    symbol=symbol,
                )
            )

        if trade.quantity <= 0:
            issues.append(
                ValidationIssue(
                    check_name="non_positive_quantity",
                    category=ValidationCategory.INVALID_VOLUME,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Invalid trade quantity: {trade.quantity}",
                    symbol=symbol,
                )
            )

        # 3. Out-of-order event check (event timestamp older than previously observed event)
        if symbol in self._last_event_time:
            last_ts = self._last_event_time[symbol]
            if trade.timestamp < last_ts:
                time_diff = (last_ts - trade.timestamp).total_seconds()
                issues.append(
                    ValidationIssue(
                        check_name="out_of_order_trade",
                        category=ValidationCategory.OUT_OF_ORDER_EVENT,
                        severity=ValidationSeverity.WARNING,
                        message=f"Trade arrived out of order: {trade.timestamp} is {time_diff:.3f}s older than previous event {last_ts}",
                        symbol=symbol,
                        details={
                            "last_ts": last_ts.isoformat(),
                            "trade_ts": trade.timestamp.isoformat(),
                        },
                    )
                )
                stats.out_of_order_events += 1

        # 4. Outlier price jump check
        if symbol in self._last_price and trade.price > 0:
            last = self._last_price[symbol]
            pct_change = abs(trade.price - last) / last * 100.0
            if pct_change > self.max_price_change_pct:
                issues.append(
                    ValidationIssue(
                        check_name="price_outlier_jump",
                        category=ValidationCategory.PRICE_OUTLIER,
                        severity=ValidationSeverity.ERROR,
                        message=f"Price jumped by {pct_change:.2f}% from {last} to {trade.price}",
                        symbol=symbol,
                        details={
                            "pct_change": pct_change,
                            "previous_price": last,
                            "new_price": trade.price,
                        },
                    )
                )

        # Update last price and event time if valid
        if not any(i.severity == ValidationSeverity.CRITICAL for i in issues):
            self._last_price[symbol] = trade.price
            self._last_receive_time[symbol] = datetime.now(UTC)
            if (
                symbol not in self._last_event_time
                or trade.timestamp > self._last_event_time[symbol]
            ):
                self._last_event_time[symbol] = trade.timestamp
            stats.last_successful_event = trade.timestamp

        return self._evaluate_and_record(issues, symbol=symbol, event_time=trade.timestamp)

    def validate_candle(self, candle: Candle) -> ValidationResult:
        """Validate candle OHLC consistency, volume, and continuity."""
        issues: list[ValidationIssue] = []
        symbol = candle.symbol.upper()
        stats = self._get_or_create_stats(symbol)

        # 1. OHLC consistency
        if candle.high < candle.open:
            issues.append(
                ValidationIssue(
                    check_name="ohlc_high_less_than_open",
                    category=ValidationCategory.CANDLE_OHLC_INCONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"High ({candle.high}) is less than Open ({candle.open})",
                    symbol=symbol,
                )
            )
        if candle.high < candle.close:
            issues.append(
                ValidationIssue(
                    check_name="ohlc_high_less_than_close",
                    category=ValidationCategory.CANDLE_OHLC_INCONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"High ({candle.high}) is less than Close ({candle.close})",
                    symbol=symbol,
                )
            )
        if candle.low > candle.open:
            issues.append(
                ValidationIssue(
                    check_name="ohlc_low_greater_than_open",
                    category=ValidationCategory.CANDLE_OHLC_INCONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Low ({candle.low}) is greater than Open ({candle.open})",
                    symbol=symbol,
                )
            )
        if candle.low > candle.close:
            issues.append(
                ValidationIssue(
                    check_name="ohlc_low_greater_than_close",
                    category=ValidationCategory.CANDLE_OHLC_INCONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Low ({candle.low}) is greater than Close ({candle.close})",
                    symbol=symbol,
                )
            )
        if candle.low > candle.high:
            issues.append(
                ValidationIssue(
                    check_name="ohlc_low_greater_than_high",
                    category=ValidationCategory.CANDLE_OHLC_INCONSISTENCY,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Low ({candle.low}) is greater than High ({candle.high})",
                    symbol=symbol,
                )
            )

        # 2. Volume validation
        if candle.volume < 0:
            issues.append(
                ValidationIssue(
                    check_name="negative_volume",
                    category=ValidationCategory.INVALID_VOLUME,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Negative volume: {candle.volume}",
                    symbol=symbol,
                )
            )

        # 3. Time boundaries
        if candle.end_time <= candle.start_time:
            issues.append(
                ValidationIssue(
                    check_name="invalid_candle_time_range",
                    category=ValidationCategory.INVALID_TIMESTAMP,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"end_time ({candle.end_time}) <= start_time ({candle.start_time})",
                    symbol=symbol,
                )
            )

        # 4. Missing candle / gap detection
        candle_key = f"{symbol}:{candle.timeframe}"
        if candle_key in self._last_candle_end_time:
            expected_start = self._last_candle_end_time[candle_key]
            if candle.start_time > expected_start:
                gap_seconds = (candle.start_time - expected_start).total_seconds()
                issues.append(
                    ValidationIssue(
                        check_name="missing_candle_gap",
                        category=ValidationCategory.MISSING_INTERVAL,
                        severity=ValidationSeverity.WARNING,
                        message=f"Candle sequence gap detected: {gap_seconds:.1f}s missing between {expected_start} and {candle.start_time}",
                        symbol=symbol,
                        details={
                            "expected_start": expected_start.isoformat(),
                            "gap_seconds": gap_seconds,
                        },
                    )
                )
                stats.missing_intervals += 1

        if not any(i.severity == ValidationSeverity.CRITICAL for i in issues):
            self._last_candle_end_time[candle_key] = candle.end_time
            self._last_successful_candle_time[symbol] = candle.end_time
            stats.last_successful_candle = candle.end_time

        return self._evaluate_and_record(issues, symbol=symbol, event_time=candle.end_time)

    def check_stale_data(self, symbol: str) -> bool:
        """Check if market data for a symbol is stale."""
        now = datetime.now(UTC)
        clean_sym = symbol.upper()
        times = [
            t
            for t in (self._last_receive_time.get(clean_sym), self._last_event_time.get(clean_sym))
            if t is not None
        ]
        if not times:
            return False

        last_time = max(times)
        elapsed = (now - last_time).total_seconds()
        if elapsed > self.stale_data_timeout_seconds:
            issue = ValidationIssue(
                check_name="stale_data_timeout",
                category=ValidationCategory.STALE_DATA,
                severity=ValidationSeverity.CRITICAL,
                message=f"Market data stale for {clean_sym}: no events for {elapsed:.1f}s (max {self.stale_data_timeout_seconds}s)",
                symbol=clean_sym,
                details={"elapsed_seconds": elapsed},
            )
            self._record_incident(issue)
            stats = self._get_or_create_stats(clean_sym)
            stats.stale_data_incidents += 1
            self.trip_breaker(issue.message)
            return True
        return False

    def check_feed_staleness(self, symbol: str) -> bool:
        """Alias for check_stale_data."""
        return self.check_stale_data(symbol)

    def get_stats(self, symbol: str | None = None) -> DataQualityStats:
        """Return data quality statistics for a symbol or aggregate across all symbols."""
        if symbol and symbol != "ALL":
            return self._get_or_create_stats(symbol)

        # Aggregate across all tracked symbols
        agg = DataQualityStats(provider="binance", symbol="ALL")
        for s in self._stats.values():
            agg.total_events_received += s.total_events_received
            agg.valid_events += s.valid_events
            agg.invalid_events += s.invalid_events
            agg.duplicate_events += s.duplicate_events
            agg.out_of_order_events += s.out_of_order_events
            agg.missing_intervals += s.missing_intervals
            agg.stale_data_incidents += s.stale_data_incidents
            agg.reconnection_count += s.reconnection_count
            agg.processing_failures += s.processing_failures
            for cat, count in s.category_counts.items():
                agg.category_counts[cat] = agg.category_counts.get(cat, 0) + count

            if s.last_successful_event and (
                agg.last_successful_event is None
                or s.last_successful_event > agg.last_successful_event
            ):
                agg.last_successful_event = s.last_successful_event
            if s.last_successful_candle and (
                agg.last_successful_candle is None
                or s.last_successful_candle > agg.last_successful_candle
            ):
                agg.last_successful_candle = s.last_successful_candle

        return agg

    def get_recent_incidents(self) -> list[ValidationIssue]:
        """Return auditable list of recent data quality issues."""
        return list(self._recent_incidents)

    def _get_or_create_stats(self, symbol: str) -> DataQualityStats:
        sym = symbol.upper()
        if sym not in self._stats:
            self._stats[sym] = DataQualityStats(symbol=sym)
        return self._stats[sym]

    def _record_incident(self, issue: ValidationIssue) -> None:
        self._recent_incidents.append(issue)
        stats = self._get_or_create_stats(issue.symbol or "ALL")
        stats.category_counts[issue.category.value] = (
            stats.category_counts.get(issue.category.value, 0) + 1
        )

    def _evaluate_and_record(
        self,
        issues: list[ValidationIssue],
        symbol: str | None = None,
        event_time: datetime | None = None,
    ) -> ValidationResult:
        sym = (symbol or "ALL").upper()
        stats = self._get_or_create_stats(sym)

        has_error = any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) for i in issues
        )

        for issue in issues:
            self._record_incident(issue)

        if has_error:
            stats.invalid_events += 1
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.max_consecutive_errors:
                self.trip_breaker(
                    f"Exceeded max consecutive data quality errors ({self._consecutive_errors})"
                )
        else:
            stats.valid_events += 1
            self._consecutive_errors = max(0, self._consecutive_errors - 1)

        return ValidationResult(is_valid=not has_error, issues=issues)
