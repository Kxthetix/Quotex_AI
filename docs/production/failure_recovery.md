# Failure Classification and Recovery Engine

## 1. Failure Classification

Failures are systematically classified into five operational categories:

1. **TRANSIENT**:
   - Short network timeout, momentary DNS resolution lag.
   - Action: Retry with exponential backoff and randomized jitter.
2. **RECOVERABLE**:
   - Socket connection drop, temporary exchange disconnect.
   - Action: Reconnect protocol, authenticate, detect missing candle interval, backfill, and resume.
3. **DEGRADED**:
   - Telemetry lag, secondary cache offline.
   - Action: Continue core trading pipeline using direct computation or local fallbacks; log warnings.
4. **CRITICAL**:
   - Stale market data (> 65s), model artifact mismatch, risk circuit breaker trip.
   - Action: Halt new trading decisions immediately; reject pending orders; wait for operator review.
5. **FATAL**:
   - Corrupted persistent trade ledger, unrecoverable schema mismatch.
   - Action: Trigger graceful shutdown sequence and persist all state safely.

## 2. Retry Policy

- Configurable maximum retry attempts (default: 4).
- Exponential backoff with multiplier (default: 2.0).
- Randomized full jitter to prevent thundering herd spikes.
- Configurable maximum delay ceiling (default: 10.0s).

## 3. Persistence Buffer (Database Outage Protection)

- Thread-safe bounded memory queue (`max_buffer_size = 5000`).
- When the database is unreachable, records are buffered in memory without blocking execution.
- If the buffer reaches capacity, the oldest records are dropped and an overflow metric is incremented.
- Upon database reconnection, `flush()` writes all buffered records in a transactional batch.
