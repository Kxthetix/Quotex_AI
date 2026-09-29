# Paper Trading Sessions

## 1. PaperTradingSession Entity

Every paper-trading execution belongs to an immutable session record:

```python
@dataclass
class PaperTradingSession:
    session_id: str
    start_time: datetime
    end_time: datetime | None
    configuration_version: str
    model_versions: dict[str, str]
    feature_version: str
    signal_version: str
    initial_capital: float
    status: SessionStatus  # INITIALIZING, RUNNING, DEGRADED, PAUSED, STOPPED, FAILED
    shutdown_reason: str
    symbols: list[str]
    metadata: dict[str, Any]
```

## 2. Startup Sequence (16 Steps)

1. Load configuration and environment variables.
2. Validate configuration parameters.
3. Initialize structured logging and audit channels.
4. Connect to asynchronous persistence queue and database.
5. Lock approved model versions from manifest (refuses unapproved candidate models).
6. Verify feature version compatibility.
7. Verify signal engine version compatibility.
8. Initialize independent Risk Engine and circuit breakers.
9. Initialize Paper Execution Engine with strict safety invariants.
10. Initialize Prediction Ledger.
11. Initialize System Health Monitor and Latency Tracker.
12. Establish live market data connection (WebSocket).
13. Wait for valid market state and initial candle accumulation.
14. Enable live feature computation.
15. Enable live signal generation.
16. Enable simulated paper order execution.

## 3. Graceful Shutdown Sequence

1. Halt new signal generation and new order submissions.
2. Complete in-flight order executions safely.
3. Drain and flush pending prediction evaluations.
4. Save session recovery checkpoint to disk.
5. Produce final Session Report JSON.
6. Flush asynchronous persistence worker.
7. Close network and WebSocket connections.
8. Record session stop monitoring event.
