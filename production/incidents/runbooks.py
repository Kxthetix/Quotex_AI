"""Operational Incident Runbooks (Section 48).

Standard Operating Procedures (SOP) for all 12 mandatory failure modes:
1. WebSocket outage
2. Stale market data
3. Database outage
4. Redis outage
5. Model failure
6. Model artifact corruption
7. High latency
8. Disk full
9. Backup failure
10. Circuit breaker
11. Unexpected model behavior
12. Security incident

Each runbook specifies:
1. Detection
2. Immediate safety action
3. Diagnosis
4. Recovery
5. Validation
6. Return-to-service criteria
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Runbook:
    id: str
    title: str
    component: str
    detection: str
    immediate_safety_action: str
    diagnosis: list[str]
    recovery: list[str]
    validation: list[str]
    return_to_service_criteria: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RUNBOOKS: dict[str, Runbook] = {
    "websocket_outage": Runbook(
        id="RB-01",
        title="WebSocket Market Data Disconnection Outage",
        component="MarketData",
        detection="Heartbeat timeout > 15s or abrupt socket close event.",
        immediate_safety_action="Halt order placement; mark market data status DISCONNECTED; block new signal decisions.",
        diagnosis=[
            "Inspect network connectivity to exchange endpoint.",
            "Verify exchange API status page and maintenance announcements.",
            "Check DNS resolution and local interface bandwidth.",
        ],
        recovery=[
            "Initiate exponential backoff reconnect policy (initial 1.0s, max 30s).",
            "Re-authenticate and resubscribe to candle/ticker channels.",
            "Trigger MarketDataGapRecovery to backfill missing candle intervals.",
        ],
        validation=[
            "Verify gap recovery completed without unrecoverable time skips.",
            "Confirm newly arrived ticks match sequential timestamps.",
        ],
        return_to_service_criteria=[
            "Socket connection stable for >= 60 seconds.",
            "Gap reconciliation 100% complete.",
            "Readiness coordinator approves MarketData dependency.",
        ],
    ),
    "stale_market_data": Runbook(
        id="RB-02",
        title="Stale Market Data Detection",
        component="MarketData",
        detection="Event age or last closed candle age exceeds 65.0 seconds.",
        immediate_safety_action="Trip stale-data breaker; set TradingState to BLOCKED; veto pending orders.",
        diagnosis=[
            "Verify if feed publisher is hung or delayed.",
            "Check local clock drift against NTP servers.",
            "Examine message queue lag in market data ingestion pipeline.",
        ],
        recovery=[
            "Restart feed listener process if queue is clogged.",
            "Request REST backfill for lagging symbol timeframes.",
            "Re-synchronize system clock if drift > 500ms.",
        ],
        validation=[
            "Verify current candle timestamp is within 30s of real time.",
            "Confirm tick volume matches historical distribution.",
        ],
        return_to_service_criteria=[
            "At least 3 consecutive fresh candles ingested with latency < 5s.",
            "Data-quality breaker reset by Operator role.",
        ],
    ),
    "database_outage": Runbook(
        id="RB-03",
        title="PostgreSQL / TimescaleDB Outage",
        component="Database",
        detection="Connection pool timeout, DB connection refused, or query execution failure.",
        immediate_safety_action="Divert writes to bounded in-memory PersistenceBuffer; block new trading decisions if audit/ledger cannot be buffered.",
        diagnosis=[
            "Check database service process status and container health.",
            "Inspect database disk space and write-ahead log (WAL) volume.",
            "Verify DB connection credentials and connection pool saturation.",
        ],
        recovery=[
            "Restart PostgreSQL/TimescaleDB daemon if crashed.",
            "Scale connection pool if exhaustion occurred.",
            "Once reachable, flush bounded in-memory PersistenceBuffer in transactional batches.",
        ],
        validation=[
            "Verify SELECT 1 liveness and check schema integrity.",
            "Confirm PersistenceBuffer queue dropped to zero with zero lost records.",
        ],
        return_to_service_criteria=[
            "Database latency < 50ms on standard queries.",
            "All buffered ledger entries safely written and verified.",
            "Readiness check for database returns HEALTHY.",
        ],
    ),
    "redis_outage": Runbook(
        id="RB-04",
        title="Redis Cache / Event Bus Outage",
        component="Redis",
        detection="RedisConnectionError, timeout on pub/sub channel, or ping failure.",
        immediate_safety_action="Switch feature cache to degraded direct-computation mode; pause asynchronous fan-out.",
        diagnosis=[
            "Check Redis server process and memory limits (OOM killer).",
            "Verify network reachability and authentication.",
        ],
        recovery=[
            "Restart Redis instance.",
            "Verify persistence file (AOF / RDB) sanity.",
            "Reconnect client connection pool with retry backoff.",
        ],
        validation=[
            "Execute Redis PING command.",
            "Verify pub/sub broadcast and subscriber reception.",
        ],
        return_to_service_criteria=[
            "Redis responds with PONG in < 5ms.",
            "Feature caching restored without errors.",
        ],
    ),
    "model_failure": Runbook(
        id="RB-05",
        title="Model Service Inference Failure",
        component="ModelService",
        detection="Inference exception, output NaN/Inf, or prediction latency > 500ms.",
        immediate_safety_action="Block signal engine from executing trades; fall back to NO-TRADE default.",
        diagnosis=[
            "Inspect inference input feature vector for NaN, Inf, or schema drift.",
            "Check CPU/GPU memory saturation.",
            "Verify model file integrity and serialization format.",
        ],
        recovery=[
            "Reload active model artifact from verified disk archive.",
            "If issue persists, execute controlled rollback to previous model version via ModelRegistry.",
        ],
        validation=[
            "Pass synthetic validation tensor through inference pipeline.",
            "Verify output probabilities sum to 1.0 and pass sanity checks.",
        ],
        return_to_service_criteria=[
            "Model inference passes 100 validation samples without errors.",
            "Latency within SLA (< 50ms).",
        ],
    ),
    "model_artifact_corruption": Runbook(
        id="RB-06",
        title="Model Artifact Corruption / Checksum Mismatch",
        component="ModelRegistry",
        detection="SHA-256 hash of loaded model does not match registry manifest.",
        immediate_safety_action="Refuse to load corrupted artifact; block startup or inference immediately.",
        diagnosis=[
            "Check disk sectors and filesystem integrity.",
            "Examine deployment logs for truncated or incomplete file copy.",
        ],
        recovery=[
            "Retrieve verified backup copy of model artifact from BackupManager archive.",
            "Re-verify restored file SHA-256 against ModelRegistry record.",
        ],
        validation=[
            "Confirm checksum matches registry manifest byte-for-byte.",
            "Verify successful test load in isolated sandbox.",
        ],
        return_to_service_criteria=[
            "Checksum validation passes 100%.",
            "Model successfully loaded and verified healthy.",
        ],
    ),
    "high_latency": Runbook(
        id="RB-07",
        title="High Pipeline Processing Latency",
        component="Pipeline",
        detection="End-to-end processing latency exceeds 250ms for >= 3 consecutive cycles.",
        immediate_safety_action="Rate-limit event processing; reject stale incoming ticks.",
        diagnosis=[
            "Profile latency breakdown: data ingestion, feature generation, inference, risk engine.",
            "Check CPU throttling, memory paging, and garbage collection pauses.",
        ],
        recovery=[
            "Purge redundant feature caches.",
            "Throttle non-essential background tasks (backups, audits).",
        ],
        validation=[
            "Measure p95 and p99 latency under simulated load.",
        ],
        return_to_service_criteria=[
            "End-to-end p95 latency returns below 50ms.",
            "No dropped event queues.",
        ],
    ),
    "disk_full": Runbook(
        id="RB-08",
        title="Disk Space Approaching Capacity",
        component="Infrastructure",
        detection="Free disk space falls below 10% (or < 5GB).",
        immediate_safety_action="Disable non-essential debug logging; pause non-critical backups.",
        diagnosis=[
            "Inspect large directories (logs/, raw data/, temp archives).",
            "Check for uncompressed old log files or stale backup zip files.",
        ],
        recovery=[
            "Enforce retention policy: archive and purge logs older than 30 days.",
            "Prune scratch files and temporary test artifacts.",
        ],
        validation=[
            "Verify disk utilization drops below 75%.",
        ],
        return_to_service_criteria=[
            "At least 20% free disk space available.",
            "Logging and monitoring functions operating normally.",
        ],
    ),
    "backup_failure": Runbook(
        id="RB-09",
        title="Scheduled Backup Execution / Verification Failure",
        component="BackupManager",
        detection="Backup process exited with non-zero code or BackupVerifier failed 5-point check.",
        immediate_safety_action="Fire WARNING alert to operational monitoring; retain previous successful backup.",
        diagnosis=[
            "Inspect backup log for file lock or read permissions errors.",
            "Check disk space on backup destination volume.",
            "Verify SHA-256 calculation error or archive truncation.",
        ],
        recovery=[
            "Clear failed temporary archive fragments.",
            "Re-run BackupManager for failed domain with verbose verification.",
        ],
        validation=[
            "Run BackupVerifier.verify_backup on new archive.",
            "Perform sandbox test restore via RestoreManager.",
        ],
        return_to_service_criteria=[
            "Backup archive verified valid and recoverable.",
            "Manifest recorded in durable backup catalog.",
        ],
    ),
    "circuit_breaker": Runbook(
        id="RB-10",
        title="Circuit Breaker Tripped (Risk or Data Quality)",
        component="RiskEngine",
        detection="Circuit breaker state transitions to TRIPPED.",
        immediate_safety_action="Block all order generation; cancel pending unfilled orders; maintain existing positions safely.",
        diagnosis=[
            "Identify trip trigger: max drawdown, daily loss limit, stale data, or rapid volatility.",
            "Inspect market conditions and recent trade ledger entries.",
        ],
        recovery=[
            "Investigate and resolve root cause.",
            "Require explicit OPERATOR or ADMIN role authorization to reset breaker.",
        ],
        validation=[
            "Verify risk metrics (drawdown, margin, exposure) are within safe thresholds.",
            "Confirm market data feed is fresh and healthy.",
        ],
        return_to_service_criteria=[
            "Authorized role issues signed breaker reset command.",
            "Audit record logged with justification.",
            "Readiness coordinator verifies full system readiness.",
        ],
    ),
    "unexpected_model_behavior": Runbook(
        id="RB-11",
        title="Unexpected Model Prediction Behavior / Severe Drift",
        component="LearningLoop",
        detection="High population stability index (PSI > 0.25) or anomalous prediction distribution.",
        immediate_safety_action="De-weight model signals; switch SignalEngine to conservative or NO-TRADE mode.",
        diagnosis=[
            "Compare current market regime features with training distribution.",
            "Check for sudden volatility shock or exchange data anomaly.",
        ],
        recovery=[
            "Initiate candidate retraining or rollback to previously verified benchmark model.",
            "Log model audit incident in ModelRegistry.",
        ],
        validation=[
            "Run prediction distribution checks on new data window.",
            "Verify calibration curve and Brier score.",
        ],
        return_to_service_criteria=[
            "Candidate model passes all 5 validation gates (Performance, Stability, Calibration, Drift, Stress).",
            "Explicit Model Promotion event recorded by Model Manager.",
        ],
    ),
    "security_incident": Runbook(
        id="RB-12",
        title="Security Incident / Unauthorized Access Attempt",
        component="Security",
        detection="Repeated AccessDeniedErrors, unauthorized configuration change attempt, or credential leak.",
        immediate_safety_action="Immediately revoke compromised credentials; lock administrative endpoints; isolate session.",
        diagnosis=[
            "Inspect audit trail for actor ID, IP address, and attempted actions.",
            "Check source repositories and logs for leaked secrets.",
        ],
        recovery=[
            "Rotate all API keys, database credentials, and session tokens.",
            "Apply patch to access control policies if misconfigured.",
        ],
        validation=[
            "Verify all services authenticate successfully with rotated credentials.",
            "Audit trail integrity verified via SHA-256 hash chain check.",
        ],
        return_to_service_criteria=[
            "All secrets rotated and verified.",
            "Security audit passes with zero critical findings.",
            "ADMIN sign-off recorded in incident log.",
        ],
    ),
}


class RunbookRegistry:
    """Provides lookup and inspection of all 12 production runbooks."""

    @classmethod
    def get_runbook(cls, runbook_id_or_name: str) -> Runbook | None:
        if runbook_id_or_name in RUNBOOKS:
            return RUNBOOKS[runbook_id_or_name]
        for rb in RUNBOOKS.values():
            if rb.id == runbook_id_or_name.upper():
                return rb
        return None

    @classmethod
    def list_runbooks(cls) -> list[Runbook]:
        return list(RUNBOOKS.values())
