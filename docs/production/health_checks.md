# Service Health Checks and Readiness Model

## 1. Service Health States

Every critical component exposes one of five discrete health states:

| Health State | Operational Semantics | System Reaction |
| :--- | :--- | :--- |
| `HEALTHY` | Component functioning nominally within SLA. | Unrestricted live operations. |
| `DEGRADED` | Non-critical component impaired (e.g. telemetry lag). | Operates in fallback / cached mode. |
| `BLOCKED` | Safety breaker tripped (e.g. stale data, circuit breaker). | New decisions halted; existing positions held. |
| `FAILED` | Process error, connection loss, or unhandled exception. | Immediate recovery triggered; decisions blocked. |
| `UNKNOWN` | No recent probe data received within timeout window. | Treated as DEGRADED or FAILED. |

## 2. Probe Classifications

Probes are split into three distinct check types (Sections 5 & 6):

1. **Liveness Checks**:
   - Question: *Is the service process alive and responding to basic heartbeats?*
   - Example: Event loop responsive, process PID active.
2. **Readiness Checks**:
   - Question: *Can the service safely process new orders and signals?*
   - Example: Model weights loaded in memory, feature cache warm, risk limits valid.
3. **Dependency Checks**:
   - Question: *Are downstream and external infrastructure services reachable?*
   - Example: PostgreSQL connection pool active, Redis server reachable, exchange WebSocket online.

## 3. System Readiness Gate

The `ReadinessCoordinator` evaluates all registered probes. The master readiness status is:
- `is_ready = True` ONLY when all mandatory dependencies report `HEALTHY`.
- If any mandatory dependency fails (e.g. Database unreachable or Market Data stale):
  `is_ready = False` and the system is BLOCKED from placing orders or evaluating trades.
