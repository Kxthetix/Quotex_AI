# Role-Based Access Control (RBAC)

## 1. Roles and Capabilities

Access is governed strictly through Role-Based Access Control (RBAC):

| Role | Permitted Actions | Prohibited Actions |
| :--- | :--- | :--- |
| **`READ_ONLY_MONITOR`** | View metrics, stream telemetry, inspect state. | All state mutations, recovery triggers, config changes. |
| **`OPERATOR`** | Trigger recovery, pause trading, reset circuit breakers. | Model promotions/rollbacks, secret management, config alteration. |
| **`MODEL_MANAGER`** | Evaluate candidates, promote models, rollback models. | Risk parameter alterations, circuit breaker resets. |
| **`ADMIN`** | Unrestricted access across all operational, risk, and model domains. | N/A |

## 2. High-Impact Action Auditing

High-impact permissions require explicit authorization and automatically append an immutable audit record:
- `Permission.PROMOTE_MODEL`
- `Permission.ROLLBACK_MODEL`
- `Permission.CHANGE_RISK_LIMITS`
- `Permission.RESET_CIRCUIT_BREAKER`
- `Permission.MODIFY_CONFIG`
- `Permission.MANAGE_SECRETS`

Any unauthorized access attempt raises an `AccessDeniedError` and immediately logs an `ACCESS_DENIED` audit entry with actor ID and timestamp.
