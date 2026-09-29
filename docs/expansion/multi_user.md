# Multi-User Architecture and Tenant Isolation

## 1. Domain Entities

- `Tenant`: Organizational boundary (e.g. `tenant_alpha`).
- `UserAccount`: Authenticated identity (`user_id`, `tenant_id`, `roles`, `api_credentials`).
- `UserScopedPortfolio`: Dedicated capital container strictly bound to `(tenant_id, owner_user_id)`.

## 2. Tenant Isolation Enforcement (Section 38, 59)

Every read or mutation request must pass through `MultiUserManager.authorize_portfolio_access()`:
- User A in Tenant Alpha cannot access Portfolio B owned by User B in Tenant Beta.
- Cross-user queries trigger `TenantIsolationViolationError`.
- Unauthorized access attempts are logged as security audit incidents.

## 3. User Roles

- `VIEWER`: Read-only access to assigned portfolios and metrics.
- `TRADER`: Order execution within user-scoped portfolio and risk limits.
- `ANALYST`: Feature inspection and backtest exploration.
- `MODEL_MANAGER`: Candidate evaluation and promotion workflows.
- `ADMIN`: Tenant-wide administrative management.
