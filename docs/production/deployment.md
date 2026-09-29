# Controlled Production Deployment

## 1. Deployment Lifecycle Gates

Before any production service is deployed or restarted, the `DeploymentValidator` enforces 7 mandatory pre-flight checks:

1. **Python Runtime Version**: Python version must be >= 3.11.
2. **Configuration Completeness**: All required sections (`features`, `signal`, `risk`) must be defined.
3. **Risk Boundary Validation**:
   - `max_drawdown_limit` must lie within $[0.01, 0.50]$.
   - `daily_loss_limit` must lie within $[0.005, \text{max\_drawdown\_limit}]$.
4. **Mandatory Secrets Presence**: Critical credentials must be configured.
5. **Model Artifact Integrity**: Active model files must exist and match catalog hashes.
6. **Database Reachability**: Database connection pool must respond.
7. **Redis Reachability**: Redis cache ping must succeed.

If any check fails, deployment is rejected before state mutations occur.
