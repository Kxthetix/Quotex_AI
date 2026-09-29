# Deployment Rollback Procedures

## 1. Rollback Strategy

Production deployments must always be reversible. The `DeploymentRollbackManager` retains release snapshots in `data/deployment/releases.jsonl`:
- Application version.
- Active model version.
- Configuration snapshot.
- Deployer identity and deployment timestamp.

## 2. Automated Rollback Execution

If post-deployment monitoring detects anomalous latency, elevated error rates, or model failure:
1. Operator or automated monitor invokes `DeploymentRollbackManager.rollback(actor, reason)`.
2. The active release status transitions to `ROLLED_BACK`.
3. The latest superseded known-good release is re-activated.
4. Active configurations and model version pointers revert to the previous release.
5. An immutable audit record `DEPLOYMENT_ROLLED_BACK` is written.
6. Services reload verified configurations.
