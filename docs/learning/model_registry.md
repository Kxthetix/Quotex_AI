# Enterprise Model Registry

## 1. Overview & Purpose

The `EnterpriseModelRegistry` manages model artifacts, versioning, explicit lifecycle states, and integrity verification.

## 2. Model Lifecycle States (Section 35)

Every registered model exists in an explicit state:

| State | Description |
| --- | --- |
| `DEVELOPMENT` | Initial model design and prototyping. |
| `CANDIDATE` | Retrained model stored in candidate isolation. |
| `VALIDATING` | Model currently executing the 10 validation gates. |
| `VALIDATED` | Model passed all validation gates; eligible for review. |
| `APPROVAL_REQUIRED` | Review package prepared; awaits sign-off. |
| `APPROVED` | Governance board or lead operator signed off. |
| `PRODUCTION` | Active model driving live paper-trading inference. |
| `REJECTED` | Failed validation gates or rejected during review/rollback. |
| `RETIRED` | Previous production model superseded by a new candidate. |
| `ARCHIVED` | Preserved for historical audit and compliance. |

## 3. Storage Hierarchy

```
data/models/
├── production/                   # Active production models
│   └── direction_classifier_v001.joblib
├── candidates/                   # Isolated candidate models
│   └── direction_classifier_v002_cand.joblib
├── archive/                      # Retired and archived models
├── enterprise_registry_index.json# Complete index metadata
├── model_lineage.json            # End-to-end lineage records
├── promotion_history.json        # Promotion audit trail
└── rollback_history.json         # Rollback audit trail
```

## 4. Artifact Integrity Checks

Before loading any model, the registry computes the SHA256 checksum of the artifact file on disk and verifies it against the recorded checksum in `enterprise_registry_index.json`. Any mismatch raises an `ArtifactIntegrityViolation` error and blocks execution.
