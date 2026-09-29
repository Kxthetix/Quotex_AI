# Incident Management System

## 1. Incident Lifecycle

Operational incidents progress through five explicit states:

```
  [OPEN]
     │
     ▼
[INVESTIGATING]
     │
     ▼
  [MITIGATED]
     │
     ▼
  [RESOLVED]
     │
     ▼
   [CLOSED]
```

## 2. Incident Model

Each incident records:
- `incident_id`: Unique identifier (e.g. `inc_20260930_120000_1`).
- `timestamp`: UTC detection time.
- `severity`: `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`.
- `component`: Impacted service or infrastructure.
- `detected_by`: Automated probe, monitor, or operator handle.
- `description`: Summary of the symptom or disruption.
- `actions`: Chronological log of diagnostic actions and containment steps.
- `resolution`: Root cause summary and permanent remediation notes.
- `resolved_at`: UTC resolution timestamp.

All transitions are durably stored in `data/incidents/incidents.jsonl`.
