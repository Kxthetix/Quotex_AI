# Candidate Retraining Workflow

## 1. Overview & Triggers

Candidate retraining is only initiated when empirical evidence justifies it:
- Sustained model performance degradation (below critical threshold e.g. 48%)
- Severe concept drift detected across evaluated prediction outcomes
- Scheduled retraining cadence reached

Retraining creates a **Candidate Model** stored in candidate isolation. It **NEVER** automatically updates or replaces active production models.

## 2. Temporal Ordering & Anti-Leakage (Sections 20, 44)

The `CandidateDatasetBuilder` enforces strict chronological boundaries:
- Training start $\le$ Training end $<$ Validation start $\le$ Validation end $<$ Test start $\le$ Test end
- Training cutoff $T$: Observations arriving after $T$ can **never** enter historical training datasets.
- Feature scalers and imputers (`ChronologicalPreprocessor`) are fitted strictly on the training partition and transformed onto validation/test.

## 3. Dataset & Label Versioning

- Every dataset receives an immutable `dataset_version` (e.g. `ds_v001`).
- Feature definitions match locked `feature_version` (e.g. `v1.0.0`).
- Target labels match locked `label_version` (e.g. `v1.0.0`).
