# Training Reproducibility & Governance

## 1. Overview & Requirements (Section 38)

Given identical dataset versions, feature versions, configurations, and random seeds, candidate retraining must produce deterministic, identical model artifacts and performance metrics.

## 2. Invariants

- **Fixed Random Seed**: Injected into all estimator initializations (Random Forest, LightGBM, Logistic Regression).
- **Sorted Data Ingestion**: Input candle timestamps and feature matrices are explicitly sorted.
- **Isolated Preprocessing**: `ChronologicalPreprocessor` calculates medians and scaling parameters strictly on training observations and stores them for reuse.
- **MLflow Tracking**: Hyperparameters, dataset metadata, and artifacts are logged alongside registry metadata.
