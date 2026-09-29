# Model Lineage & Provenance

## 1. Overview & Provenance Guarantee

Every trained and registered model in the system contains a complete `ModelLineageRecord` stored in `model_lineage.json`.

It answers definitively: **"Exactly how was this model produced?"**

## 2. Lineage Chain

```
parent production model
        |
        v
training dataset (version, timeframe, symbols)
        |
        v
feature version & schema
        |
        v
label version & target horizon
        |
        v
training configuration & hyperparameters
        |
        v
random seed & dependency versions
        |
        v
candidate model
        |
        v
validation result
        |
        v
promotion / rejection record
```

## 3. Record Schema

- `model_version`: Immutable semantic identifier.
- `parent_production_model`: Version of the active production model during training.
- `dataset_version`: Exact dataset version.
- `feature_version`: Feature engine version.
- `label_version`: Label engine version.
- `training_period`: `(start_time, end_time)` of training partition.
- `validation_period`: `(start_time, end_time)` of validation partition.
- `test_period`: `(start_time, end_time)` of test partition.
- `random_seed`: Deterministic integer seed.
- `code_version`: Git commit or release tag.
- `created_at`: UTC timestamp.
