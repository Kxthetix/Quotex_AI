# Enterprise 4-Pillar Backup Strategy

## 1. The Four Backup Pillars

A production trading system cannot rely on database snapshots alone. The system explicitly archives four separate domains:

| Pillar | Included Data | Retention Policy | Redaction |
| :--- | :--- | :--- | :--- |
| **1. Raw Market Data** | Parquet partitions, raw tick feeds, candle stores. | 365 Days | None |
| **2. Database State** | Prediction ledger, outcomes, trade history, SQLite/Timescale dumps, audit trail. | 90 Days | None |
| **3. Model Artifacts** | Production weights, candidate weights, calibration curves, encoders, `registry.json`. | Permanent / Immutable | None |
| **4. Configuration** | Feature specs, signal parameters, risk limits, runtime environments. | 90 Days | Full Secret Redaction |

## 2. Backup Manifest

Every backup produces a paired JSON manifest:
```json
{
  "backup_id": "bak_database_20260930_120000_a1b2c3",
  "backup_type": "DATABASE",
  "created_at": "2026-09-30T12:00:00Z",
  "archive_filename": "bak_database_20260930_120000_a1b2c3.zip",
  "archive_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "file_count": 12,
  "total_bytes": 1048576,
  "files": [
    {
      "filename": "trading.db",
      "size_bytes": 524288,
      "sha256": "8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"
    }
  ]
}
```

## 3. Secret Redaction in Configuration Backups

When packaging configuration files (`.json`, `.yaml`, `.yml`), the backup manager recursively parses the dictionary and substitutes any field matching `api_key`, `secret`, `password`, `token`, `auth`, or `credential` with `***REDACTED***`.
