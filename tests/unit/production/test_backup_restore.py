"""Unit tests for Backup Manager, Verification, and Restore Testing (Sections 26-33, 55)."""

import json
from pathlib import Path

from production.backup.backup_manager import BackupManager, BackupType
from production.backup.restore import RestoreManager
from production.backup.verifier import BackupVerifier


def test_backup_and_verify_all_domains(tmp_path: Path):
    # Setup mock file structure
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "btc_1m.parquet").write_bytes(b"PARQUET_MOCK_DATA_BYTE_STREAM")

    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "ledger.json").write_text('{"records": [1, 2, 3]}', encoding="utf-8")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "gbm_model.bin").write_bytes(b"MODEL_WEIGHTS_BIN_DATA")

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "system.json").write_text('{"api_key": "secret_abc_123", "rate": 10}', encoding="utf-8")

    backup_root = tmp_path / "backups"
    manager = BackupManager(backup_root=backup_root)

    # 1. Raw data backup
    raw_manifest = manager.backup_raw_data(raw_data_dirs=[raw_dir])
    assert raw_manifest.file_count == 1
    raw_archive = backup_root / "raw_data" / raw_manifest.archive_filename
    res = BackupVerifier.verify_backup(raw_archive)
    assert res.is_valid is True
    assert len(res.checks_passed) == 5

    # 2. Database backup
    db_manifest = manager.backup_database_state(db_paths=[db_dir])
    db_archive = backup_root / "database" / db_manifest.archive_filename
    assert BackupVerifier.verify_backup(db_archive).is_valid is True

    # 3. Model artifacts backup
    model_manifest = manager.backup_model_artifacts(model_dirs=[models_dir])
    model_archive = backup_root / "models" / model_manifest.archive_filename
    assert BackupVerifier.verify_backup(model_archive).is_valid is True

    # 4. Configuration backup (with secret redaction)
    cfg_manifest = manager.backup_configuration(config_dirs=[config_dir])
    cfg_archive = backup_root / "config" / cfg_manifest.archive_filename
    assert BackupVerifier.verify_backup(cfg_archive).is_valid is True

    # Check secret was redacted in the archive
    restore_sandbox = tmp_path / "restore_sandbox"
    restore_res = RestoreManager.restore(cfg_archive, restore_sandbox)
    assert restore_res.success is True

    restored_config = json.loads((restore_sandbox / "system.json").read_text(encoding="utf-8"))
    assert restored_config["api_key"] == "***REDACTED***"
    assert restored_config["rate"] == 10


def test_restore_cycle_and_tamper_detection(tmp_path: Path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "important.dat").write_text("CRITICAL_FINANCIAL_RECORDS", encoding="utf-8")

    backup_root = tmp_path / "backups"
    manager = BackupManager(backup_root=backup_root)
    manifest = manager.backup_database_state(db_paths=[src_dir])

    archive_path = backup_root / "database" / manifest.archive_filename

    # Clean restore
    sandbox_dir = tmp_path / "sandbox"
    assert RestoreManager.test_restore_cycle(archive_path, sandbox_dir) is True
    assert (sandbox_dir / "important.dat").read_text(encoding="utf-8") == "CRITICAL_FINANCIAL_RECORDS"

    # Now tamper with the archive
    with open(archive_path, "ab") as f:
        f.write(b"CORRUPTION_BYTES")

    # Verification must fail
    res = BackupVerifier.verify_backup(archive_path)
    assert res.is_valid is False
    assert "Checksum mismatch" in res.message or "archive_readability" in res.checks_failed
