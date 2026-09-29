"""Restore Procedures and Restore Testing (Sections 32, 33).

Provides automated extraction, restoration, and integrity re-validation
to confirm that backups are genuinely recoverable before a disaster occurs.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from production.backup.backup_manager import compute_file_sha256
from production.backup.verifier import BackupVerifier


@dataclass
class RestoreResult:
    success: bool
    backup_id: str
    target_directory: str
    restored_files: list[str]
    message: str
    details: dict[str, Any]


class RestoreManager:
    """Handles controlled restoration of backup archives into target environments."""

    @classmethod
    def restore(
        cls,
        archive_path: Path | str,
        target_dir: Path | str,
        manifest_path: Path | str | None = None,
        verify_first: bool = True,
    ) -> RestoreResult:
        """Extract and validate backup archive into target directory."""
        archive_p = Path(archive_path)
        target_p = Path(target_dir)
        target_p.mkdir(parents=True, exist_ok=True)

        # 1. Pre-restore verification
        if verify_first:
            verif = BackupVerifier.verify_backup(archive_p, manifest_path=manifest_path)
            if not verif.is_valid:
                return RestoreResult(
                    success=False,
                    backup_id=verif.backup_id,
                    target_directory=str(target_p),
                    restored_files=[],
                    message=f"Pre-restore verification failed: {verif.message}",
                    details={"verification_details": verif.details},
                )
            backup_id = verif.backup_id
        else:
            backup_id = archive_p.stem

        # Locate manifest
        if manifest_path is None:
            manifest_p = archive_p.with_suffix(".manifest.json")
            if not manifest_p.exists():
                manifest_p = archive_p.parent / f"{archive_p.stem}.manifest.json"
        else:
            manifest_p = Path(manifest_path)

        expected_files: dict[str, str] = {}
        if manifest_p.exists():
            try:
                manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
                expected_files = {
                    f["filename"]: f.get("sha256", "")
                    for f in manifest_data.get("files", [])
                }
            except Exception:
                pass

        # 2. Extract files
        restored: list[str] = []
        try:
            with zipfile.ZipFile(archive_p, "r") as zf:
                zf.extractall(target_p)
                restored = zf.namelist()
        except Exception as e:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                target_directory=str(target_p),
                restored_files=[],
                message=f"Extraction failure: {e}",
                details={"error": str(e)},
            )

        # 3. Post-restore integrity verification
        tampered_or_corrupt: list[str] = []
        for filename, expected_hash in expected_files.items():
            extracted_file = target_p / filename
            if not extracted_file.exists():
                tampered_or_corrupt.append(f"{filename} (missing after extraction)")
                continue

            if expected_hash:
                actual_hash = compute_file_sha256(extracted_file)
                if actual_hash != expected_hash:
                    tampered_or_corrupt.append(f"{filename} (hash mismatch)")

        if tampered_or_corrupt:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                target_directory=str(target_p),
                restored_files=restored,
                message=f"Restored files failed integrity check: {tampered_or_corrupt}",
                details={"corrupted": tampered_or_corrupt},
            )

        return RestoreResult(
            success=True,
            backup_id=backup_id,
            target_directory=str(target_p),
            restored_files=restored,
            message=f"Successfully restored {len(restored)} files to {target_p}.",
            details={"file_count": len(restored)},
        )

    @classmethod
    def test_restore_cycle(
        cls,
        archive_path: Path | str,
        sandbox_dir: Path | str,
    ) -> bool:
        """Section 33: End-to-end restore cycle testing."""
        result = cls.restore(archive_path, sandbox_dir, verify_first=True)
        return result.success
