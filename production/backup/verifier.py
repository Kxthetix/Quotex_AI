"""Backup Verification (Section 31).

Ensures backups are not merely assumed functional, but verified for:
1. File existence
2. Checksum and SHA-256 integrity
3. Manifest consistency
4. Archive readability (testzip)
5. Presence of expected critical artifacts
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from production.backup.backup_manager import BackupManifest, compute_file_sha256


@dataclass
class VerificationResult:
    is_valid: bool
    backup_id: str
    message: str
    checks_passed: list[str]
    checks_failed: list[str]
    details: dict[str, Any]


class BackupVerifier:
    """Rigorous backup verification engine."""

    @classmethod
    def verify_backup(
        cls,
        archive_path: Path | str,
        manifest_path: Path | str | None = None,
    ) -> VerificationResult:
        """Run all 5 verification gates against a backup archive."""
        archive_p = Path(archive_path)
        passed: list[str] = []
        failed: list[str] = []
        details: dict[str, Any] = {}

        # 1. Existence check
        if not archive_p.exists() or archive_p.stat().st_size == 0:
            return VerificationResult(
                is_valid=False,
                backup_id="UNKNOWN",
                message=f"Backup archive does not exist or is empty: {archive_p}",
                checks_passed=[],
                checks_failed=["existence_check"],
                details={"path": str(archive_p)},
            )
        passed.append("existence_check")

        # 2. Manifest check
        if manifest_path is None:
            # Look for manifest with matching base name
            manifest_p = archive_p.with_suffix(".manifest.json")
            if not manifest_p.exists():
                stem = archive_p.stem
                manifest_p = archive_p.parent / f"{stem}.manifest.json"
        else:
            manifest_p = Path(manifest_path)

        if not manifest_p.exists():
            failed.append("manifest_check")
            return VerificationResult(
                is_valid=False,
                backup_id=archive_p.stem,
                message=f"Missing manifest file at {manifest_p}",
                checks_passed=passed,
                checks_failed=failed,
                details={},
            )

        try:
            manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
            backup_id = manifest_data.get("backup_id", archive_p.stem)
            expected_hash = manifest_data.get("archive_hash")
            passed.append("manifest_check")
        except Exception as e:
            failed.append("manifest_check")
            return VerificationResult(
                is_valid=False,
                backup_id=archive_p.stem,
                message=f"Invalid manifest JSON: {e}",
                checks_passed=passed,
                checks_failed=failed,
                details={},
            )

        # 3. Checksum verification
        actual_hash = compute_file_sha256(archive_p)
        if actual_hash != expected_hash:
            failed.append("checksum_integrity")
            return VerificationResult(
                is_valid=False,
                backup_id=backup_id,
                message=f"Checksum mismatch: expected {expected_hash}, calculated {actual_hash}",
                checks_passed=passed,
                checks_failed=failed,
                details={"expected": expected_hash, "actual": actual_hash},
            )
        passed.append("checksum_integrity")

        # 4. Archive readability (testzip)
        try:
            with zipfile.ZipFile(archive_p, "r") as zf:
                corrupted_file = zf.testzip()
                if corrupted_file is not None:
                    failed.append("archive_readability")
                    return VerificationResult(
                        is_valid=False,
                        backup_id=backup_id,
                        message=f"Archive corrupted at file: {corrupted_file}",
                        checks_passed=passed,
                        checks_failed=failed,
                        details={"corrupted_file": corrupted_file},
                    )
                passed.append("archive_readability")

                # 5. File list integrity against manifest
                archive_namelist = set(zf.namelist())
                manifest_files = {f["filename"] for f in manifest_data.get("files", [])}
                missing_in_archive = manifest_files - archive_namelist
                if missing_in_archive:
                    failed.append("file_inventory_match")
                    return VerificationResult(
                        is_valid=False,
                        backup_id=backup_id,
                        message=f"Manifest files missing from archive: {missing_in_archive}",
                        checks_passed=passed,
                        checks_failed=failed,
                        details={"missing": list(missing_in_archive)},
                    )
                passed.append("file_inventory_match")
        except Exception as e:
            failed.append("archive_readability")
            return VerificationResult(
                is_valid=False,
                backup_id=backup_id,
                message=f"Error inspecting archive zip: {e}",
                checks_passed=passed,
                checks_failed=failed,
                details={"error": str(e)},
            )

        details["file_count"] = len(manifest_data.get("files", []))
        details["archive_bytes"] = archive_p.stat().st_size

        return VerificationResult(
            is_valid=True,
            backup_id=backup_id,
            message="Backup successfully verified across all 5 verification gates.",
            checks_passed=passed,
            checks_failed=[],
            details=details,
        )
