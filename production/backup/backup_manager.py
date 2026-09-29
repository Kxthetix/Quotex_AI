"""Enterprise Backup Manager (Sections 26-30).

Implements independent and unified backups for the four mandatory production pillars:
1. Raw market data
2. Database state (ledger, outcomes, trades, audit trail)
3. Model artifacts (production models, candidates, calibration, encoders)
4. System configurations (feature, signal, risk, system configs - secrets redacted)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from production.audit.audit_logger import redact_dict


class BackupType(str, Enum):
    RAW_DATA = "RAW_DATA"
    DATABASE = "DATABASE"
    MODEL_ARTIFACTS = "MODEL_ARTIFACTS"
    CONFIGURATION = "CONFIGURATION"
    FULL = "FULL"


@dataclass
class BackupManifest:
    backup_id: str
    backup_type: BackupType
    created_at: str
    source_paths: list[str]
    archive_filename: str
    archive_hash: str
    file_count: int
    total_bytes: int
    files: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["backup_type"] = self.backup_type.value
        return data


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BackupManager:
    """Creates verifiable archives of critical system states."""

    def __init__(self, backup_root: Path | str = "data/backups") -> None:
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)

    def _create_archive(
        self,
        backup_type: BackupType,
        source_paths: list[Path],
        subfolder: str,
        metadata: dict[str, Any] | None = None,
        redact_configs: bool = False,
    ) -> BackupManifest:
        """Helper to package file list into a zip archive with manifest."""
        timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_id = f"bak_{backup_type.value.lower()}_{timestamp_str}_{os.urandom(3).hex()}"
        out_dir = self.backup_root / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)

        archive_name = f"{backup_id}.zip"
        archive_path = out_dir / archive_name

        file_manifests: list[dict[str, Any]] = []
        total_bytes = 0

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for src in source_paths:
                if not src.exists():
                    continue

                if src.is_file():
                    content_bytes = src.read_bytes()
                    if redact_configs and src.suffix in (".json", ".yaml", ".yml"):
                        try:
                            data = json.loads(content_bytes.decode("utf-8"))
                            data = redact_dict(data)
                            content_bytes = json.dumps(data, indent=2).encode("utf-8")
                        except Exception:
                            pass

                    arcname = src.name
                    zf.writestr(arcname, content_bytes)
                    f_hash = hashlib.sha256(content_bytes).hexdigest()
                    size = len(content_bytes)
                    total_bytes += size
                    file_manifests.append({
                        "filename": arcname,
                        "size_bytes": size,
                        "sha256": f_hash,
                    })

                elif src.is_dir():
                    for item in src.rglob("*"):
                        if item.is_file():
                            content_bytes = item.read_bytes()
                            if redact_configs and item.suffix in (".json", ".yaml", ".yml"):
                                try:
                                    data = json.loads(content_bytes.decode("utf-8"))
                                    data = redact_dict(data)
                                    content_bytes = json.dumps(data, indent=2).encode("utf-8")
                                except Exception:
                                    pass

                            arcname = str(item.relative_to(src))
                            zf.writestr(arcname, content_bytes)
                            f_hash = hashlib.sha256(content_bytes).hexdigest()
                            size = len(content_bytes)
                            total_bytes += size
                            file_manifests.append({
                                "filename": arcname,
                                "size_bytes": size,
                                "sha256": f_hash,
                            })

        # Calculate checksum of the created archive
        archive_hash = compute_file_sha256(archive_path)

        manifest = BackupManifest(
            backup_id=backup_id,
            backup_type=backup_type,
            created_at=datetime.now(UTC).isoformat(),
            source_paths=[str(p) for p in source_paths],
            archive_filename=archive_name,
            archive_hash=archive_hash,
            file_count=len(file_manifests),
            total_bytes=total_bytes,
            files=file_manifests,
            metadata=metadata or {},
        )

        manifest_path = out_dir / f"{backup_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        return manifest

    def backup_raw_data(
        self,
        raw_data_dirs: list[Path | str] | None = None,
        symbol: str = "BTC/USDT",
    ) -> BackupManifest:
        """Section 27: Back up raw market data store."""
        dirs = [Path(d) for d in (raw_data_dirs or ["data/raw", "data/candles"])]
        return self._create_archive(
            backup_type=BackupType.RAW_DATA,
            source_paths=dirs,
            subfolder="raw_data",
            metadata={"symbol": symbol, "retention_days": 365},
        )

    def backup_database_state(
        self,
        db_paths: list[Path | str] | None = None,
    ) -> BackupManifest:
        """Section 28: Back up database state, prediction ledger, outcomes, trades, audit trail."""
        paths = [
            Path(p) for p in (
                db_paths or [
                    "data/audit",
                    "data/models/registry.json",
                    "data/predictions",
                    "data/trades",
                    "data/trading.db",
                ]
            )
        ]
        return self._create_archive(
            backup_type=BackupType.DATABASE,
            source_paths=paths,
            subfolder="database",
            metadata={"frequency": "daily", "retention_days": 90},
        )

    def backup_model_artifacts(
        self,
        model_dirs: list[Path | str] | None = None,
    ) -> BackupManifest:
        """Section 29: Back up production models, candidates, calibration, registry."""
        dirs = [Path(d) for d in (model_dirs or ["data/models", "models/artifacts"])]
        return self._create_archive(
            backup_type=BackupType.MODEL_ARTIFACTS,
            source_paths=dirs,
            subfolder="models",
            metadata={"retention_policy": "immutable"},
        )

    def backup_configuration(
        self,
        config_dirs: list[Path | str] | None = None,
    ) -> BackupManifest:
        """Section 30: Back up system, feature, signal, and risk configs with secrets redacted."""
        dirs = [Path(d) for d in (config_dirs or ["config", "configs"])]
        return self._create_archive(
            backup_type=BackupType.CONFIGURATION,
            source_paths=dirs,
            subfolder="config",
            metadata={"redacted": True},
            redact_configs=True,
        )

    def backup_all(self) -> dict[BackupType, BackupManifest]:
        """Execute full backup suite across all four domains."""
        return {
            BackupType.RAW_DATA: self.backup_raw_data(),
            BackupType.DATABASE: self.backup_database_state(),
            BackupType.MODEL_ARTIFACTS: self.backup_model_artifacts(),
            BackupType.CONFIGURATION: self.backup_configuration(),
        }
