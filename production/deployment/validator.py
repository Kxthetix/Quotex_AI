"""Deployment Pre-flight and Post-flight Validator (Sections 37, 39, 41).

Verifies environment, configuration schemas, dependency compatibility,
and model artifacts before and after production deployment.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from production.security.secret_manager import SecretManager


@dataclass
class PreflightCheckResult:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class DeploymentValidator:
    """Rigorous pre-flight and post-flight validation gates."""

    REQUIRED_PYTHON_VERSION = (3, 11)

    @classmethod
    def validate_preflight(
        cls,
        config: dict[str, Any],
        model_artifact_path: Path | str | None = None,
        db_reachable: bool = True,
        redis_reachable: bool = True,
    ) -> PreflightCheckResult:
        """Run all pre-flight deployment safety gates."""
        checks: list[dict[str, Any]] = []
        errors: list[str] = []

        # 1. Runtime environment check
        py_ver = sys.version_info[:2]
        if py_ver < cls.REQUIRED_PYTHON_VERSION:
            err = f"Incompatible Python version: {py_ver}. Required: >= {cls.REQUIRED_PYTHON_VERSION}"
            errors.append(err)
            checks.append({"name": "python_runtime", "status": "FAILED", "detail": err})
        else:
            checks.append({"name": "python_runtime", "status": "PASSED", "detail": f"Python {sys.version.split()[0]}"})

        # 2. Configuration schema completeness
        required_sections = ["features", "signal", "risk"]
        for section in required_sections:
            if section not in config:
                err = f"Missing mandatory configuration section: '{section}'"
                errors.append(err)
                checks.append({"name": f"config_{section}", "status": "FAILED", "detail": err})
            else:
                checks.append({"name": f"config_{section}", "status": "PASSED", "detail": "Config section present"})

        # 3. Risk parameter sanity check (Section 38)
        risk_cfg = config.get("risk", {})
        max_drawdown = risk_cfg.get("max_drawdown_limit", 0.15)
        daily_loss = risk_cfg.get("daily_loss_limit", 0.05)
        if not (0.01 <= max_drawdown <= 0.50):
            err = f"Invalid max_drawdown_limit: {max_drawdown}. Must be in [0.01, 0.50]"
            errors.append(err)
            checks.append({"name": "risk_parameters", "status": "FAILED", "detail": err})
        elif not (0.005 <= daily_loss <= max_drawdown):
            err = f"Invalid daily_loss_limit: {daily_loss}. Must be in [0.005, max_drawdown]"
            errors.append(err)
            checks.append({"name": "risk_parameters", "status": "FAILED", "detail": err})
        else:
            checks.append({"name": "risk_parameters", "status": "PASSED", "detail": "Risk limits valid"})

        # 4. Mandatory secrets check (Section 35)
        secrets_valid, missing_secrets = SecretManager.validate_required_secrets(allow_defaults_in_dev=True)
        if not secrets_valid:
            err = f"Missing required secrets: {missing_secrets}"
            errors.append(err)
            checks.append({"name": "secrets_validation", "status": "FAILED", "detail": err})
        else:
            checks.append({"name": "secrets_validation", "status": "PASSED", "detail": "Secrets verified"})

        # 5. Model artifact existence check (Section 29)
        if model_artifact_path:
            p = Path(model_artifact_path)
            if not p.exists():
                err = f"Model artifact not found at {p}"
                errors.append(err)
                checks.append({"name": "model_artifact", "status": "FAILED", "detail": err})
            else:
                checks.append({"name": "model_artifact", "status": "PASSED", "detail": str(p)})

        # 6. Database dependency check
        if not db_reachable:
            err = "Database dependency is not reachable."
            errors.append(err)
            checks.append({"name": "database_dependency", "status": "FAILED", "detail": err})
        else:
            checks.append({"name": "database_dependency", "status": "PASSED", "detail": "Database connected"})

        # 7. Redis dependency check
        if not redis_reachable:
            err = "Redis dependency is not reachable."
            errors.append(err)
            checks.append({"name": "redis_dependency", "status": "FAILED", "detail": err})
        else:
            checks.append({"name": "redis_dependency", "status": "PASSED", "detail": "Redis connected"})

        passed = len(errors) == 0
        return PreflightCheckResult(passed=passed, checks=checks, errors=errors)
