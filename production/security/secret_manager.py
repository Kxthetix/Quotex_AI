"""Secret Management and Redaction (Sections 34, 35).

Ensures credentials, API keys, database credentials, and tokens are safely
loaded, validated at startup, and masked from logs, exceptions, and backups.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Default list of mandatory secrets expected in a production trading setup
DEFAULT_REQUIRED_SECRETS = [
    "TRADING_DATABASE_URL",
    "TRADING_REDIS_URL",
    "MARKET_DATA_API_KEY",
]


class SecretValidationError(Exception):
    """Raised when mandatory production secrets are missing or malformed."""
    pass


class SecretManager:
    """Safely manages configuration secrets without leaking them in logs or traces."""

    def __init__(self, env_prefix: str = "AI_TRADER_") -> None:
        self.env_prefix = env_prefix

    @staticmethod
    def get_secret(key: str, default: str | None = None) -> str | None:
        """Retrieve a secret from environment variables."""
        return os.environ.get(key, default)

    @classmethod
    def validate_required_secrets(
        cls,
        required_keys: list[str] | None = None,
        allow_defaults_in_dev: bool = True,
    ) -> tuple[bool, list[str]]:
        """Verify that all required environment variables/secrets are defined.
        
        Returns:
            tuple[is_valid, missing_keys]
        """
        keys = required_keys or DEFAULT_REQUIRED_SECRETS
        missing: list[str] = []

        is_production = os.environ.get("TRADING_ENV", "development").lower() == "production"

        for k in keys:
            val = os.environ.get(k)
            if not val:
                # If in development and allowed, accept dummy fallback, else record missing
                if not is_production and allow_defaults_in_dev:
                    continue
                missing.append(k)

        if missing and is_production:
            raise SecretValidationError(
                f"Startup blocked: Missing mandatory production secrets: {', '.join(missing)}"
            )

        return len(missing) == 0, missing

    @staticmethod
    def mask_secret(value: str | None) -> str:
        """Produce a safe masked representation of a secret string."""
        if not value:
            return "******"
        if len(value) <= 6:
            return "******"
        return f"{value[:2]}****{value[-2:]}"

    @classmethod
    def sanitize_connection_string(cls, conn_str: str) -> str:
        """Strip passwords from DB/Redis URIs for safe logging.
        
        Example: postgresql://user:pass@localhost:5432/db -> postgresql://user:***@localhost:5432/db
        """
        pattern = r"(://[^:]+:)([^@]+)(@)"
        return re.sub(pattern, r"\1***\3", conn_str)
