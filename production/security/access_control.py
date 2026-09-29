"""Role-Based Access Control (RBAC) (Sections 36, 38).

Enforces permissions across:
- Read-only monitoring
- Operational control
- Model management
- Risk limit modifications
- Administrative actions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from production.audit.audit_logger import AuditLogger


class Role(str, Enum):
    READ_ONLY_MONITOR = "READ_ONLY_MONITOR"
    OPERATOR = "OPERATOR"
    MODEL_MANAGER = "MODEL_MANAGER"
    ADMIN = "ADMIN"


class Permission(str, Enum):
    VIEW_METRICS = "VIEW_METRICS"
    VIEW_STATE = "VIEW_STATE"
    TRIGGER_RECOVERY = "TRIGGER_RECOVERY"
    TRIGGER_PAUSE = "TRIGGER_PAUSE"
    RESET_CIRCUIT_BREAKER = "RESET_CIRCUIT_BREAKER"
    PROMOTE_MODEL = "PROMOTE_MODEL"
    ROLLBACK_MODEL = "ROLLBACK_MODEL"
    CHANGE_RISK_LIMITS = "CHANGE_RISK_LIMITS"
    MODIFY_CONFIG = "MODIFY_CONFIG"
    MANAGE_SECRETS = "MANAGE_SECRETS"
    ADMIN_ALL = "ADMIN_ALL"


ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.READ_ONLY_MONITOR: {
        Permission.VIEW_METRICS,
        Permission.VIEW_STATE,
    },
    Role.OPERATOR: {
        Permission.VIEW_METRICS,
        Permission.VIEW_STATE,
        Permission.TRIGGER_RECOVERY,
        Permission.TRIGGER_PAUSE,
        Permission.RESET_CIRCUIT_BREAKER,
    },
    Role.MODEL_MANAGER: {
        Permission.VIEW_METRICS,
        Permission.VIEW_STATE,
        Permission.PROMOTE_MODEL,
        Permission.ROLLBACK_MODEL,
    },
    Role.ADMIN: {
        Permission.VIEW_METRICS,
        Permission.VIEW_STATE,
        Permission.TRIGGER_RECOVERY,
        Permission.TRIGGER_PAUSE,
        Permission.RESET_CIRCUIT_BREAKER,
        Permission.PROMOTE_MODEL,
        Permission.ROLLBACK_MODEL,
        Permission.CHANGE_RISK_LIMITS,
        Permission.MODIFY_CONFIG,
        Permission.MANAGE_SECRETS,
        Permission.ADMIN_ALL,
    },
}

HIGH_IMPACT_PERMISSIONS = {
    Permission.PROMOTE_MODEL,
    Permission.ROLLBACK_MODEL,
    Permission.CHANGE_RISK_LIMITS,
    Permission.RESET_CIRCUIT_BREAKER,
    Permission.MODIFY_CONFIG,
    Permission.MANAGE_SECRETS,
}


class AccessDeniedError(Exception):
    """Raised when an actor lacks permission for an operation."""
    pass


@dataclass
class UserSession:
    """Represents an authenticated user or automated service actor."""
    user_id: str
    roles: list[Role] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """Check if any assigned role grants the permission."""
        for role in self.roles:
            perms = ROLE_PERMISSIONS.get(role, set())
            if Permission.ADMIN_ALL in perms or permission in perms:
                return True
        return False


class AccessControlManager:
    """Manages authorization checks and audits sensitive actions."""

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        self.audit_logger = audit_logger

    def authorize(
        self,
        user: UserSession,
        permission: Permission,
        resource: str,
        reason: str = "Authorized operational access",
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Verify actor has required permission, logging high-impact or denied actions."""
        authorized = user.has_permission(permission)

        if not authorized:
            if self.audit_logger:
                self.audit_logger.record_action(
                    actor=user.user_id,
                    action=f"ACCESS_DENIED:{permission.value}",
                    resource=resource,
                    reason=f"Access denied for roles: {[r.value for r in user.roles]}",
                    previous_state=None,
                    new_state={"status": "DENIED", "attempted_permission": permission.value},
                )
            raise AccessDeniedError(
                f"Actor '{user.user_id}' lacks permission '{permission.value}' for resource '{resource}'"
            )

        # Audit high-impact allowed actions
        if permission in HIGH_IMPACT_PERMISSIONS and self.audit_logger:
            self.audit_logger.record_action(
                actor=user.user_id,
                action=f"HIGH_IMPACT_ACTION:{permission.value}",
                resource=resource,
                reason=reason,
                previous_state=None,
                new_state=context,
            )

        return True
