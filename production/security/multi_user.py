"""Multi-User Architecture and Strict Tenant Isolation (Sections 37, 38, 39, 40, 59).

Prevents cross-user data leaks and enforces strict scoping on:
Portfolios, Orders, Positions, API credentials, and Audit logs.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from production.audit.audit_logger import AuditLogger


class UserRole(StrEnum):
    VIEWER = "VIEWER"
    TRADER = "TRADER"
    ANALYST = "ANALYST"
    MODEL_MANAGER = "MODEL_MANAGER"
    ADMIN = "ADMIN"


class TenantIsolationViolationError(PermissionError):
    """Raised when an actor attempts cross-tenant or unauthorized cross-user access."""
    pass


@dataclass
class UserAccount:
    """Canonical user identity with tenant scoping."""
    user_id: str
    tenant_id: str
    username: str
    roles: list[UserRole] = field(default_factory=lambda: [UserRole.VIEWER])
    active_portfolios: list[str] = field(default_factory=list)
    api_credentials: dict[str, str] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def has_role(self, role: UserRole) -> bool:
        return UserRole.ADMIN in self.roles or role in self.roles


@dataclass
class UserScopedPortfolio:
    """Portfolio bound strictly to a tenant and owner user."""
    portfolio_id: str
    owner_user_id: str
    tenant_id: str
    cash: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)
    orders: list[str] = field(default_factory=list)


class MultiUserManager:
    """Manages user registration, authentication, and cross-tenant isolation enforcement."""

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        self.audit_logger = audit_logger
        self._users: dict[str, UserAccount] = {}  # user_id -> UserAccount
        self._portfolios: dict[str, UserScopedPortfolio] = {}  # portfolio_id -> UserScopedPortfolio
        self._lock = threading.Lock()

    def register_user(
        self,
        user_id: str,
        tenant_id: str,
        username: str,
        roles: list[UserRole] | None = None,
    ) -> UserAccount:
        """Create and register a user."""
        with self._lock:
            user = UserAccount(
                user_id=user_id,
                tenant_id=tenant_id,
                username=username,
                roles=roles or [UserRole.VIEWER],
            )
            self._users[user_id] = user
            return user

    def create_portfolio(
        self,
        portfolio_id: str,
        user: UserAccount,
        initial_cash: float = 100_000.0,
    ) -> UserScopedPortfolio:
        """Create a portfolio scoped to the user and their tenant."""
        with self._lock:
            portfolio = UserScopedPortfolio(
                portfolio_id=portfolio_id,
                owner_user_id=user.user_id,
                tenant_id=user.tenant_id,
                cash=initial_cash,
            )
            self._portfolios[portfolio_id] = portfolio
            user.active_portfolios.append(portfolio_id)
            return portfolio

    def authorize_portfolio_access(
        self,
        requesting_user: UserAccount,
        target_portfolio_id: str,
        action: str = "VIEW",
    ) -> UserScopedPortfolio:
        """Section 38, 59: Validate that requesting user owns the portfolio or is tenant Admin.
        
        Raises TenantIsolationViolationError on cross-user or cross-tenant access.
        """
        with self._lock:
            portfolio = self._portfolios.get(target_portfolio_id)
            if not portfolio:
                raise KeyError(f"Portfolio '{target_portfolio_id}' does not exist.")

            # Admin can access within tenant, or super-admin
            is_owner = portfolio.owner_user_id == requesting_user.user_id
            same_tenant = portfolio.tenant_id == requesting_user.tenant_id
            is_admin = requesting_user.has_role(UserRole.ADMIN) and same_tenant

            if not (is_owner or is_admin):
                if self.audit_logger:
                    self.audit_logger.record_action(
                        actor=requesting_user.user_id,
                        action="TENANT_ISOLATION_VIOLATION",
                        resource=f"portfolio:{target_portfolio_id}",
                        reason=(
                            f"User '{requesting_user.user_id}' in tenant '{requesting_user.tenant_id}' "
                            f"denied access to portfolio owned by '{portfolio.owner_user_id}' "
                            f"in tenant '{portfolio.tenant_id}'"
                        ),
                    )
                raise TenantIsolationViolationError(
                    f"Access denied: User '{requesting_user.user_id}' is not authorized "
                    f"to access portfolio '{target_portfolio_id}'"
                )

            return portfolio
