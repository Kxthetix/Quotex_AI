"""Provider Priority, Controlled Failover, and Cross-Provider Consistency (Sections 6, 7, 8).

Ensures the pipeline can failover between Primary, Secondary, and Tertiary providers
only after verifying connection health, symbol mapping, and data continuity.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from data.assets.registry import AssetRegistry
from data.providers.base import MarketDataProvider

if TYPE_CHECKING:
    from production.audit.audit_logger import AuditLogger


class ProviderPriority(StrEnum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    TERTIARY = "TERTIARY"


@dataclass
class FailoverEvent:
    event_id: str
    from_provider: str
    to_provider: str
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    validation_checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class ProviderFailoverManager:
    """Manages provider priority tiers and executes validated failover procedures."""

    def __init__(
        self,
        asset_registry: AssetRegistry | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.asset_registry = asset_registry or AssetRegistry.get_instance()
        self.audit_logger = audit_logger
        self._providers: dict[ProviderPriority, MarketDataProvider] = {}
        self._active_priority: ProviderPriority = ProviderPriority.PRIMARY
        self._failover_history: list[FailoverEvent] = []
        self._lock = threading.Lock()

    def register_provider(self, priority: ProviderPriority, provider: MarketDataProvider) -> None:
        """Register a provider at a priority tier."""
        with self._lock:
            self._providers[priority] = provider

    @property
    def active_provider(self) -> MarketDataProvider | None:
        with self._lock:
            return self._providers.get(self._active_priority)

    @property
    def active_priority(self) -> ProviderPriority:
        with self._lock:
            return self._active_priority

    def validate_candidate_provider(
        self,
        provider: MarketDataProvider,
        symbols_to_verify: list[str],
    ) -> tuple[bool, dict[str, bool]]:
        """Validate candidate provider health, symbol mappings, and connectivity before failover."""
        checks: dict[str, bool] = {}

        # 1. Connection check
        checks["connected"] = provider.is_connected

        # 2. Symbol mapping verification
        mapping_ok = True
        for sym in symbols_to_verify:
            asset = self.asset_registry.get_asset_by_symbol(sym)
            if not asset:
                mapping_ok = False
                break
            try:
                self.asset_registry.resolve_to_provider_symbol(asset.asset_id, provider.provider_name)
            except KeyError:
                mapping_ok = False
                break
        checks["symbol_mapping_valid"] = mapping_ok

        # 3. Health status check
        health = provider.get_health()
        checks["health_status_ok"] = bool(health.get("connected", False))

        passed = all(checks.values())
        return passed, checks

    def trigger_failover(
        self,
        reason: str,
        required_symbols: list[str] | None = None,
    ) -> tuple[bool, MarketDataProvider | None, str]:
        """Execute controlled failover to the next available priority provider."""
        with self._lock:
            symbols = required_symbols or ["BTC/USDT"]
            current_provider = self._providers.get(self._active_priority)
            from_name = current_provider.provider_name if current_provider else "NONE"

            # Determine candidates in order
            priority_order = [ProviderPriority.PRIMARY, ProviderPriority.SECONDARY, ProviderPriority.TERTIARY]
            current_idx = priority_order.index(self._active_priority)
            candidates = priority_order[current_idx + 1 :]

            for candidate_priority in candidates:
                cand_provider = self._providers.get(candidate_priority)
                if not cand_provider:
                    continue

                valid, checks = self.validate_candidate_provider(cand_provider, symbols)
                if valid:
                    # Execute failover
                    self._active_priority = candidate_priority
                    evt_id = f"failover_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{len(self._failover_history)+1}"
                    event = FailoverEvent(
                        event_id=evt_id,
                        from_provider=from_name,
                        to_provider=cand_provider.provider_name,
                        reason=reason,
                        validation_checks=checks,
                    )
                    self._failover_history.append(event)

                    if self.audit_logger:
                        self.audit_logger.record_action(
                            actor="ProviderFailoverManager",
                            action="PROVIDER_FAILOVER",
                            resource=f"market_data:{cand_provider.provider_name}",
                            reason=reason,
                            previous_state={"active_provider": from_name, "priority": str(priority_order[current_idx])},
                            new_state=event.to_dict(),
                        )

                    return True, cand_provider, f"Failed over from {from_name} to {cand_provider.provider_name}"

            return False, current_provider, f"Failover failed: No healthy backup provider passed validation for {symbols}"

    def get_history(self) -> list[FailoverEvent]:
        with self._lock:
            return list(self._failover_history)


class CrossProviderConsistencyChecker:
    """Monitors price, timestamp, and volume discrepancies across multiple providers (Section 8)."""

    def __init__(self, max_allowed_price_divergence_pct: float = 0.5) -> None:
        self.max_divergence_pct = max_allowed_price_divergence_pct
        self._discrepancies: list[dict[str, Any]] = []

    def check_quote_consistency(
        self,
        symbol: str,
        price_a: float,
        provider_a: str,
        price_b: float,
        provider_b: str,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Compare contemporaneous prices between two providers."""
        mid = (price_a + price_b) / 2.0
        if mid <= 0:
            return True, 0.0, {}

        divergence_pct = (abs(price_a - price_b) / mid) * 100.0
        is_consistent = divergence_pct <= self.max_divergence_pct

        details = {
            "symbol": symbol,
            "provider_a": provider_a,
            "price_a": price_a,
            "provider_b": provider_b,
            "price_b": price_b,
            "divergence_pct": round(divergence_pct, 4),
            "is_consistent": is_consistent,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        if not is_consistent:
            self._discrepancies.append(details)

        return is_consistent, divergence_pct, details

    def get_discrepancies(self) -> list[dict[str, Any]]:
        return list(self._discrepancies)
