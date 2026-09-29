"""Model Governance package for Phase 8 Learning Loop."""

from learning.model_governance.auditor import AuditorConfig, ModelAuditor
from learning.model_governance.compatibility import (
    CompatibilityCheckResult,
    ModelCompatibilityValidator,
    SystemEnvironmentSpecification,
)
from learning.model_governance.models import (
    AuditRecommendation,
    DriftStatus,
    FeatureDriftEntry,
    GateResult,
    GateStatus,
    ModelAuditResult,
    ModelLifecycleState,
    ModelLineageRecord,
    PromotionRecord,
    RollbackRecord,
    ValidationResult,
)
from learning.model_governance.promotion import ControlledPromotionManager, PromotionError
from learning.model_governance.registry import EnterpriseModelRegistry
from learning.model_governance.reports import (
    DriftReporter,
    ModelAuditReporter,
    ModelComparisonReporter,
)
from learning.model_governance.shadow import ShadowComparisonRecord, ShadowModelManager
from learning.model_governance.telemetry import GovernanceTelemetryProvider

__all__ = [
    "AuditorConfig",
    "ModelAuditor",
    "CompatibilityCheckResult",
    "ModelCompatibilityValidator",
    "SystemEnvironmentSpecification",
    "AuditRecommendation",
    "DriftStatus",
    "FeatureDriftEntry",
    "GateResult",
    "GateStatus",
    "ModelAuditResult",
    "ModelLifecycleState",
    "ModelLineageRecord",
    "PromotionRecord",
    "RollbackRecord",
    "ValidationResult",
    "ControlledPromotionManager",
    "PromotionError",
    "EnterpriseModelRegistry",
    "DriftReporter",
    "ModelAuditReporter",
    "ModelComparisonReporter",
    "ShadowComparisonRecord",
    "ShadowModelManager",
    "GovernanceTelemetryProvider",
]
