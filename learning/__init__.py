"""Learning and evaluation package for Phase 4."""

from learning.dataset import (
    ChronologicalPreprocessor,
    ChronologicalSplitter,
    DatasetGenerator,
    DatasetMetadata,
    DatasetSplits,
)
from learning.evaluation.calibration import (
    ProbabilityCalibrator,
    calculate_expected_calibration_error,
)
from learning.evaluation.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    evaluate_classifier,
    evaluate_regressor,
)
from learning.labels.engine import LabelEngine
from learning.labels.models import LabelDefinition, LabelSet
from learning.tracking import ExperimentTracker

from learning.model_governance import (
    AuditRecommendation,
    AuditorConfig,
    ControlledPromotionManager,
    DriftReporter,
    DriftStatus,
    EnterpriseModelRegistry,
    FeatureDriftEntry,
    GovernanceTelemetryProvider,
    ModelAuditReporter,
    ModelAuditResult,
    ModelAuditor,
    ModelComparisonReporter,
    ModelCompatibilityValidator,
    ModelLifecycleState,
    ModelLineageRecord,
    PromotionError,
    PromotionRecord,
    RollbackRecord,
    ShadowComparisonRecord,
    ShadowModelManager,
    ValidationResult,
)
from learning.retraining import (
    CandidateDatasetBuilder,
    CandidateStabilityMetrics,
    CandidateTrainConfig,
    CandidateTrainer,
    DatasetBuildConfig,
    RetrainingDecisionEngine,
    WalkForwardCandidateValidator,
)
from learning.scheduler import LearningLoopScheduler, SchedulerCycleResult
from learning.validation import ValidationEngine

__all__ = [
    "ChronologicalPreprocessor",
    "ChronologicalSplitter",
    "ClassificationMetrics",
    "DatasetGenerator",
    "DatasetMetadata",
    "DatasetSplits",
    "ExperimentTracker",
    "LabelDefinition",
    "LabelEngine",
    "LabelSet",
    "ProbabilityCalibrator",
    "RegressionMetrics",
    "calculate_expected_calibration_error",
    "evaluate_classifier",
    "evaluate_regressor",
    # Phase 8 Governance & Learning Loop
    "EnterpriseModelRegistry",
    "ModelLifecycleState",
    "ModelLineageRecord",
    "PromotionRecord",
    "RollbackRecord",
    "ValidationResult",
    "ModelAuditor",
    "AuditorConfig",
    "ModelAuditResult",
    "AuditRecommendation",
    "DriftStatus",
    "FeatureDriftEntry",
    "ControlledPromotionManager",
    "PromotionError",
    "ModelCompatibilityValidator",
    "ShadowModelManager",
    "ShadowComparisonRecord",
    "GovernanceTelemetryProvider",
    "ModelAuditReporter",
    "DriftReporter",
    "ModelComparisonReporter",
    "CandidateDatasetBuilder",
    "DatasetBuildConfig",
    "CandidateTrainer",
    "CandidateTrainConfig",
    "RetrainingDecisionEngine",
    "WalkForwardCandidateValidator",
    "CandidateStabilityMetrics",
    "ValidationEngine",
    "LearningLoopScheduler",
    "SchedulerCycleResult",
]
