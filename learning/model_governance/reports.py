"""Audit, Drift, and Model Comparison Report Generators for Phase 8 (Sections 39, 40, 41).

Generates structured Markdown and JSON/dictionary reports:
- ModelAuditReport (Section 39)
- DriftReport (Section 40)
- ModelComparisonReport (Section 41)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from learning.model_governance.models import (
    FeatureDriftEntry,
    ModelAuditResult,
)


class ModelAuditReporter:
    """Generates comprehensive Model Audit Reports (Section 39)."""

    @staticmethod
    def generate_report(audit_result: ModelAuditResult) -> str:
        """Produce structured Markdown audit report matching Section 39 requirements."""
        lines = [
            f"# Model Audit Report: `{audit_result.model_version}`",
            "",
            f"- **Audit ID**: `{audit_result.audit_id}`",
            f"- **Generated At**: {audit_result.timestamp.isoformat()}",
            f"- **Audit Period**: {audit_result.audit_period_start} to {audit_result.audit_period_end}",
            f"- **Recommendation State**: **{audit_result.recommendation_state}**",
            "",
            "## 1. Sample Size & Health Overview",
            f"- Total Window Predictions: {audit_result.prediction_count}",
            f"- Evaluated Ground-Truth Count: {audit_result.performance_metrics.get('evaluated_count', 0)}",
            "",
            "## 2. Predictive Performance Metrics",
        ]

        for k, v in audit_result.performance_metrics.items():
            lines.append(f"- **{k}**: {v}")

        lines.extend([
            "",
            "## 3. Calibration Metrics (Section 16)",
        ])
        for k, v in audit_result.calibration_metrics.items():
            lines.append(f"- **{k}**: {v}")

        lines.extend([
            "",
            "## 4. Drift Summary (Sections 11-15)",
            f"```json",
            json.dumps(audit_result.drift_metrics, indent=2, default=str),
            "```",
            "",
            "## 5. Regime Performance Breakdown (Section 15)",
            "| Regime | Count | Accuracy | Mean Error |",
            "| --- | --- | --- | --- |",
        ])

        for regime, data in audit_result.regime_metrics.items():
            count = data.get("count", 0)
            acc = data.get("accuracy", 0.0)
            err = data.get("mean_error", 0.0)
            lines.append(f"| {regime} | {count} | {acc:.2%} | {err:.6f} |")

        lines.extend([
            "",
            "## 6. Audit Warnings & Degradation Findings",
        ])
        if audit_result.warnings:
            for w in audit_result.warnings:
                lines.append(f"- ⚠️ {w}")
        else:
            lines.append("- No degradation or drift warnings detected. Model nominal.")

        lines.extend([
            "",
            "## 7. Retraining & Lifecycle Recommendation",
            f"State: **{audit_result.recommendation_state}**",
            "",
            "> [!NOTE]",
            "> This audit is advisory evidence. Candidate retraining or promotions require explicit governed workflows.",
        ])

        return "\n".join(lines)


class DriftReporter:
    """Generates detailed Data and Concept Drift Reports (Section 40)."""

    @staticmethod
    def generate_report(
        feature_drift_entries: list[FeatureDriftEntry],
        prediction_drift: dict[str, Any] | None = None,
        concept_drift: dict[str, Any] | None = None,
        regime_drift: dict[str, Any] | None = None,
    ) -> str:
        """Format detailed drift metrics without collapsing into a single score."""
        lines = [
            "# Comprehensive Drift Detection Report",
            f"- Generated At: {datetime.now(UTC).isoformat()}",
            "",
            "## 1. Feature Distribution Drift (Data Drift)",
            "| Feature | Metric | Value | Threshold | Status | Miss% Cur | Miss% Ref |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]

        for entry in feature_drift_entries:
            lines.append(
                f"| `{entry.feature_name}` | {entry.metric_name} | {entry.metric_value:.4f} | "
                f"{entry.threshold:.4f} | **{entry.status}** | {entry.missing_value_rate_current:.2%} | "
                f"{entry.missing_value_rate_reference:.2%} |"
            )

        lines.extend([
            "",
            "## 2. Prediction Output Drift",
        ])
        if prediction_drift:
            lines.extend([
                f"- **Overall Status**: **{prediction_drift.get('status')}**",
                f"- **PSI (Confidence)**: {prediction_drift.get('psi_confidence')}",
                f"- **PSI (Expected Return)**: {prediction_drift.get('psi_expected_return')}",
                f"- **Max Action Proportion Shift**: {prediction_drift.get('max_action_frequency_shift')}",
                f"- Action Distribution Shift: `{prediction_drift.get('reference_action_proportions')}` -> `{prediction_drift.get('current_action_proportions')}`",
            ])
        else:
            lines.append("- No prediction drift data evaluated.")

        lines.extend([
            "",
            "## 3. Concept Drift (Input-to-Outcome Mapping)",
        ])
        if concept_drift:
            lines.extend([
                f"- **Status**: **{concept_drift.get('status')}**",
                f"- **Reference Accuracy**: {concept_drift.get('reference_accuracy')}",
                f"- **Current Accuracy**: {concept_drift.get('current_accuracy')}",
                f"- **Accuracy Drop**: {concept_drift.get('accuracy_drop')}",
                f"- **Error Ratio**: {concept_drift.get('error_ratio')}",
            ])
        else:
            lines.append("- No concept drift data evaluated.")

        lines.extend([
            "",
            "## 4. Market Regime Composition Drift",
        ])
        if regime_drift:
            lines.extend([
                f"- **Status**: **{regime_drift.get('status')}**",
                f"- **Max Regime Shift**: {regime_drift.get('max_regime_shift')}",
                f"- Reference Distribution: `{regime_drift.get('reference_distribution')}`",
                f"- Current Distribution: `{regime_drift.get('current_distribution')}`",
            ])
        else:
            lines.append("- No regime drift data evaluated.")

        return "\n".join(lines)


class ModelComparisonReporter:
    """Generates side-by-side comparison reports between Production and Candidate (Section 41)."""

    @staticmethod
    def generate_report(
        production_meta: dict[str, Any],
        candidate_meta: dict[str, Any],
        comparison_metrics: dict[str, Any],
    ) -> str:
        """Produce side-by-side comparison table clearly attributing metrics to each model."""
        prod_ver = production_meta.get("model_version", "unknown_prod")
        cand_ver = candidate_meta.get("model_version", "unknown_cand")

        lines = [
            f"# Model Comparison: Current Production vs Candidate",
            "",
            f"| Metric / Dimension | Production (`{prod_ver}`) | Candidate (`{cand_ver}`) | Delta / Assessment |",
            "| --- | --- | --- | --- |",
            f"| **Algorithm** | {production_meta.get('algorithm')} | {candidate_meta.get('algorithm')} | - |",
            f"| **Dataset Version** | {production_meta.get('dataset_version')} | {candidate_meta.get('dataset_version')} | - |",
            f"| **Feature Version** | {production_meta.get('feature_version')} | {candidate_meta.get('feature_version')} | - |",
        ]

        # Predictive & Calibration Metrics
        prod_acc = comparison_metrics.get("production_accuracy", 0.0)
        cand_acc = comparison_metrics.get("candidate_accuracy", 0.0)
        acc_delta = cand_acc - prod_acc
        lines.append(f"| **Accuracy** | {prod_acc:.2%} | {cand_acc:.2%} | {'+' if acc_delta >= 0 else ''}{acc_delta:.2%} |")

        prod_ece = comparison_metrics.get("production_ece", 0.0)
        cand_ece = comparison_metrics.get("candidate_ece", 0.0)
        lines.append(f"| **ECE (Calibration)** | {prod_ece:.4f} | {cand_ece:.4f} | {cand_ece - prod_ece:+.4f} |")

        # Trading & Risk Metrics
        prod_sharpe = comparison_metrics.get("production_sharpe", 0.0)
        cand_sharpe = comparison_metrics.get("candidate_sharpe", 0.0)
        lines.append(f"| **Sharpe Ratio** | {prod_sharpe:.2f} | {cand_sharpe:.2f} | {cand_sharpe - prod_sharpe:+.2f} |")

        prod_dd = comparison_metrics.get("production_max_dd", 0.0)
        cand_dd = comparison_metrics.get("candidate_max_dd", 0.0)
        lines.append(f"| **Max Drawdown** | {prod_dd:.2%} | {cand_dd:.2%} | {cand_dd - prod_dd:+.2%} |")

        # Latency / Resource Usage
        prod_lat = comparison_metrics.get("production_latency_ms", 1.2)
        cand_lat = comparison_metrics.get("candidate_latency_ms", 1.3)
        lines.append(f"| **Inference Latency** | {prod_lat:.2f} ms | {cand_lat:.2f} ms | {cand_lat - prod_lat:+.2f} ms |")

        lines.extend([
            "",
            "## Non-Inferiority & Gate Assessment",
            f"- Validation Passed: **{comparison_metrics.get('validation_passed', False)}**",
            f"- Recommendation: **{comparison_metrics.get('recommendation', 'AWAITING_REVIEW')}**",
            "",
            "> [!IMPORTANT]",
            "> Candidate models can NEVER automatically replace production. Explicit human/controlled promotion required.",
        ])

        return "\n".join(lines)
