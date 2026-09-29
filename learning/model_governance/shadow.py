"""Shadow Model Inference and Evaluation Engine for Phase 8 (Sections 28, 29).

Enables side-by-side evaluation of Candidate vs Production models on identical live observations:
- Production model: ACTIVE (drives paper signals and portfolio execution)
- Candidate model: SHADOW (receives identical features, records predictions, does NOT place orders)
- Shadow Model Ledger: Records timestamp, production prediction, candidate prediction, and eventual outcome.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from models.base import BaselineModel
from monitoring.logging import get_logger
from paper_trading.models import PredictionRecord

logger = get_logger(__name__)


@dataclass
class ShadowComparisonRecord:
    """Side-by-side record of production vs shadow candidate predictions on identical inputs."""
    comparison_id: str = field(default_factory=lambda: f"shd_{uuid.uuid4().hex[:10]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    symbol: str = "BTCUSDT"
    production_model_version: str = ""
    candidate_model_version: str = ""
    production_action: str = ""
    candidate_action: str = ""
    production_confidence: float = 0.0
    candidate_confidence: float = 0.0
    actual_outcome: str | None = None
    realized_return: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "production_model_version": self.production_model_version,
            "candidate_model_version": self.candidate_model_version,
            "production_action": self.production_action,
            "candidate_action": self.candidate_action,
            "production_confidence": round(self.production_confidence, 4),
            "candidate_confidence": round(self.candidate_confidence, 4),
            "actual_outcome": self.actual_outcome,
            "realized_return": self.realized_return,
        }


class ShadowModelManager:
    """Coordinates shadow model execution alongside active production models."""

    def __init__(self) -> None:
        self._shadow_models: dict[str, BaselineModel] = {}
        self._ledger: list[ShadowComparisonRecord] = []

    def register_shadow_candidate(self, candidate_model: BaselineModel) -> None:
        """Register a candidate model to run in shadow mode."""
        self._shadow_models[candidate_model.model_version] = candidate_model
        logger.info("shadow_candidate_registered", version=candidate_model.model_version)

    def evaluate_shadow(
        self,
        symbol: str,
        features_df: pd.DataFrame,
        production_prediction: PredictionRecord,
        timestamp: datetime | None = None,
    ) -> list[ShadowComparisonRecord]:
        """Run shadow inference across registered candidate models on identical features."""
        now = timestamp or datetime.now(UTC)
        records: list[ShadowComparisonRecord] = []

        if features_df.empty or not self._shadow_models:
            return records

        for cand_ver, cand_model in self._shadow_models.items():
            try:
                probs = cand_model.predict_proba(features_df)
                prob_up = float(probs[-1][1]) if len(probs[-1]) > 1 else float(probs[-1][0])

                if prob_up > 0.55:
                    cand_action = "BUY"
                    cand_conf = (prob_up - 0.50) * 2.0
                elif prob_up < 0.45:
                    cand_action = "SELL"
                    cand_conf = (0.50 - prob_up) * 2.0
                else:
                    cand_action = "NO_TRADE"
                    cand_conf = 0.0

                rec = ShadowComparisonRecord(
                    timestamp=now,
                    symbol=symbol.upper(),
                    production_model_version=production_prediction.model_version,
                    candidate_model_version=cand_ver,
                    production_action=production_prediction.signal_action,
                    candidate_action=cand_action,
                    production_confidence=production_prediction.confidence,
                    candidate_confidence=cand_conf,
                )
                self._ledger.append(rec)
                records.append(rec)

            except Exception as e:
                logger.error("shadow_inference_failed", candidate=cand_ver, error=str(e))

        return records

    def append_outcome(
        self,
        symbol: str,
        timestamp: datetime,
        actual_outcome: str,
        realized_return: float,
    ) -> None:
        """Append ground-truth outcomes to shadow comparison records for analysis."""
        for rec in self._ledger:
            if rec.symbol == symbol and rec.actual_outcome is None:
                # Approximate timestamp matching within 60s
                diff = abs((rec.timestamp - timestamp).total_seconds())
                if diff < 120.0:
                    rec.actual_outcome = actual_outcome
                    rec.realized_return = realized_return

    def get_comparison_summary(self) -> dict[str, Any]:
        """Produce comparison metrics between production and shadow candidates."""
        evaluated = [r for r in self._ledger if r.actual_outcome is not None]
        if not evaluated:
            return {"total_observations": len(self._ledger), "evaluated_count": 0}

        prod_correct = sum(1 for r in evaluated if (r.production_action == "BUY" and (r.realized_return or 0) > 0) or (r.production_action == "SELL" and (r.realized_return or 0) < 0))
        cand_correct = sum(1 for r in evaluated if (r.candidate_action == "BUY" and (r.realized_return or 0) > 0) or (r.candidate_action == "SELL" and (r.realized_return or 0) < 0))

        return {
            "total_observations": len(self._ledger),
            "evaluated_observations": len(evaluated),
            "production_accuracy": round(prod_correct / len(evaluated), 4),
            "candidate_accuracy": round(cand_correct / len(evaluated), 4),
        }

    def get_ledger(self) -> list[ShadowComparisonRecord]:
        return list(self._ledger)
