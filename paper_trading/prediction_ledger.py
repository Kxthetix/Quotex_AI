"""Prediction Ledger for Phase 7 Paper Trading.

Maintains an immutable, append-only ledger of every prediction generated
by the intelligence layer during paper trading.

Evaluation fields (realized return, MFE, MAE, actual outcome, error)
are appended upon reaching the evaluation horizon without mutating
original prediction fields.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from monitoring.logging import get_logger
from paper_trading.models import PredictionRecord, PredictionStatus

logger = get_logger(__name__)


class PredictionLedger:
    """Thread-safe append-only ledger for all model predictions and their eventual outcomes."""

    def __init__(self) -> None:
        self._predictions: dict[str, PredictionRecord] = {}
        self._order: list[str] = []  # chronological order of prediction_ids
        self._pending_ids: set[str] = set()

    def record_prediction(self, prediction: PredictionRecord) -> PredictionRecord:
        """Record a new immutable prediction."""
        pid = prediction.prediction_id
        if pid in self._predictions:
            logger.warning("duplicate_prediction_id_ignored", prediction_id=pid)
            return self._predictions[pid]

        self._predictions[pid] = prediction
        self._order.append(pid)
        if prediction.status == PredictionStatus.PENDING:
            self._pending_ids.add(pid)

        logger.debug(
            "prediction_recorded",
            prediction_id=pid,
            symbol=prediction.symbol,
            action=prediction.signal_action,
            confidence=round(prediction.confidence, 4),
        )
        return prediction

    def append_evaluation(
        self,
        prediction_id: str,
        evaluation_time: datetime,
        realized_return: float,
        mfe: float,
        mae: float,
        actual_outcome: str,
        prediction_error: float,
        metadata: dict[str, Any] | None = None,
    ) -> PredictionRecord:
        """Append evaluation data to an existing prediction record.

        CRITICAL INVARIANT: Original prediction fields (timestamp, confidence,
        expected return, entry/stop/target) are NEVER overwritten.
        """
        pred = self._predictions.get(prediction_id)
        if not pred:
            raise KeyError(f"Prediction {prediction_id} not found in ledger")

        pred.status = PredictionStatus.EVALUATED
        pred.evaluation_time = evaluation_time if evaluation_time.tzinfo else evaluation_time.replace(tzinfo=UTC)
        pred.realized_return = realized_return
        pred.mfe = mfe
        pred.mae = mae
        pred.actual_outcome = actual_outcome
        pred.prediction_error = prediction_error
        if metadata:
            pred.metadata.update(metadata)

        self._pending_ids.discard(prediction_id)
        logger.info(
            "prediction_evaluated",
            prediction_id=prediction_id,
            realized_return=round(realized_return, 4),
            outcome=actual_outcome,
            mfe=round(mfe, 4),
            mae=round(mae, 4),
        )
        return pred

    def get_prediction(self, prediction_id: str) -> PredictionRecord | None:
        return self._predictions.get(prediction_id)

    def get_pending_predictions(self) -> list[PredictionRecord]:
        """Retrieve all predictions awaiting evaluation horizon."""
        return [self._predictions[pid] for pid in self._pending_ids if pid in self._predictions]

    def get_all_predictions(self) -> list[PredictionRecord]:
        return [self._predictions[pid] for pid in self._order]

    def get_evaluated_predictions(self) -> list[PredictionRecord]:
        return [p for p in self._predictions.values() if p.status == PredictionStatus.EVALUATED]

    def count_by_action(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for p in self._predictions.values():
            counts[p.signal_action] += 1
        return dict(counts)

    def count_by_outcome(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for p in self._predictions.values():
            if p.status == PredictionStatus.EVALUATED and p.actual_outcome:
                counts[p.actual_outcome] += 1
        return dict(counts)
