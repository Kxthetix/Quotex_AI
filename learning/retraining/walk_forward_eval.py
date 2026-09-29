"""Walk-Forward Candidate Model Evaluation for Phase 8 (Sections 21, 26).

Reuses Phase 5 Walk-Forward infrastructure to validate candidate models
chronologically across rolling out-of-sample test windows.
Guarantees that candidates are not selected based on a single favorable period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtesting.walk_forward import WalkForwardConfig, WalkForwardEngine, WalkForwardResult
from data.ingestion.models import Candle
from models.base import BaselineModel
from monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CandidateStabilityMetrics:
    """Summary of out-of-sample stability across rolling windows."""
    mean_sharpe: float
    min_sharpe: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    is_stable: bool


class WalkForwardCandidateValidator:
    """Evaluates candidate models across rolling chronological windows."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig(
            train_bars=300,
            val_bars=50,
            test_bars=50,
            step_bars=50,
        )

    def evaluate_candidate(
        self,
        candidate_model: BaselineModel,
        candles: list[Candle],
    ) -> CandidateStabilityMetrics:
        """Run walk-forward simulation across rolling windows."""
        if len(candles) < (self.config.train_bars + self.config.val_bars + self.config.test_bars):
            logger.warning("insufficient_candles_for_walk_forward_validation")
            return CandidateStabilityMetrics(
                mean_sharpe=0.0,
                min_sharpe=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                is_stable=False,
            )

        # Run walk-forward simulation
        engine = WalkForwardEngine(self.config)
        result: WalkForwardResult = engine.run(candles, model_version=candidate_model.model_version)

        # Evaluate window stability
        sharpes = [w.metrics.sharpe_ratio for w in result.windows if not np.isnan(w.metrics.sharpe_ratio)]
        mean_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
        min_sharpe = float(np.min(sharpes)) if sharpes else 0.0
        max_dd = float(np.max([w.metrics.max_drawdown for w in result.windows])) if result.windows else 0.0
        total_trades = sum(w.metrics.total_trades for w in result.windows)

        # Stability rule: non-negative mean Sharpe and max drawdown < 20%
        is_stable = mean_sharpe >= 0.0 and max_dd < 0.20

        return CandidateStabilityMetrics(
            mean_sharpe=round(mean_sharpe, 4),
            min_sharpe=round(min_sharpe, 4),
            max_drawdown=round(max_dd, 4),
            win_rate=round(result.aggregate_metrics.win_rate, 4),
            total_trades=total_trades,
            is_stable=is_stable,
        )
