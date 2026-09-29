"""Critical Prediction Evaluation Test for Phase 7 (Section 51).

Verifies:
1. Creation of immutable PredictionRecord at timestamp T with horizon H.
2. After H elapses, OutcomeTracker fetches market outcome.
3. Computes realized return, MFE, MAE, actual_outcome ("CORRECT", "INCORRECT", "NEUTRAL"),
   and prediction error.
4. Appends evaluation fields to ledger.
5. Invariant check: original prediction fields (timestamp, entry_price, expected_return,
   confidence, signal_action, etc.) remain 100% UNCHANGED.
"""

from datetime import UTC, datetime, timedelta

import pytest

from data.ingestion.models import Candle
from paper_trading.models import PredictionRecord, PredictionStatus
from paper_trading.outcome_tracker import OutcomeTracker
from paper_trading.prediction_ledger import PredictionLedger


class TestCriticalPredictionEvaluation:
    """Verifies non-destructive outcome evaluation and excursion metrics."""

    def test_section_51_prediction_evaluation_and_immutability(self):
        ledger = PredictionLedger()
        tracker = OutcomeTracker(prediction_ledger=ledger, default_horizon_seconds=900.0)

        t0 = datetime(2026, 9, 29, 10, 0, 0, tzinfo=UTC)

        # 1. Create Prediction at timestamp T0 with 15m horizon
        original_pred = PredictionRecord(
            prediction_id="pred_test_sec51",
            symbol="BTCUSDT",
            timestamp=t0,
            timeframe="1m",
            model_version="v1.0.0",
            feature_version="v1.0.0",
            signal_version="v1.0.0",
            signal_action="BUY",
            confidence=0.82,
            regime="TREND_UP",
            setup="STANDARD",
            expected_return=0.015,
            expected_volatility=0.008,
            entry_price=50000.0,
            stop_price=49500.0,
            target_price=51000.0,
            evaluation_horizon_seconds=900.0,  # 15 minutes
            status=PredictionStatus.PENDING,
        )
        ledger.record_prediction(original_pred)

        # Snapshot original values for immutability check
        orig_id = original_pred.prediction_id
        orig_ts = original_pred.timestamp
        orig_conf = original_pred.confidence
        orig_action = original_pred.signal_action
        orig_entry = original_pred.entry_price
        orig_exp_ret = original_pred.expected_return
        orig_stop = original_pred.stop_price
        orig_target = original_pred.target_price

        # Check before horizon: 5 minutes later (no evaluation yet)
        t_early = t0 + timedelta(minutes=5)
        evaluated_early = tracker.check_pending_evaluations(current_time=t_early)
        assert len(evaluated_early) == 0
        assert original_pred.status == PredictionStatus.PENDING

        # Feed candles spanning the 15-minute horizon
        # Market moves up to 51,000 (high) and low is 49,900, closes at 50,800
        for i in range(16):
            c_time = t0 + timedelta(minutes=i)
            c = Candle(
                source="binance",
                symbol="BTCUSDT",
                timeframe="1m",
                start_time=c_time,
                end_time=c_time + timedelta(minutes=1),
                open=50000.0 + i * 50,
                high=50050.0 + i * 60,
                low=49950.0 + i * 40,
                close=50020.0 + i * 52,
                volume=100.0,
            )
            tracker.record_candle(c)

        # 2. Check after horizon: 16 minutes later (horizon H = 15m exceeded)
        t_eval = t0 + timedelta(minutes=16)
        evaluated = tracker.check_pending_evaluations(
            current_time=t_eval,
            current_prices={"BTCUSDT": 50800.0},
        )

        assert len(evaluated) == 1
        evaluated_pred = evaluated[0]

        # 3. Verify evaluated fields
        assert evaluated_pred.status == PredictionStatus.EVALUATED
        assert evaluated_pred.evaluation_time == t_eval
        # Realized return = (50800 - 50000) / 50000 = +0.016 (+1.6%)
        assert pytest.approx(evaluated_pred.realized_return, rel=1e-3) == 0.016
        # Market moved up for a BUY prediction -> CORRECT
        assert evaluated_pred.actual_outcome == "CORRECT"
        # MFE must be positive
        assert evaluated_pred.mfe > 0.0
        # MAE must be non-negative
        assert evaluated_pred.mae >= 0.0
        # Prediction error: abs(0.016 - 0.015) = 0.001
        assert pytest.approx(evaluated_pred.prediction_error, rel=1e-3) == 0.001

        # 4. Invariant Check: Original prediction fields remain 100% UNCHANGED
        assert evaluated_pred.prediction_id == orig_id
        assert evaluated_pred.timestamp == orig_ts
        assert evaluated_pred.confidence == orig_conf
        assert evaluated_pred.signal_action == orig_action
        assert evaluated_pred.entry_price == orig_entry
        assert evaluated_pred.expected_return == orig_exp_ret
        assert evaluated_pred.stop_price == orig_stop
        assert evaluated_pred.target_price == orig_target
