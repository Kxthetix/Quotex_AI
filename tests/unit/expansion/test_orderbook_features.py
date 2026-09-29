"""Unit tests for Order Book Microstructure, Gap Detection, and Features (Sections 10-18, 55)."""

from datetime import UTC, datetime, timedelta

import pytest

from backtesting.orderbook_cost_model import OrderBookCostModel
from data.orderbook.models import (
    OrderBookLevel,
    OrderBookSnapshot,
    OrderBookUpdate,
)
from data.orderbook.tracker import OrderBookStatus, OrderBookTracker
from features.orderbook_features import (
    OrderBookFeatureEngine,
    OrderBookFeatures,
)


def create_sample_snapshot(seq: int = 100, t: datetime | None = None) -> OrderBookSnapshot:
    now = t or datetime.now(UTC)
    bids = [
        OrderBookLevel(price=50000.0, size=2.0, level=1),
        OrderBookLevel(price=49990.0, size=3.0, level=2),
        OrderBookLevel(price=49980.0, size=5.0, level=3),
        OrderBookLevel(price=49970.0, size=4.0, level=4),
        OrderBookLevel(price=49960.0, size=6.0, level=5),
    ]
    asks = [
        OrderBookLevel(price=50010.0, size=1.0, level=1),
        OrderBookLevel(price=50020.0, size=4.0, level=2),
        OrderBookLevel(price=50030.0, size=2.0, level=3),
        OrderBookLevel(price=50040.0, size=5.0, level=4),
        OrderBookLevel(price=50050.0, size=8.0, level=5),
    ]
    return OrderBookSnapshot(
        timestamp=now,
        asset_id="asset_btc_usdt",
        symbol="BTC/USDT",
        provider="binance",
        sequence_number=seq,
        bids=bids,
        asks=asks,
    )


def test_orderbook_feature_calculation_all_metrics():
    snap = create_sample_snapshot()
    features = OrderBookFeatureEngine.calculate_features(snap)

    assert features.feature_version == "orderbook_features_v1"
    assert features.bid_ask_spread == 10.0  # 50010 - 50000
    assert features.relative_spread > 0.0
    assert -1.0 <= features.top_of_book_imbalance <= 1.0
    assert -1.0 <= features.n_level_imbalance <= 1.0
    assert -1.0 <= features.depth_weighted_imbalance <= 1.0
    assert features.microprice > 50000.0
    assert 0.0 <= features.book_pressure <= 1.0
    assert 0.0 <= features.liquidity_concentration <= 1.0


def test_orderbook_temporal_leakage_protection_section_14():
    """Section 14: Feature at time T may only use information available at or before T."""
    t_decision = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)

    # Valid: snapshot at or before decision time
    past_snap = create_sample_snapshot(t=t_decision - timedelta(seconds=1))
    features = OrderBookFeatureEngine.calculate_features(past_snap, as_of_time=t_decision)
    assert features is not None

    # Violation: snapshot from future relative to decision time
    future_snap = create_sample_snapshot(t=t_decision + timedelta(seconds=2))
    with pytest.raises(ValueError) as exc:
        OrderBookFeatureEngine.calculate_features(future_snap, as_of_time=t_decision)
    assert "Temporal leakage violation" in str(exc.value)


def test_critical_orderbook_gap_test_section_55():
    """Section 55 Critical Order-Book Test:
    Snapshot + Incremental updates -> inject sequence gap ->
    book becomes invalid -> feature generation stops -> recovery requested ->
    valid snapshot restores state -> processing resumes.
    """
    tracker = OrderBookTracker(asset_id="asset_btc_usdt", symbol="BTC/USDT")
    snap1 = create_sample_snapshot(seq=100)

    # 1. Initialize tracker with Snapshot
    assert tracker.apply_snapshot(snap1) is True
    assert tracker.status == OrderBookStatus.VALID

    # 2. Incremental update seq 101 succeeds
    upd101 = OrderBookUpdate(
        timestamp=datetime.now(UTC),
        asset_id="asset_btc_usdt",
        symbol="BTC/USDT",
        provider="binance",
        sequence_number=101,
        bids_delta=[(50000.0, 3.5)],
    )
    ok, status, _ = tracker.apply_update(upd101)
    assert ok is True
    assert status == OrderBookStatus.VALID

    # 3. Inject sequence gap: update 105 received (missing 102, 103, 104)
    upd105 = OrderBookUpdate(
        timestamp=datetime.now(UTC),
        asset_id="asset_btc_usdt",
        symbol="BTC/USDT",
        provider="binance",
        sequence_number=105,
        bids_delta=[(49990.0, 4.0)],
    )
    ok, status, msg = tracker.apply_update(upd105)

    # Book MUST enter GAP_DETECTED
    assert ok is False
    assert status == OrderBookStatus.GAP_DETECTED
    assert "Sequence gap detected" in msg

    # Feature generation stops: get_snapshot returns None
    corrupted_snap = tracker.get_snapshot()
    assert corrupted_snap is None

    # 4. Snapshot recovery: apply fresh full snapshot at seq 105
    recovery_snap = create_sample_snapshot(seq=105)
    assert tracker.apply_snapshot(recovery_snap) is True
    assert tracker.status == OrderBookStatus.VALID

    # 5. Processing cleanly resumes
    restored_snap = tracker.get_snapshot()
    assert restored_snap is not None
    features = OrderBookFeatureEngine.calculate_features(restored_snap)
    assert features.bid_ask_spread > 0


def test_orderbook_cost_model_depth_walking():
    snap = create_sample_snapshot()
    cost_model = OrderBookCostModel(fee_bps=5.0)

    # Order size 1.0 matches best ask (1.0 at 50010.0) -> zero depth slippage
    est1 = cost_model.estimate_execution(snap, side="BUY", size=1.0)
    assert est1.effective_avg_price == 50010.0
    assert est1.slippage_bps == 0.0
    assert est1.depth_exhausted_levels == 1

    # Order size 3.0 exceeds level 1 (1.0 @ 50010) and takes 2.0 from level 2 (4.0 @ 50020)
    # Expected avg = (1.0*50010 + 2.0*50020) / 3.0 = 150050 / 3 = 50016.6667
    est2 = cost_model.estimate_execution(snap, side="BUY", size=3.0)
    assert est2.effective_avg_price > 50010.0
    assert est2.slippage_bps > 0.0
    assert est2.depth_exhausted_levels == 2
    assert est2.unfilled_size == 0.0
