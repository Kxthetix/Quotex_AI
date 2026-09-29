"""Critical anti-leakage verification tests for Phase 4.

Mandatory tests enforcing Section 47 (Features at T vs Labels at T)
and Section 48 (Strict chronological train/val/test boundary without data leakage).
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from features.engine import FeatureEngine
from learning.dataset import ChronologicalPreprocessor, ChronologicalSplitter, DatasetGenerator
from learning.evaluation.calibration import ProbabilityCalibrator
from learning.labels.engine import LabelEngine
from models.direction.model import DirectionModel


def _generate_synthetic_candles(
    n_candles: int = 150,
    start_time: datetime | None = None,
    base_price: float = 100.0,
    interval_seconds: int = 60,
) -> pd.DataFrame:
    """Generate deterministic synthetic candles."""
    start = start_time or datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    records = []
    price = base_price
    np.random.seed(42)

    for i in range(n_candles):
        t_start = start + timedelta(seconds=i * interval_seconds)
        t_end = t_start + timedelta(seconds=interval_seconds)
        ret = np.sin(i / 10.0) * 0.005 + 0.001
        o = price
        c = price * (1.0 + ret)
        h = max(o, c) * 1.002
        l = min(o, c) * 0.998
        v = 1000.0 + (i % 20) * 50.0
        price = c

        records.append(
            {
                "start_time": t_start,
                "end_time": t_end,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )

    return pd.DataFrame(records)


class TestCriticalLeakageSeparation:
    """Automated verification of Section 47: Features at T vs Labels at T."""

    def test_section_47_critical_label_feature_separation(self) -> None:
        """Section 47 Verification:

        1. Generate features at T.
        2. Generate label at T.
        3. Add future market data.
        4. Recalculate features at T.
        5. Assert features(T) have not changed.
        6. Confirm label(T) changes only if new future candles complete the required horizon.
        """
        feature_engine = FeatureEngine()
        label_engine = LabelEngine()

        # Step 1 & 2: Generate candles up to timestamp T (e.g. 60 candles)
        candles_t = _generate_synthetic_candles(n_candles=60)
        target_idx = 40
        time_at_t = candles_t.loc[target_idx, "end_time"]

        features_at_t_initial = feature_engine.compute_features_dataframe(candles_t)
        labels_at_t_initial = label_engine.compute_labels(candles_t)

        vector_at_t_initial = features_at_t_initial.iloc[target_idx].to_dict()
        label_at_t_initial = labels_at_t_initial.iloc[target_idx].to_dict()

        # Step 3: Add future market data after T (e.g. 50 additional candles)
        extended_candles = _generate_synthetic_candles(n_candles=110)
        # Ensure the first 60 candles are bit-for-bit identical
        for col in ["open", "high", "low", "close", "volume"]:
            np.testing.assert_array_equal(
                candles_t[col].values,
                extended_candles.iloc[:60][col].values,
            )

        # Step 4: Recalculate features on extended candles
        features_extended = feature_engine.compute_features_dataframe(extended_candles)
        labels_extended = label_engine.compute_labels(extended_candles)

        vector_at_t_after_future = features_extended.iloc[target_idx].to_dict()
        label_at_t_after_future = labels_extended.iloc[target_idx].to_dict()

        # Step 5: Assert features(T) have NOT changed at all!
        for feat_name, feat_val in vector_at_t_initial.items():
            val_after = vector_at_t_after_future[feat_name]
            if np.isnan(feat_val):
                assert np.isnan(val_after), f"Feature {feat_name} changed from NaN to {val_after}"
            else:
                assert np.isclose(feat_val, val_after, atol=1e-8), (
                    f"Leakage detected! Feature '{feat_name}' at T={time_at_t} changed when future candles were added. "
                    f"Initial={feat_val}, After future data={val_after}"
                )

        # Step 6: Verify future label calculation behavior
        # Look at horizon 60 (bar index 100). For an observation at index 55 in candles_t (only 60 candles total),
        # horizon 15 (requires index 70) was NaN in initial candles, but becomes populated in extended candles.
        tail_idx = 55
        initial_tail_labels = label_engine.compute_labels(candles_t)
        extended_tail_labels = label_engine.compute_labels(extended_candles)

        # In candles_t (length 60), index 55 + horizon 15 = index 70 (> 59), so direction_15 must be NaN initially
        assert np.isnan(initial_tail_labels.loc[tail_idx, "direction_15"])

        # In extended_candles (length 110), index 55 + horizon 15 = index 70 (<= 109), so direction_15 must now be computed!
        assert not np.isnan(extended_tail_labels.loc[tail_idx, "direction_15"])

        # But features at index 55 MUST remain identical!
        feat_initial_55 = features_at_t_initial.iloc[tail_idx].to_dict()
        feat_extended_55 = features_extended.iloc[tail_idx].to_dict()
        for feat_name, feat_val in feat_initial_55.items():
            val_after = feat_extended_55[feat_name]
            if np.isnan(feat_val):
                assert np.isnan(val_after)
            else:
                assert np.isclose(feat_val, val_after, atol=1e-8)


class TestCriticalTrainTestBoundary:
    """Automated verification of Section 48: Strict chronological split & no leakage into preprocessing."""

    def test_section_48_chronological_ordering_and_preprocessor_isolation(self) -> None:
        """Section 48 Verification:

        1. Verify max(training_timestamp) < min(test_timestamp).
        2. Verify no test observations are used during:
           - scaling / preprocessing
           - model fitting
           - probability calibration
        """
        candles = _generate_synthetic_candles(n_candles=200)
        generator = DatasetGenerator()
        df, metadata = generator.create_dataset(candles, symbol="BTC/USDT", timeframe="1m")

        splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        splits = splitter.split(df, metadata)

        # 1. Enforce strict temporal ordering
        max_train_time = splits.train_df["timestamp"].max()
        min_val_time = splits.val_df["timestamp"].min()
        max_val_time = splits.val_df["timestamp"].max()
        min_test_time = splits.test_df["timestamp"].min()

        assert max_train_time < min_val_time, (
            f"Train and Val overlap! max_train={max_train_time}, min_val={min_val_time}"
        )
        assert max_val_time < min_test_time, (
            f"Val and Test overlap! max_val={max_val_time}, min_test={min_test_time}"
        )
        assert max_train_time < min_test_time, (
            f"Train and Test overlap! max_train={max_train_time}, min_test={min_test_time}"
        )

        # 2. Test preprocessor isolation:
        # Preprocessor fitted ONLY on train
        preprocessor = ChronologicalPreprocessor(feature_columns=metadata.feature_columns)
        preprocessor.fit(splits.train_df)

        train_mean = preprocessor.scaler.mean_.copy()
        train_scale = preprocessor.scaler.scale_.copy()

        # Transform val and test using parameters learned ONLY from train
        X_val_scaled = preprocessor.transform(splits.val_df)
        X_test_scaled = preprocessor.transform(splits.test_df)

        # Assert preprocessor internal parameters did NOT change when transforming test data
        np.testing.assert_array_equal(preprocessor.scaler.mean_, train_mean)
        np.testing.assert_array_equal(preprocessor.scaler.scale_, train_scale)

        # Now, deliberately perturb test data with massive outliers
        perturbed_test_df = splits.test_df.copy()
        feat_col = metadata.feature_columns[0]
        perturbed_test_df[feat_col] = perturbed_test_df[feat_col] * 1000.0 + 50000.0

        # Preprocessor transform on perturbed test should NOT mutate the scaler
        _ = preprocessor.transform(perturbed_test_df)
        np.testing.assert_array_equal(preprocessor.scaler.mean_, train_mean)
        np.testing.assert_array_equal(preprocessor.scaler.scale_, train_scale)

        # 3. Model fitting isolation
        X_train_scaled = preprocessor.transform(splits.train_df)
        y_train = splits.train_df["direction_15"].values.astype(int)

        model = DirectionModel(
            algorithm="logistic_regression", expected_features=metadata.feature_columns
        )
        model.fit(X_train_scaled, y_train)

        # 4. Calibration isolation: Calibrator fitted strictly on validation predictions
        y_val = splits.val_df["direction_15"].values.astype(int)
        val_probs = model.predict_proba(X_val_scaled)[:, 1]

        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(val_probs, y_val)

        # Test set is purely evaluated, never fitted
        test_probs = model.predict_proba(X_test_scaled)[:, 1]
        calibrated_test_probs = calibrator.calibrate(test_probs)

        assert len(calibrated_test_probs) == len(splits.test_df)
        assert np.all(calibrated_test_probs >= 0.0)
        assert np.all(calibrated_test_probs <= 1.0)
