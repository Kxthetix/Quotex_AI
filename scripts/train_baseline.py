"""Command line script to train, calibrate, evaluate, and register baseline models (Phase 4).

Usage:
    python scripts/train_baseline.py --model direction --algo logistic_regression
    python scripts/train_baseline.py --model return --algo ridge
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learning.dataset import ChronologicalPreprocessor, ChronologicalSplitter, DatasetGenerator
from learning.evaluation.calibration import (
    ProbabilityCalibrator,
    calculate_expected_calibration_error,
)
from learning.evaluation.metrics import evaluate_classifier, evaluate_regressor
from learning.tracking import ExperimentTracker
from models.base import ModelStatus
from models.baseline.statistical import HistoricalMeanModel, MajorityClassModel
from models.direction.model import DirectionModel
from models.registry import LocalModelRegistry
from models.returns.model import ReturnModel
from models.trade_quality.model import TradeQualityModel
from models.volatility.model import VolatilityModel
from scripts.calculate_features import generate_synthetic_candles


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and ML models for Phase 4.")
    parser.add_argument(
        "--model",
        type=str,
        default="direction",
        choices=[
            "direction",
            "return",
            "volatility",
            "trade_quality",
            "majority_class",
            "historical_mean",
        ],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="default",
        help="Algorithm variant (e.g. logistic_regression, random_forest, lightgbm, ridge)",
    )
    parser.add_argument(
        "--candles", type=int, default=600, help="Number of candles for training dataset"
    )
    args = parser.parse_args()

    print(f"Generating dataset with {args.candles} candles...")
    candles = generate_synthetic_candles(count=args.candles, symbol="BTC/USDT")
    generator = DatasetGenerator()
    df, metadata = generator.create_dataset(candles, symbol="BTC/USDT", timeframe="1m")

    # Chronological Split: 70% Train, 15% Val, 15% Test
    splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split(df, metadata)

    # Preprocessor learned ONLY on train
    preprocessor = ChronologicalPreprocessor(feature_columns=metadata.feature_columns)
    preprocessor.fit(splits.train_df)

    x_train = preprocessor.transform(splits.train_df)
    x_val = preprocessor.transform(splits.val_df)
    x_test = preprocessor.transform(splits.test_df)

    # Initialize model
    algo = (
        "logistic_regression"
        if args.algo == "default" and args.model in ("direction", "trade_quality")
        else ("ridge" if args.algo == "default" else args.algo)
    )

    registry = LocalModelRegistry()
    tracker = ExperimentTracker()

    if args.model == "direction":
        y_train = splits.train_df["direction_15"].values.astype(int)
        y_val = splits.val_df["direction_15"].values.astype(int)
        y_test = splits.test_df["direction_15"].values.astype(int)

        model = DirectionModel(
            algorithm=algo,
            model_name=f"direction_{algo}",
            expected_features=metadata.feature_columns,
        )
        model.fit(x_train, y_train)

        # Calibrate on validation set
        val_probs = model.predict_proba(x_val)
        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(val_probs, y_val)

        # Evaluate on test set
        raw_test_probs = model.predict_proba(x_test)
        cal_test_probs = calibrator.calibrate(raw_test_probs)
        test_preds = cal_test_probs.argmax(axis=1)

        metrics = evaluate_classifier(y_test, test_preds, cal_test_probs)
        ece = calculate_expected_calibration_error(y_test, cal_test_probs)
        metrics_dict = metrics.to_dict()
        metrics_dict["ece"] = round(ece, 5)

        print("\n=== Model Evaluation (Test Split) ===")
        for k, v in metrics_dict.items():
            print(f"  {k:15s}: {v}")

    elif args.model == "majority_class":
        y_train = splits.train_df["direction_15"].values.astype(int)
        y_test = splits.test_df["direction_15"].values.astype(int)

        model = MajorityClassModel(expected_features=metadata.feature_columns)
        model.fit(x_train, y_train)
        test_preds = model.predict(x_test)
        test_probs = model.predict_proba(x_test)

        metrics = evaluate_classifier(y_test, test_preds, test_probs)
        metrics_dict = metrics.to_dict()

        print("\n=== Statistical Baseline: Majority Class (Test Split) ===")
        for k, v in metrics_dict.items():
            print(f"  {k:15s}: {v}")

    elif args.model == "return":
        y_train = splits.train_df["target_return_15"].values
        y_test = splits.test_df["target_return_15"].values

        model = ReturnModel(
            algorithm=algo,
            model_name=f"return_{algo}",
            expected_features=metadata.feature_columns,
        )
        model.fit(x_train, y_train)
        test_preds = model.predict(x_test)

        metrics = evaluate_regressor(y_test, test_preds)
        metrics_dict = metrics.to_dict()

        print("\n=== Return Regressor Evaluation (Test Split) ===")
        for k, v in metrics_dict.items():
            print(f"  {k:15s}: {v}")

    elif args.model == "historical_mean":
        y_train = splits.train_df["target_return_15"].values
        y_test = splits.test_df["target_return_15"].values

        model = HistoricalMeanModel(expected_features=metadata.feature_columns)
        model.fit(x_train, y_train)
        test_preds = model.predict(x_test)

        metrics = evaluate_regressor(y_test, test_preds)
        metrics_dict = metrics.to_dict()

        print("\n=== Statistical Baseline: Historical Mean (Test Split) ===")
        for k, v in metrics_dict.items():
            print(f"  {k:15s}: {v}")

    elif args.model == "volatility":
        y_train = splits.train_df["volatility_15"].values
        y_test = splits.test_df["volatility_15"].values

        model = VolatilityModel(
            algorithm=algo,
            model_name=f"volatility_{algo}",
            expected_features=metadata.feature_columns,
        )
        model.fit(x_train, y_train)
        test_preds = model.predict(x_test)

        metrics = evaluate_regressor(y_test, test_preds)
        metrics_dict = metrics.to_dict()

        print("\n=== Volatility Regressor Evaluation (Test Split) ===")
        for k, v in metrics_dict.items():
            print(f"  {k:15s}: {v}")

    elif args.model == "trade_quality":
        y_train = splits.train_df["trade_quality_15"].values.astype(int)
        y_test = splits.test_df["trade_quality_15"].values.astype(int)

        model = TradeQualityModel(
            algorithm=algo,
            model_name=f"trade_quality_{algo}",
            expected_features=metadata.feature_columns,
        )
        model.fit(x_train, y_train)
        test_preds = model.predict(x_test)
        test_probs = model.predict_proba(x_test)

        metrics = evaluate_classifier(y_test, test_preds, test_probs)
        metrics_dict = metrics.to_dict()

        print("\n=== Trade-Quality Classifier Evaluation (Test Split) ===")
        for k, v in metrics_dict.items():
            print(f"  {k:15s}: {v}")

    # Register model in LocalModelRegistry with CANDIDATE status
    artifact_path = registry.register(
        model=model,
        dataset_id=metadata.dataset_id,
        status=ModelStatus.CANDIDATE,
    )
    print(f"\nModel registered in local registry: {artifact_path}")

    # Log to MLflow
    train_period = (splits.train_df["timestamp"].min(), splits.train_df["timestamp"].max())
    run_id = tracker.log_model_run(
        model=model,
        dataset_id=metadata.dataset_id,
        metrics=metrics_dict,
        training_period=train_period,
        artifact_path=artifact_path,
    )
    print(f"Logged experiment run to MLflow [Run ID: {run_id}]")


if __name__ == "__main__":
    main()
