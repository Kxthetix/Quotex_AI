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
        "--candles", type=int, default=None, help="Number of candles for training dataset"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to real Parquet dataset (e.g. data/datasets/market_regimes/...)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=15,
        help="Prediction horizon in bars (e.g. 5, 15, 60, 240 bars)",
    )
    parser.add_argument(
        "--fee-hurdle-bps",
        type=float,
        default=0.0,
        help="Fee hurdle in basis points (e.g. 15.0 bps for friction-cleared direction)",
    )
    args = parser.parse_args()

    if args.data_file:
        from data.storage.parquet_store import load_candles_from_parquet

        data_path = Path(args.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {data_path}")
        candles = load_candles_from_parquet(data_path)
        if args.candles and len(candles) > args.candles:
            candles = candles[: args.candles]
        print(f"Loaded {len(candles)} real candles from {data_path.resolve()}.")
        symbol = candles[0].symbol if candles else "BTC/USDT"
        timeframe = candles[0].timeframe if candles else "1h"
    else:
        candle_count = args.candles if args.candles is not None else 600
        print(f"Generating synthetic dataset with {candle_count} candles...")
        candles = generate_synthetic_candles(count=candle_count, symbol="BTC/USDT")
        symbol = "BTC/USDT"
        timeframe = "1m"

    from learning.labels.engine import LabelEngine

    label_engine = LabelEngine(
        horizons=[args.horizon],
        fee_hurdle_bps=args.fee_hurdle_bps,
    )
    generator = DatasetGenerator(label_engine=label_engine)
    df, metadata = generator.create_dataset(candles, symbol=symbol, timeframe=timeframe)

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

    h = args.horizon
    dir_col = (
        f"direction_hurdle_{h}"
        if args.fee_hurdle_bps > 0 and f"direction_hurdle_{h}" in splits.train_df.columns
        else f"direction_{h}"
    )
    ret_col = (
        f"target_return_{h}"
        if f"target_return_{h}" in splits.train_df.columns
        else f"return_{h}"
    )
    vol_col = (
        f"target_volatility_{h}"
        if f"target_volatility_{h}" in splits.train_df.columns
        else f"volatility_{h}"
    )
    tq_col = f"trade_quality_{h}"

    print(f"\n--- Training Target: Horizon {h} bars (Direction: {dir_col}, Return: {ret_col}) ---")

    if args.model == "direction":
        y_train = splits.train_df[dir_col].values.astype(int)
        y_val = splits.val_df[dir_col].values.astype(int)
        y_test = splits.test_df[dir_col].values.astype(int)

        model = DirectionModel(
            algorithm=algo,
            model_name=f"direction_{algo}_h{h}",
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
        y_train = splits.train_df[dir_col].values.astype(int)
        y_test = splits.test_df[dir_col].values.astype(int)

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
        y_train = splits.train_df[ret_col].values
        y_test = splits.test_df[ret_col].values

        model = ReturnModel(
            algorithm=algo,
            model_name=f"return_{algo}_h{h}",
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
        y_train = splits.train_df[ret_col].values
        y_test = splits.test_df[ret_col].values

        model = HistoricalMeanModel(expected_features=metadata.feature_columns)
        model.fit(x_train, y_train)
        test_preds = model.predict(x_test)

        metrics = evaluate_regressor(y_test, test_preds)
        metrics_dict = metrics.to_dict()

        print("\n=== Statistical Baseline: Historical Mean (Test Split) ===")
        for k, v in metrics_dict.items():
            print(f"  {k:15s}: {v}")

    elif args.model == "volatility":
        y_train = splits.train_df[vol_col].values
        y_test = splits.test_df[vol_col].values

        model = VolatilityModel(
            algorithm=algo,
            model_name=f"volatility_{algo}_h{h}",
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
        y_train = splits.train_df[tq_col].values.astype(int)
        y_test = splits.test_df[tq_col].values.astype(int)

        model = TradeQualityModel(
            algorithm=algo,
            model_name=f"trade_quality_{algo}_h{h}",
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
