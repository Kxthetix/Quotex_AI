"""Train and run an AI-assisted 30-candle forecast simulation.

The models forecast outcomes over the full horizon. A Monte Carlo path simulation
uses the predicted return and volatility to show a range of possible intermediate
closes; it is not a deterministic 30-candle price prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.regime.engine import MarketRegimeEngine
from analysis.regime.models import MarketRegime, RegimeAssessment
from analysis.technical.rules import TechnicalRuleEngine
from data.storage.parquet_store import load_candles_from_parquet
from features.config import FeatureEngineConfig
from features.engine import FeatureEngine, candles_to_dataframe
from features.models import FeatureVector
from learning.labels.engine import LabelEngine
from learning.dataset import ChronologicalPreprocessor
from models.direction.model import DirectionModel
from models.regime.model import LearnedRegimeModel
from models.returns.model import ReturnModel
from models.trade_quality.model import TradeQualityModel
from models.volatility.model import VolatilityModel
from signals.engine import IntelligenceSnapshot, SignalEngine


def seconds_per_bar(timeframe: str) -> int:
    if len(timeframe) < 2 or not timeframe[:-1].isdigit():
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    unit_seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(timeframe[-1].lower())
    interval = int(timeframe[:-1])
    if unit_seconds is None or interval <= 0:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return interval * unit_seconds


def target_column(name: str) -> str:
    """Keep forward labels disjoint from backward-looking feature names."""
    return f"target_{name}"


def make_feature_vectors(feature_frame: pd.DataFrame, candles, feature_version: str):
    return [
        FeatureVector(
            symbol=candle.symbol,
            timestamp=candle.end_time,
            timeframe=candle.timeframe,
            feature_version=feature_version,
            values={
                name: (float(value) if pd.notna(value) and np.isfinite(value) else None)
                for name, value in row.items()
            },
        )
        for candle, (_, row) in zip(candles, feature_frame.iterrows(), strict=True)
    ]


def future_volatility_labels(closes: np.ndarray, horizon: int, bars_per_year: float) -> np.ndarray:
    """Return annualized realized volatility using the input candle timeframe."""
    result = np.full(len(closes), np.nan, dtype=float)
    log_close = np.log(closes)
    log_returns = np.diff(log_close)
    for index in range(len(closes) - horizon):
        window = log_returns[index : index + horizon]
        if len(window) > 1:
            result[index] = float(np.std(window, ddof=1) * np.sqrt(bars_per_year))
    return result


def build_models(dataset, feature_columns, train_end, validation_end, horizon):
    # Purge a full forecast horizon at each boundary so target windows do not cross partitions.
    train = dataset.iloc[: max(0, train_end - horizon)]
    validation = dataset.iloc[train_end : max(train_end, validation_end - horizon)]
    test = dataset.iloc[validation_end:]
    if len(train) < 500 or len(validation) < 100 or len(test) < 100:
        raise ValueError("Not enough valid rows after purging forecast horizons from the data splits.")

    preprocessor = ChronologicalPreprocessor(feature_columns=feature_columns)
    x_train = preprocessor.fit_transform(train)
    x_validation = preprocessor.transform(validation)
    x_test = preprocessor.transform(test)

    direction = DirectionModel(
        algorithm="random_forest",
        model_name=f"direction_h{horizon}",
        expected_features=feature_columns,
        hyperparameters={"n_estimators": 160, "max_depth": 8, "min_samples_split": 10},
    ).fit(x_train, train[target_column(f"direction_{horizon}")].to_numpy(dtype=int))
    returns = ReturnModel(
        algorithm="ridge",
        model_name=f"return_h{horizon}",
        expected_features=feature_columns,
        hyperparameters={"alpha": 10.0},
    ).fit(x_train, train[target_column(f"return_{horizon}")].to_numpy(dtype=float))
    volatility = VolatilityModel(
        algorithm="ridge",
        model_name=f"volatility_h{horizon}",
        expected_features=feature_columns,
        hyperparameters={"alpha": 10.0},
    ).fit(x_train, train[target_column(f"volatility_{horizon}")].to_numpy(dtype=float))
    quality = TradeQualityModel(
        algorithm="random_forest",
        model_name=f"trade_quality_h{horizon}",
        expected_features=feature_columns,
        hyperparameters={"n_estimators": 160, "max_depth": 8, "min_samples_split": 10},
    ).fit(x_train, train[target_column(f"trade_quality_{horizon}")].to_numpy(dtype=int))
    future_regime_model = LearnedRegimeModel(
        algorithm="random_forest",
        model_name=f"future_regime_h{horizon}",
        expected_features=feature_columns,
        hyperparameters={"n_estimators": 160, "max_depth": 8, "random_state": 42},
    ).fit(x_train, train[target_column(f"future_regime_{horizon}")].astype(str).to_numpy())

    y_direction = test[target_column(f"direction_{horizon}")].to_numpy(dtype=int)
    direction_probs = direction.predict_proba(x_test)
    direction_up_col = int(np.flatnonzero(direction._estimator.classes_ == 1)[0])
    p_up = direction_probs[:, direction_up_col]

    quality_labels = test[target_column(f"trade_quality_{horizon}")].to_numpy(dtype=int)
    quality_probs = quality.predict_proba(x_test)
    quality_pos_col = int(np.flatnonzero(quality._estimator.classes_ == 1)[0])

    train_direction = train[target_column(f"direction_{horizon}")].to_numpy(dtype=int)
    train_quality = train[target_column(f"trade_quality_{horizon}")].to_numpy(dtype=int)
    train_regimes = train[target_column(f"future_regime_{horizon}")].astype(str)
    test_regimes = test[target_column(f"future_regime_{horizon}")].astype(str)
    train_return_mean = float(train[target_column(f"return_{horizon}")].mean())
    train_quality_rate = float(np.mean(train_quality))
    majority_regime = str(train_regimes.mode().iloc[0])
    report = {
        "test_rows": len(test),
        "direction_accuracy": float(accuracy_score(y_direction, direction.predict(x_test))),
        "direction_brier": float(brier_score_loss(y_direction, p_up)),
        "direction_majority_baseline": float(
            max(np.mean(train_direction == 0), np.mean(train_direction == 1))
        ),
        "return_mae": float(
            mean_absolute_error(
                test[target_column(f"return_{horizon}")].to_numpy(dtype=float),
                returns.predict(x_test),
            )
        ),
        "return_mean_baseline_mae": float(
            mean_absolute_error(
                test[target_column(f"return_{horizon}")].to_numpy(dtype=float),
                np.full(len(test), train_return_mean),
            )
        ),
        "volatility_mae": float(
            mean_absolute_error(
                test[target_column(f"volatility_{horizon}")].to_numpy(dtype=float),
                volatility.predict(x_test),
            )
        ),
        "quality_accuracy": float(
            accuracy_score(quality_labels, quality.predict(x_test))
        ),
        "quality_brier": float(
            brier_score_loss(quality_labels, quality_probs[:, quality_pos_col])
        ),
        "quality_majority_baseline": float(
            max(np.mean(train_quality == 0), np.mean(train_quality == 1))
        ),
        "future_regime_accuracy": float(
            accuracy_score(
                test_regimes,
                future_regime_model.predict(x_test).astype(str),
            )
        ),
        "future_regime_majority_baseline": float(np.mean(test_regimes == majority_regime)),
        "validation_rows": len(validation),
        "train_rows": len(train),
    }

    # Refit model instances on all observations whose complete H-bar labels are known.
    all_labeled = dataset
    x_all = preprocessor.__class__(feature_columns=feature_columns).fit_transform(all_labeled)
    final_direction = DirectionModel(
        algorithm="random_forest",
        model_name=f"direction_h{horizon}_final",
        expected_features=feature_columns,
        hyperparameters={"n_estimators": 160, "max_depth": 8, "min_samples_split": 10},
    ).fit(x_all, all_labeled[target_column(f"direction_{horizon}")].to_numpy(dtype=int))
    final_returns = ReturnModel(
        algorithm="ridge",
        model_name=f"return_h{horizon}_final",
        expected_features=feature_columns,
        hyperparameters={"alpha": 10.0},
    ).fit(x_all, all_labeled[target_column(f"return_{horizon}")].to_numpy(dtype=float))
    final_volatility = VolatilityModel(
        algorithm="ridge",
        model_name=f"volatility_h{horizon}_final",
        expected_features=feature_columns,
        hyperparameters={"alpha": 10.0},
    ).fit(x_all, all_labeled[target_column(f"volatility_{horizon}")].to_numpy(dtype=float))
    final_quality = TradeQualityModel(
        algorithm="random_forest",
        model_name=f"trade_quality_h{horizon}_final",
        expected_features=feature_columns,
        hyperparameters={"n_estimators": 160, "max_depth": 8, "min_samples_split": 10},
    ).fit(x_all, all_labeled[target_column(f"trade_quality_{horizon}")].to_numpy(dtype=int))
    final_regime = LearnedRegimeModel(
        algorithm="random_forest",
        model_name=f"future_regime_h{horizon}_final",
        expected_features=feature_columns,
        hyperparameters={"n_estimators": 160, "max_depth": 8, "random_state": 42},
    ).fit(x_all, all_labeled[target_column(f"future_regime_{horizon}")].astype(str).to_numpy())
    return (
        preprocessor.__class__(feature_columns=feature_columns).fit(all_labeled),
        final_direction,
        final_returns,
        final_volatility,
        final_quality,
        final_regime,
        report,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train time-ordered AI models, evaluate a holdout, and simulate the next 30 candles."
    )
    parser.add_argument("--data-file", required=True, help="Completed OHLCV candle Parquet file")
    parser.add_argument("--horizon", type=int, default=30, help="Prediction horizon in candles")
    parser.add_argument("--paths", type=int, default=1500, help="Monte Carlo paths for projected close bands")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible path simulation seed")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("data/models/gbpnzd_real_h1_h30.joblib"),
        help="Path for the fitted research model bundle",
    )
    args = parser.parse_args()
    if args.horizon != 30:
        parser.error("This runner currently presents a 30-candle forecast; --horizon must be 30.")
    if args.paths < 100:
        parser.error("Use at least 100 simulation paths.")

    data_path = Path(args.data_file)
    candles = sorted(load_candles_from_parquet(data_path), key=lambda candle: candle.end_time)
    if len(candles) < 3_000:
        raise ValueError("At least 3,000 historical candles are required to train and evaluate these models.")
    if any(not candle.is_closed for candle in candles):
        raise ValueError("Input must contain only closed candles.")
    if any(
        a.symbol != b.symbol or a.timeframe != b.timeframe
        for a, b in zip(candles, candles[1:])
    ):
        raise ValueError("All candles must share the same symbol and timeframe.")

    timeframe = candles[0].timeframe
    seconds = seconds_per_bar(timeframe)
    bars_per_year = (365.25 * 24 * 60 * 60) / seconds
    feature_config = FeatureEngineConfig(base_timeframe=timeframe)
    feature_engine = FeatureEngine(config=feature_config)
    raw_frame = candles_to_dataframe(candles)
    feature_frame = feature_engine.compute_features_dataframe(raw_frame)
    feature_columns = list(feature_frame.columns)
    vectors = make_feature_vectors(feature_frame, candles, feature_config.version)
    regime_engine = MarketRegimeEngine()
    regime_at_each_bar = np.array(
        [regime_engine.classify(vector).regime.value for vector in vectors], dtype=object
    )

    horizon = args.horizon
    label_frame = LabelEngine(horizons=[horizon]).compute_labels(raw_frame)
    # LabelEngine's original volatility target assumes 1-minute bars; replace it with
    # the same annualized definition using this dataset's actual candle duration.
    label_frame[f"volatility_{horizon}"] = future_volatility_labels(
        raw_frame["close"].to_numpy(dtype=float), horizon, bars_per_year
    )
    future_regime = np.full(len(candles), None, dtype=object)
    future_regime[:-horizon] = regime_at_each_bar[horizon:]

    dataset = feature_frame.copy()
    for name in label_frame.columns:
        dataset[target_column(name)] = label_frame[name].to_numpy()
    dataset[target_column(f"future_regime_{horizon}")] = future_regime
    target_columns = [
        target_column(f"direction_{horizon}"),
        target_column(f"return_{horizon}"),
        target_column(f"volatility_{horizon}"),
        target_column(f"trade_quality_{horizon}"),
        target_column(f"future_regime_{horizon}"),
    ]
    dataset = dataset.dropna(subset=target_columns).reset_index(drop=True)
    if len(dataset) < 2_500:
        raise ValueError(f"Only {len(dataset)} complete labeled rows remain; at least 2,500 are required.")

    train_end = int(len(dataset) * 0.70)
    validation_end = int(len(dataset) * 0.85)
    (
        final_preprocessor,
        direction_model,
        return_model,
        volatility_model,
        quality_model,
        future_regime_model,
        holdout,
    ) = build_models(dataset, feature_columns, train_end, validation_end, horizon)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": final_preprocessor,
            "direction_model": direction_model,
            "return_model": return_model,
            "volatility_model": volatility_model,
            "trade_quality_model": quality_model,
            "future_regime_model": future_regime_model,
            "feature_columns": feature_columns,
            "symbol": candles[0].symbol,
            "timeframe": timeframe,
            "horizon_bars": horizon,
            "feature_version": feature_config.version,
            "random_seed": args.seed,
        },
        args.model_out,
    )
    manifest = {
        "model_bundle": args.model_out.name,
        "symbol": candles[0].symbol,
        "timeframe": timeframe,
        "horizon_bars": horizon,
        "source_data": data_path.name,
        "trained_through": candles[-horizon - 1].end_time.isoformat(),
        "forecast_origin": candles[-1].end_time.isoformat(),
        "rows": len(candles),
        "labeled_rows": len(dataset),
        "feature_count": len(feature_columns),
        "volatility_target": "annualized using actual candle timeframe",
        "holdout_metrics": holdout,
        "model_status": "research_candidate",
        "feed_type": "real_fx_proxy_not_broker_otc",
    }
    manifest_path = args.model_out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    latest_x = final_preprocessor.transform(feature_frame.iloc[[-1]])
    dir_probs = direction_model.predict_proba(latest_x)[0]
    direction_classes = list(direction_model._estimator.classes_)
    p_up = float(dir_probs[direction_classes.index(1)]) if 1 in direction_classes else 0.0
    pred_return = float(return_model.predict(latest_x)[0])
    pred_vol = float(volatility_model.predict(latest_x)[0])
    pred_quality_probs = quality_model.predict_proba(latest_x)[0]
    quality_classes = list(quality_model._estimator.classes_)
    p_quality = (
        float(pred_quality_probs[quality_classes.index(1)]) if 1 in quality_classes else 0.0
    )
    regime_probs = future_regime_model.predict_proba(latest_x)[0]
    regime_classes = list(future_regime_model.classes_)
    predicted_regime = MarketRegime(str(regime_classes[int(np.argmax(regime_probs))]))
    regime_confidence = float(np.max(regime_probs))

    current_vector = vectors[-1]
    current_regime = regime_engine.classify(current_vector)
    future_regime_assessment = RegimeAssessment(
        symbol=current_vector.symbol,
        timestamp=current_vector.timestamp + timedelta(seconds=seconds * horizon),
        timeframe=timeframe,
        regime=predicted_regime,
        confidence=regime_confidence,
        supporting_features={"horizon_bars": horizon, "model": "future_regime_rf"},
        active_subregimes=[predicted_regime],
    )
    technical_assessment = TechnicalRuleEngine().evaluate(current_vector)
    signal = SignalEngine().generate(
        IntelligenceSnapshot(
            symbol=current_vector.symbol,
            timestamp=current_vector.timestamp,
            technical_assessment=technical_assessment,
            regime_assessment=future_regime_assessment,
            direction_prob_up=p_up,
            expected_return=pred_return,
            expected_volatility=pred_vol,
            trade_quality_prob=p_quality,
            model_version="rf-ridge-h30",
            feature_version=feature_config.version,
            metadata={"forecast_horizon_bars": horizon},
        )
    )

    # Monte Carlo path distribution: the ML return and volatility forecasts parameterize
    # a simple lognormal path. The model is not re-queried on synthetic future features.
    latest_close = candles[-1].close
    annualized_vol = float(np.clip(pred_vol, 0.0, 5.0))
    step_sigma = annualized_vol / np.sqrt(bars_per_year)
    step_mu = np.log1p(np.clip(pred_return, -0.95, 2.0)) / horizon - 0.5 * step_sigma**2
    rng = np.random.default_rng(args.seed)
    increments = rng.normal(step_mu, step_sigma, size=(args.paths, horizon))
    simulated_prices = latest_close * np.exp(np.cumsum(increments, axis=1))
    lower, median, upper = np.quantile(simulated_prices, [0.10, 0.50, 0.90], axis=0)
    up_probability_by_step = np.mean(simulated_prices > latest_close, axis=0)

    print("AI-ASSISTED 30-CANDLE FORECAST SIMULATION — RESEARCH ONLY")
    print(f"Asset/timeframe: {candles[0].symbol} / {timeframe}")
    print(
        f"Source: {data_path.name} | training snapshot ends {candles[-1].end_time.isoformat()} "
        f"| {len(candles):,} historical candles"
    )
    print(
        f"Chronological holdout ({holdout['test_rows']:,} samples; 70/15/15 split, "
        f"30-bar purge): direction accuracy={holdout['direction_accuracy']:.3f}, "
        f"vs majority={holdout['direction_majority_baseline']:.3f}, "
        f"direction Brier={holdout['direction_brier']:.3f}, "
        f"return MAE={holdout['return_mae']:.3%} "
        f"(mean baseline {holdout['return_mean_baseline_mae']:.3%}), "
        f"annualized-volatility MAE={holdout['volatility_mae']:.3f}, "
        f"trade-quality accuracy={holdout['quality_accuracy']:.3f} "
        f"(majority {holdout['quality_majority_baseline']:.3f}), "
        f"quality Brier={holdout['quality_brier']:.3f}, "
        f"future-regime accuracy={holdout['future_regime_accuracy']:.3f} "
        f"(majority {holdout['future_regime_majority_baseline']:.3f})"
    )
    print(f"Current close: {latest_close:.4f} at {candles[-1].end_time.isoformat()}")
    print(f"Saved fitted model bundle: {args.model_out.resolve()}")
    print(f"Saved training manifest: {manifest_path.resolve()}")
    print(
        f"30-bar ML outputs: P(up)={p_up:.1%}, expected return={pred_return:+.2%}, "
        f"annualized volatility={pred_vol:.2%}, long-side quality P(target before stop)={p_quality:.1%}"
    )
    print(
        f"Regime: current={current_regime.regime.value} ({current_regime.confidence:.0%}); "
        f"predicted at +30 bars={predicted_regime.value} ({regime_confidence:.0%})"
    )
    print(
        f"Signal fusion: {signal.action.value} | score={signal.fusion_score:+.3f} "
        f"| conviction={signal.conviction:.1%} | {len(signal.evidence)} evidence streams"
    )
    print("\nProjected closes from Monte Carlo paths (median with 10th–90th percentile range):")
    print(" step | forecast time             | median close | p10–p90 range       | paths above start")
    for step in range(horizon):
        forecast_time = candles[-1].end_time + timedelta(seconds=seconds * (step + 1))
        print(
            f" {step + 1:>4} | {forecast_time.isoformat():25.25s} | {median[step]:>12.4f} | "
            f"{lower[step]:.4f}–{upper[step]:.4f} | {up_probability_by_step[step]:>7.1%}"
        )
    print(
        "\nThese rows are a stochastic range simulation from horizon-level AI estimates, "
        "not 30 separately verified candle predictions or a trade instruction."
    )


if __name__ == "__main__":
    main()
