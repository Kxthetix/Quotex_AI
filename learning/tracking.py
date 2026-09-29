"""Experiment tracking and MLflow integration for Phase 4.

Records model training runs, hyperparameters, feature/label versions, dataset IDs,
chronological periods, and evaluation metrics for scientific reproducibility.
"""

import contextlib
import os
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import mlflow

from configs.config import get_settings
from models.base import BaselineModel
from monitoring.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Manages MLflow experiment tracking with graceful local fallback."""

    def __init__(
        self,
        experiment_name: str = "trading_intelligence_phase4",
        tracking_uri: str | None = None,
    ) -> None:
        self.settings = get_settings()
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or self.settings.mlflow_tracking_uri

        # Enable file store backend for local execution in MLflow 3+
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        self._init_mlflow()

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking URI and experiment."""
        uri_to_use = self.tracking_uri
        if uri_to_use.startswith("http://") or uri_to_use.startswith("https://"):
            import socket
            import urllib.parse

            parsed = urllib.parse.urlparse(uri_to_use)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            is_reachable = False
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.3)
                try:
                    sock.connect((host, port))
                    is_reachable = True
                except Exception:
                    is_reachable = False

            if not is_reachable:
                local_fallback = f"file:///{Path('data/mlruns').absolute().as_posix()}"
                logger.info(
                    "Remote MLflow server '%s' not reachable; using local store '%s'.",
                    uri_to_use,
                    local_fallback,
                )
                uri_to_use = local_fallback

        try:
            mlflow.set_tracking_uri(uri_to_use)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            # Fallback to local directory tracking if URI fails
            local_fallback = f"file:///{Path('data/mlruns').absolute().as_posix()}"
            logger.warning(
                "Unable to initialize MLflow URI '%s' (%s). Falling back to '%s'.",
                uri_to_use,
                e,
                local_fallback,
            )
            mlflow.set_tracking_uri(local_fallback)
            mlflow.set_experiment(self.experiment_name)

    @contextlib.contextmanager
    def track_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Context manager for an MLflow run."""
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            yield run

    def log_model_run(
        self,
        model: BaselineModel,
        dataset_id: str,
        metrics: dict[str, float],
        training_period: tuple[datetime, datetime] | tuple[str, str] = ("", ""),
        val_period: tuple[datetime, datetime] | tuple[str, str] = ("", ""),
        test_period: tuple[datetime, datetime] | tuple[str, str] = ("", ""),
        artifact_path: Path | str | None = None,
    ) -> str:
        """Log a comprehensive model training run to MLflow."""
        run_name = f"{model.model_name}_{model.model_version}"
        train_start = (
            training_period[0].isoformat()
            if isinstance(training_period[0], datetime)
            else str(training_period[0])
        )
        train_end = (
            training_period[1].isoformat()
            if isinstance(training_period[1], datetime)
            else str(training_period[1])
        )

        tags = {
            "model_name": model.model_name,
            "model_version": model.model_version,
            "algorithm": model.algorithm,
            "feature_version": model.feature_version,
            "label_version": model.label_version,
            "dataset_id": dataset_id,
            "training_start": train_start,
            "training_end": train_end,
            "status": model.status.value,
        }

        with self.track_run(run_name=run_name, tags=tags) as run:
            run_id = run.info.run_id

            # Log parameters
            params_to_log = dict(model.parameters)
            params_to_log["algorithm"] = model.algorithm
            params_to_log["feature_version"] = model.feature_version
            params_to_log["label_version"] = model.label_version
            params_to_log["dataset_id"] = dataset_id
            params_to_log["num_features"] = len(model.expected_features)
            mlflow.log_params(params_to_log)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log artifact if provided
            if artifact_path and Path(artifact_path).exists():
                mlflow.log_artifact(str(artifact_path))

            logger.info("Logged MLflow run %s for model '%s'", run_id, run_name)
            return run_id
