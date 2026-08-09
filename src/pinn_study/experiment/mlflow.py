from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow

from .config import MLflowConfig
from .result import MLflowResult


class MLflowExperiment:
    """Lightweight MLflow wrapper for experiment recording."""

    def __init__(self, config: MLflowConfig) -> None:
        self.config = config
        self._allow_file_store_if_needed()
        mlflow.set_tracking_uri(self.config.tracking_uri)
        self._set_experiment()
        self._active_run = None

    def _allow_file_store_if_needed(self) -> None:
        if self.config.tracking_uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    def _set_experiment(self) -> None:
        if self.config.artifact_location:
            mlflow.set_experiment(
                experiment_name=self.config.experiment_name,
                artifact_location=self.config.artifact_location,
            )
        else:
            mlflow.set_experiment(
                experiment_name=self.config.experiment_name,
            )

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> MLflowResult:
        if self._active_run is not None:
            raise RuntimeError("An MLflow run is already active.")

        effective_tags = {**self.config.tags, **(tags or {})}
        effective_tags.update(self.read_git_metadata())

        self._active_run = mlflow.start_run(
            run_name=run_name or self.config.run_name,
            tags=effective_tags,
        )

        return self._collect_result()

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active_run is None:
            return

        mlflow.end_run(status=status)
        self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        self._require_active_run()
        if not params:
            return

        safe_params = {
            key: value if isinstance(value, (str, int, float, bool)) else str(value)
            for key, value in params.items()
        }

        mlflow.log_params(safe_params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        self._require_active_run()
        if not metrics:
            return

        if step is None:
            mlflow.log_metrics(metrics)
        else:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self,
        path: Path,
        artifact_path: str | None = None,
    ) -> None:
        self._require_active_run()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {path}")

        if path.is_dir():
            self.log_artifacts(path, artifact_path=artifact_path)
            return

        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(
        self,
        path: Path,
        artifact_path: str | None = None,
    ) -> None:
        self._require_active_run()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Artifacts path does not exist: {path}")

        mlflow.log_artifacts(str(path), artifact_path=artifact_path)

    def read_git_metadata(
        self,
        cwd: Path | None = None,
    ) -> dict[str, str]:
        cwd = cwd or Path.cwd()
        metadata: dict[str, str] = {}

        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status_output = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            metadata["git.commit"] = commit
            metadata["git.branch"] = branch
            metadata["git.dirty"] = str(bool(status_output)).lower()
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            metadata = {}

        return metadata

    def _collect_result(self) -> MLflowResult:
        if self._active_run is None:
            raise RuntimeError("No active MLflow run is available.")

        info = self._active_run.info

        return MLflowResult(
            experiment_name=self.config.experiment_name,
            run_id=info.run_id,
            tracking_uri=self.config.tracking_uri,
            artifact_uri=info.artifact_uri,
            status=info.status,
        )

    def _require_active_run(self) -> None:
        if self._active_run is None:
            raise RuntimeError("An MLflow run must be started before logging.")
