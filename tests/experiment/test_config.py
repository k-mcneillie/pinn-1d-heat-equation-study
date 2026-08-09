from pathlib import Path

import pytest

from pinn_study.experiment.config import MLflowConfig


def test_mlflow_config_defaults() -> None:
    config = MLflowConfig()

    assert config.tracking_uri == "file:./mlruns"
    assert config.experiment_name == "default"
    assert config.run_name is None
    assert config.tags == {}


def test_mlflow_config_custom_values() -> None:
    config = MLflowConfig(
        tracking_uri="file:/tmp/mlruns",
        experiment_name="test-experiment",
        run_name="test-run",
        artifact_location="file:/tmp/artifacts",
        tags={"team": "research"},
    )

    assert config.tracking_uri == "file:/tmp/mlruns"
    assert config.experiment_name == "test-experiment"
    assert config.run_name == "test-run"
    assert config.artifact_location == "file:/tmp/artifacts"
    assert config.tags["team"] == "research"


def test_mlflow_config_require_non_empty_experiment_name() -> None:
    with pytest.raises(ValueError, match="experiment_name must not be empty"):
        MLflowConfig(experiment_name="")


def test_mlflow_config_require_non_empty_tracking_uri() -> None:
    with pytest.raises(ValueError, match="tracking_uri must not be empty"):
        MLflowConfig(tracking_uri="")
