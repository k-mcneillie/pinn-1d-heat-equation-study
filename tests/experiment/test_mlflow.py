from pathlib import Path

import mlflow

from pinn_study.experiment.config import MLflowConfig
from pinn_study.experiment.mlflow import MLflowExperiment


def test_mlflow_run_lifecycle(tmp_path: Path) -> None:
    tracking_dir = tmp_path / "mlruns"
    config = MLflowConfig(
        tracking_uri=f"file:{tracking_dir}",
        experiment_name="test_experiment",
        run_name="test_run",
    )

    experiment = MLflowExperiment(config)
    result = experiment.start_run()

    assert result.experiment_name == config.experiment_name
    assert result.run_id
    assert result.tracking_uri == config.tracking_uri
    assert result.status == "RUNNING"

    experiment.log_params({"learning_rate": 0.01, "batch_size": 32})
    experiment.log_metrics({"loss": 0.1}, step=1)

    artifact_file = tmp_path / "artifact.txt"
    artifact_file.write_text("hello")
    experiment.log_artifact(artifact_file)

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "nested.txt").write_text("world")
    experiment.log_artifacts(artifact_dir, artifact_path="nested")

    experiment.end_run()

    client = mlflow.tracking.MlflowClient(tracking_uri=config.tracking_uri)
    runs = client.search_runs([config.experiment_name], order_by=["attributes.start_time DESC"])

    assert runs
    run = runs[0]
    assert run.data.params["learning_rate"] == "0.01"
    assert run.data.params["batch_size"] == "32"
    assert run.data.metrics["loss"] == 0.1

    artifacts = client.list_artifacts(run.info.run_id)
    artifact_paths = {entry.path for entry in artifacts}
    assert "artifact.txt" in artifact_paths or "artifact.txt" in {p.name for p in artifacts}


def test_mlflow_git_metadata_degrades_gracefully(tmp_path: Path) -> None:
    config = MLflowConfig(
        tracking_uri=f"file:{tmp_path / 'mlruns'}",
        experiment_name="test_experiment_git",
    )

    experiment = MLflowExperiment(config)
    git_metadata = experiment.read_git_metadata(cwd=tmp_path)

    assert isinstance(git_metadata, dict)
    assert "git.commit" not in git_metadata or isinstance(git_metadata["git.commit"], str)
    assert "git.branch" not in git_metadata or isinstance(git_metadata["git.branch"], str)
    assert "git.dirty" not in git_metadata or git_metadata["git.dirty"] in {"true", "false"}
