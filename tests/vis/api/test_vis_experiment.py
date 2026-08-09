"""Tests for the complete experiment visualisation API."""

from pathlib import Path

from pinn_study.pinn.training.result import TrainingResult
from pinn_study.vis.api.experiment import visualise_experiment
from pinn_study.vis.result import VisualizationResult


def test_visualise_experiment_returns_result(tmp_path: Path) -> None:
    """Experiment visualisation returns a VisualizationResult."""
    result = visualise_experiment(
        dataset=None,
        training_result=TrainingResult(
            epochs=[1, 2],
            losses=[1.0, 0.5],
            learning_rates=[1e-3, 1e-3],
            loss_components={},
        ),
        validation_data=None,
        model=None,
        xai_data=None,
        output_dir=tmp_path,
    )

    assert isinstance(result, VisualizationResult)
    assert result.output_dir == tmp_path