"""Tests for the training visualisation API."""

from pathlib import Path

from pinn_study.pinn.training.result import TrainingResult
from pinn_study.vis.api.training import visualise_training
from pinn_study.vis.result import VisualizationResult


def test_visualise_training_returns_result(tmp_path: Path) -> None:
    """Training visualisation returns a VisualizationResult."""
    result = visualise_training(
        result=TrainingResult(
            epochs=[1, 2, 3],
            losses=[1.0, 0.5, 0.25],
            learning_rates=[1e-3, 1e-3, 1e-4],
            loss_components={
                "pde": [0.8, 0.4, 0.2],
            },
        ),
        output_dir=tmp_path,
    )

    assert isinstance(result, VisualizationResult)
    assert result.output_dir == tmp_path
    assert result.figures