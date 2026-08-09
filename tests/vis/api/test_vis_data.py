"""Tests for the data visualisation API."""

from pathlib import Path

from pinn_study.vis.api.data import visualise_dataset
from pinn_study.vis.result import VisualizationResult


def test_visualise_dataset_returns_result(tmp_path: Path) -> None:
    """Dataset visualisation returns a VisualizationResult."""
    result = visualise_dataset(
        dataset=None,
        output_dir=tmp_path,
    )

    assert isinstance(result, VisualizationResult)
    assert result.output_dir == tmp_path