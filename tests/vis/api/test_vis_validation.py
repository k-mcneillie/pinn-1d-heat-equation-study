"""Tests for the validation visualisation API."""

from pathlib import Path

from pinn_study.vis.api.validation import visualise_validation
from pinn_study.vis.result import VisualizationResult


def test_visualise_validation_returns_result(tmp_path: Path) -> None:
    """Validation visualisation returns a VisualizationResult."""
    result = visualise_validation(
        validation_data=None,
        output_dir=tmp_path,
    )

    assert isinstance(result, VisualizationResult)
    assert result.output_dir == tmp_path