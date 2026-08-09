"""Tests for the XAI visualisation API."""

from pathlib import Path

from pinn_study.vis.api.xai import visualise_xai
from pinn_study.vis.result import VisualizationResult


def test_visualise_xai_returns_result(tmp_path: Path) -> None:
    """XAI visualisation returns a VisualizationResult."""
    result = visualise_xai(
        model=None,
        xai_data=None,
        output_dir=tmp_path,
    )

    assert isinstance(result, VisualizationResult)
    assert result.output_dir == tmp_path