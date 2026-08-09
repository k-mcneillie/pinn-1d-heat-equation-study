"""Tests for PINN XAI visualisation functions."""

from pathlib import Path

import torch

from pinn_study.vis.xai.residual_attribution import (
    compute_residual_attribution,
    compute_residual_sensitivity,
    plot_residual_attribution,
    plot_residual_sensitivity,
)
from pinn_study.vis.xai.sensitivity import (
    compute_solution_sensitivity,
    plot_sensitivity_vs_error,
    plot_solution_sensitivity,
)


class TestSolutionSensitivity:
    """Tests for solution sensitivity analysis."""

    def test_compute_solution_sensitivity(self) -> None:
        """Solution sensitivity is returned as a tensor."""
        x = torch.linspace(
            0.0,
            1.0,
            20,
            requires_grad=True,
        )

        prediction = x**2

        sensitivity = compute_solution_sensitivity(
            x,
            prediction,
        )

        assert isinstance(sensitivity, torch.Tensor)
        assert sensitivity.shape == prediction.shape

    def test_solution_sensitivity_plot(
        self,
        tmp_path: Path,
    ) -> None:
        """Solution sensitivity plot is saved."""
        x = torch.linspace(0.0, 1.0, 20)
        sensitivity = 2.0 * x

        output = tmp_path / "sensitivity.png"

        result = plot_solution_sensitivity(
            x,
            sensitivity,
            output,
        )

        assert result == output
        assert output.is_file()

    def test_sensitivity_vs_error_plot(
        self,
        tmp_path: Path,
    ) -> None:
        """Sensitivity/error comparison is saved."""
        sensitivity = torch.linspace(0.0, 1.0, 20)
        error = sensitivity * 0.1

        output = tmp_path / "sensitivity_error.png"

        result = plot_sensitivity_vs_error(
            sensitivity,
            error,
            output,
        )

        assert result == output
        assert output.is_file()


class TestResidualAttribution:
    """Tests for residual attribution."""

    def test_compute_residual_sensitivity(self) -> None:
        """Residual sensitivity is returned as a tensor."""
        x = torch.linspace(0.0, 1.0, 20)
        residual = x**2

        sensitivity = compute_residual_sensitivity(
            x,
            residual,
        )

        assert isinstance(sensitivity, torch.Tensor)
        assert sensitivity.shape == residual.shape

    def test_compute_residual_attribution(self) -> None:
        """Residual attribution is returned as a tensor."""
        sensitivity = torch.tensor(
            [0.0, 1.0, 2.0, 3.0],
        )

        attribution = compute_residual_attribution(
            sensitivity,
        )

        assert isinstance(attribution, torch.Tensor)
        assert attribution.shape == sensitivity.shape

    def test_residual_sensitivity_plot(
        self,
        tmp_path: Path,
    ) -> None:
        """Residual sensitivity plot is saved."""
        x = torch.linspace(0.0, 1.0, 20)
        sensitivity = 2.0 * x

        output = tmp_path / "residual_sensitivity.png"

        result = plot_residual_sensitivity(
            x,
            sensitivity,
            output,
        )

        assert result == output
        assert output.is_file()

    def test_residual_attribution_plot(
        self,
        tmp_path: Path,
    ) -> None:
        """Residual attribution plot is saved."""
        x = torch.linspace(0.0, 1.0, 20)
        attribution = torch.linspace(0.0, 1.0, 20)

        output = tmp_path / "residual_attribution.png"

        result = plot_residual_attribution(
            x,
            attribution,
            output,
        )

        assert result == output
        assert output.is_file()