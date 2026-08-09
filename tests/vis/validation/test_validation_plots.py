"""Tests for validation visualisation functions."""

from pathlib import Path

import numpy as np
import torch

from pinn_study.vis.validation.convergence import plot_convergence
from pinn_study.vis.validation.error import (
    absolute_error,
    plot_absolute_error,
    plot_error_distribution,
    plot_relative_error,
    relative_error,
)
from pinn_study.vis.validation.metrics import calculate_error_metrics
from pinn_study.vis.validation.residuals import (
    plot_pde_residual,
    plot_residual_distribution,
)
from pinn_study.vis.validation.solution import plot_solution_comparison


class TestErrorCalculations:
    """Tests for error calculations."""

    def test_absolute_error(self) -> None:
        """Absolute error is calculated correctly."""
        prediction = np.array([1.0, 2.0, 4.0])
        analytical = np.array([1.0, 3.0, 2.0])

        result = absolute_error(
            prediction,
            analytical,
        )

        np.testing.assert_allclose(
            result,
            np.array([0.0, 1.0, 2.0]),
        )

    def test_relative_error(self) -> None:
        """Relative error is calculated correctly."""
        prediction = np.array([2.0, 4.0])
        analytical = np.array([1.0, 2.0])

        result = relative_error(
            prediction,
            analytical,
        )

        np.testing.assert_allclose(
            result,
            np.array([1.0, 1.0]),
        )

    def test_relative_error_handles_zero(self) -> None:
        """Relative error remains finite at zero analytical values."""
        result = relative_error(
            np.array([1.0]),
            np.array([0.0]),
        )

        assert np.all(np.isfinite(result))


class TestErrorMetrics:
    """Tests for aggregate error metrics."""

    def test_metrics_are_returned(self) -> None:
        """Error metrics are returned as a dictionary."""
        prediction = torch.tensor([1.0, 2.0, 4.0])
        analytical = torch.tensor([1.0, 3.0, 2.0])

        result = calculate_error_metrics(
            prediction,
            analytical,
        )

        assert isinstance(result, dict)
        assert result
        assert all(isinstance(value, float) for value in result.values())


class TestValidationPlots:
    """Tests for validation figures."""

    def test_absolute_error_plot(self, tmp_path: Path) -> None:
        """Absolute error plot is saved."""
        x = np.linspace(0.0, 1.0, 100)
        analytical = np.sin(x)
        prediction = analytical + 0.01

        output = tmp_path / "absolute.png"

        result = plot_absolute_error(
            x,
            prediction,
            analytical,
            output,
        )

        assert result == output
        assert output.is_file()

    def test_relative_error_plot(self, tmp_path: Path) -> None:
        """Relative error plot is saved."""
        x = np.linspace(0.0, 1.0, 100)
        analytical = np.sin(x) + 0.1
        prediction = analytical + 0.01

        output = tmp_path / "relative.png"

        result = plot_relative_error(
            x,
            prediction,
            analytical,
            output,
        )

        assert result == output
        assert output.is_file()

    def test_error_distribution(self, tmp_path: Path) -> None:
        """Error distribution plot is saved."""
        analytical = np.sin(np.linspace(0.0, 1.0, 100))
        prediction = analytical + 0.01

        output = tmp_path / "distribution.png"

        result = plot_error_distribution(
            prediction,
            analytical,
            output,
        )

        assert result == output
        assert output.is_file()

    def test_solution_comparison(self, tmp_path: Path) -> None:
        """Solution comparison plot is saved."""
        x = np.linspace(0.0, 1.0, 100)
        analytical = np.sin(x)
        prediction = analytical + 0.01

        output = tmp_path / "solution.png"

        result = plot_solution_comparison(
            x,
            prediction,
            analytical,
            output,
        )

        assert result == output
        assert output.is_file()

    def test_pde_residual(self, tmp_path: Path) -> None:
        """PDE residual plot is saved."""
        x = np.linspace(0.0, 1.0, 100)
        residual = np.sin(x)

        output = tmp_path / "residual.png"

        result = plot_pde_residual(
            x,
            residual,
            output,
        )

        assert result == output
        assert output.is_file()

    def test_residual_distribution(self, tmp_path: Path) -> None:
        """Residual distribution plot is saved."""
        residual = np.sin(np.linspace(0.0, 1.0, 100))

        output = tmp_path / "residual_distribution.png"

        result = plot_residual_distribution(
            residual,
            output,
        )

        assert result == output
        assert output.is_file()

    def test_convergence(self, tmp_path: Path) -> None:
        """Convergence plot is generated."""
        output = tmp_path / "convergence.png"

        result = plot_convergence(
            [1, 2, 3],
            [1.0, 0.5, 0.25],
            output,
        )

        assert result is None
        assert output.is_file()