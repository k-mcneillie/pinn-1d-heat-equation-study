"""Tests for data visualisation functions."""

from pathlib import Path

import numpy as np

from pinn_study.vis.data.distributions import (
    plot_distribution_comparison,
    plot_input_distribution,
    plot_target_distribution,
)
from pinn_study.vis.data.domain import (
    plot_domain,
    plot_sampling_points,
    plot_special_points,
)
from pinn_study.vis.data.sampling import (
    plot_sampling_coverage,
    plot_sampling_density,
    plot_sampling_gaps,
)


class TestDistributionPlots:
    """Tests for distribution visualisations."""

    def test_input_distribution(self, tmp_path: Path) -> None:
        """Input distribution is saved."""
        output = tmp_path / "input.png"

        result = plot_input_distribution(
            np.linspace(0.0, 1.0, 100),
            output,
        )

        assert result == output
        assert output.is_file()
        assert output.stat().st_size > 0

    def test_target_distribution(self, tmp_path: Path) -> None:
        """Target distribution is saved."""
        output = tmp_path / "target.png"

        result = plot_target_distribution(
            np.linspace(0.0, 1.0, 100),
            output,
        )

        assert result == output
        assert output.is_file()

    def test_distribution_comparison(self, tmp_path: Path) -> None:
        """Distribution comparison is saved."""
        output = tmp_path / "comparison.png"

        result = plot_distribution_comparison(
            {
                "prediction": np.linspace(0.0, 1.0, 100),
                "analytical": np.linspace(0.0, 1.0, 100),
            },
            output,
        )

        assert result == output
        assert output.is_file()


class TestDomainPlots:
    """Tests for domain and sampling visualisations."""

    def test_domain(self, tmp_path: Path) -> None:
        """Domain plot is saved."""
        output = tmp_path / "domain.png"
        x = np.linspace(0.0, 1.0, 100)

        result = plot_domain(x, output)

        assert result == output
        assert output.is_file()

    def test_sampling_points(self, tmp_path: Path) -> None:
        """Sampling-point plot is saved."""
        output = tmp_path / "points.png"
        x = np.linspace(0.0, 1.0, 100)

        result = plot_sampling_points(x, output)

        assert result == output
        assert output.is_file()

    def test_special_points(self, tmp_path: Path) -> None:
        """Special-point plot is saved."""
        output = tmp_path / "special.png"
        x = np.linspace(0.0, 1.0, 100)

        result = plot_special_points(
            x,
            output,
            boundary=np.array([0.0, 1.0]),
            initial=np.array([0.0]),
        )

        assert result == output
        assert output.is_file()


class TestSamplingPlots:
    """Tests for sampling diagnostics."""

    def test_sampling_density(self, tmp_path: Path) -> None:
        """Sampling density plot is saved."""
        output = tmp_path / "density.png"
        x = np.linspace(0.0, 1.0, 100)

        result = plot_sampling_density(
            x,
            output,
        )

        assert result == output
        assert output.is_file()

    def test_sampling_gaps(self, tmp_path: Path) -> None:
        """Sampling gap plot is saved."""
        output = tmp_path / "gaps.png"
        x = np.linspace(0.0, 1.0, 100)

        result = plot_sampling_gaps(x, output)

        assert result == output
        assert output.is_file()

    def test_sampling_coverage(self, tmp_path: Path) -> None:
        """Sampling coverage plot is saved."""
        output = tmp_path / "coverage.png"
        x = np.linspace(0.0, 1.0, 100)

        result = plot_sampling_coverage(x, output)

        assert result == output
        assert output.is_file()