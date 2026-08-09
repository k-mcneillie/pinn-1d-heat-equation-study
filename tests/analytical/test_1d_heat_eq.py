"""Tests for the analytical one-dimensional heat equation solution."""

import pytest
import torch

from pinn_study.analytical.contract import AnalyticalSolution
from pinn_study.analytical.heat_equation_1D import HeatEquation1D


class TestHeatEquation1D:
    """Tests for HeatEquation1D."""

    def test_implements_analytical_contract(self) -> None:
        """HeatEquation1D satisfies the analytical solution contract."""
        solution: AnalyticalSolution = HeatEquation1D()

        x = torch.tensor([0.0, 0.5, 1.0])
        t = torch.tensor([0.0, 0.25, 0.5])

        result = solution(x, t, alpha=1.0)

        assert result.shape == x.shape

    def test_initial_condition(self) -> None:
        """The solution matches the initial condition at t = 0."""
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        t = torch.zeros_like(x)

        result = HeatEquation1D()(x, t, alpha=1.0)
        expected = torch.sin(torch.pi * x)

        assert torch.allclose(result, expected)

    def test_boundary_conditions(self) -> None:
        """The solution is zero at both spatial boundaries."""
        x = torch.tensor([0.0, 1.0])
        t = torch.tensor([0.25, 0.75])

        result = HeatEquation1D()(x, t, alpha=1.0)

        assert torch.allclose(result, torch.zeros_like(result))

    def test_solution_decays_over_time(self) -> None:
        """The solution amplitude decreases as time increases."""
        x = torch.tensor([0.5])

        early = HeatEquation1D()(x, torch.tensor([0.1]), alpha=1.0)
        late = HeatEquation1D()(x, torch.tensor([0.5]), alpha=1.0)

        assert torch.abs(late) < torch.abs(early)

    @pytest.mark.parametrize("alpha", [0.0, -1.0])
    def test_rejects_non_positive_alpha(self, alpha: float) -> None:
        """A non-positive thermal diffusivity raises ValueError."""
        x = torch.tensor([0.5])
        t = torch.tensor([0.5])

        with pytest.raises(ValueError):
            HeatEquation1D()(x, t, alpha=alpha)