"""Tests for the analytical heat equation solution."""

import pytest
import torch

from pinn_study.analytical.heat_equation_1D import HeatEquation1D

analytical_solution = HeatEquation1D()


def test_initial_condition() -> None:
    """The solution matches the initial condition at t = 0."""
    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    t = torch.zeros_like(x)
    result = analytical_solution(x, t)
    expected = torch.sin(torch.pi * x)
    assert torch.allclose(result, expected)


def test_boundary_conditions() -> None:
    """The solution is zero at both spatial boundaries."""
    x = torch.tensor([0.0, 1.0])
    t = torch.tensor([0.25, 0.75])
    result = analytical_solution(x, t)
    assert torch.allclose(result, torch.zeros_like(result))


def test_solution_decays_over_time() -> None:
    """The solution amplitude decreases as time increases."""
    x = torch.tensor([0.5])
    early = analytical_solution(x, torch.tensor([0.1]))
    late = analytical_solution(x, torch.tensor([0.5]))
    assert torch.abs(late) < torch.abs(early)


def test_invalid_alpha() -> None:
    """A non-positive thermal diffusivity raises ValueError."""
    x = torch.tensor([0.5])
    t = torch.tensor([0.5])
    with pytest.raises(ValueError):
        analytical_solution(x, t, alpha=0.0)
    with pytest.raises(ValueError):
        analytical_solution(x, t, alpha=-1.0)
