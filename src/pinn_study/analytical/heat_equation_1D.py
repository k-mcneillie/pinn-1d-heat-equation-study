"""Analytical solution for the one-dimensional heat equation."""

from __future__ import annotations

import torch


class HeatEquation1D:
    def __init__(self):
        pass

    def compute(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Evaluate the analytical heat equation solution.

        The solution is
            u(x, t) = exp(-alpha * pi^2 * t) * sin(pi * x)
        for the initial condition u(x, 0) = sin(pi * x) and homogeneous
        Dirichlet boundary conditions on x = 0 and x = 1.

        Args:
            x: Spatial coordinates.
            t: Temporal coordinates.
            alpha: Thermal diffusivity.

        Returns:
            Analytical solution evaluated at (x, t).

        Raises:
            ValueError: If alpha is not positive.
        """
        if alpha <= 0:
            raise ValueError("alpha must be greater than zero.")

        return torch.exp(-alpha * torch.pi**2 * t) * torch.sin(torch.pi * x)
