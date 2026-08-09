# heat_equation_1D.py
# - define the one-dimensional heat equation analytical solution.

# =============================
# Import Libraries
# =============================
from __future__ import annotations

import torch

# =============================
# Analytical Solution for 1D Heat Equation
# =============================

# The one-dimensional heat equation is a partial differential equation that describes
# the distribution of heat (or variation in temperature) in a given region over time.
# The general form of the 1D heat equation is given by:
#     ∂u/∂t = α * ∂²u/∂x²
# where:
# - u(x, t) is the temperature distribution function,
# - α is the thermal diffusivity constant,
# - x is the spatial coordinate, and
# - t is the time.


# The analytical solution to the 1D heat equation can be derived for specific initial
# and boundary conditions with the Fourier series solution given by:
#     u(x, t) = exp(-alpha * pi^2 * t) * sin(pi * x)
# In this implementation, we consider Fourier series solution for the initial condition
# given by:
#       u(x, 0) = sin(pi * x)
# and homogeneous Dirichlet boundary conditions on x = 0 and x = 1.
class HeatEquation1D:
    """Analytical solution for the one-dimensional heat equation."""

    def __init__(self):
        """Initialise the analytical solution."""
        ...

    def __call__(
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
