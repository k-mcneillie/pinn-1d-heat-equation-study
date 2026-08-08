# generator.py
# - generate data for analytical methods and model train, test and validate.

# =============================
# Import Libraries
# =============================
from __future__ import annotations

from dataclasses import dataclass

import torch


# A data class uses the @dataclass decorator to automatically generate special
# methods to handle creation, printing and comparisons.
# By default, dataclasses are mutable meaning attributes can be altered at any
# time. Adding frozen=True modifies this characteristic to make the instance
# immutable, blocking modification of the objects attributes raising a
# FrozenInstanceError error.
@dataclass(frozen=True)
class HeatEquationData:
    """
    Container for data.

    Attributes:
        interior: Interior collocation points with shape = (N, 2),
        initial: Initial-condition points with shape = (N, 2).
        boundary: Boundary-condition points with shape = (N, 2).
    """

    interior: torch.Tensor
    initial: torch.Tensor
    boundary: torch.Tensor


class Generator:
    """
    Generate collocation points for a 1D heat equation study.

    Args:
        n_interior: Number of interior collocation points.
        n_initial: Number of initial-condition points.
        n_boundary: Number of boundary-condition points.
        x_min: Lower spatial boundary.
        x_max: Upper spatial boundary.
        t_min: Initial time.
        t_max: Final time.
        seed: Seed for reproducibility.
        device: cpu, cuda or mps

    Raises:
        ValueError: If any number of points is non-positive, the spatial
            domain is invalid, or the temporal domain is invalid.
    """

    def __init__(
        self,
        n_interior: int,
        n_initial: int,
        n_boundary: int,
        *,  # All arguments after this must be kwargs.
        x_min: float = 0.0,
        x_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ) -> None:
        # Check argument magnitudes are valid.
        if n_interior <= 0:
            raise ValueError(
                f"[ERROR] n_interior = {n_interior} must be greater than zero."
            )

        if n_initial <= 0:
            raise ValueError(
                f"[ERROR] n_initial = {n_initial} must be greater than zero."
            )

        if n_boundary <= 0:
            raise ValueError(
                f"[ERROR] n_boundary = {n_boundary} must be greater than zero."
            )

        if t_min < 0:
            raise ValueError(f"[ERROR] t_min = {t_min} must be greater than zero.")

        if t_min >= t_max:
            raise ValueError(
                f"[ERROR] t_min = {t_min} must be smaller than t_max = {t_max}."
            )

        if x_min >= x_max:
            raise ValueError(
                f"[ERROR] x_min = {x_max} must be smaller than x_max = {x_max}."
            )

        # Set class attributes
        self.n_interior = n_interior
        self.n_initial = n_initial
        self.n_boundary = n_boundary

        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.t_max = t_max

        self.rng = torch.Generator().manual_seed(seed)
        self.device = device

    # Main generation method
    def generate(self) -> HeatEquationData:
        """
        Generate points.

        Returns:
            A HeatEquationData object containing interior, initial and boundary points.
        """
        return HeatEquationData(
            interior=self._generate_interior_points(),
            initial=self._generate_initial_points(),
            boundary=self._generate_boundary_points(),
        )

    def _generate_interior_points(self) -> torch.Tensor:
        """
        Generate uniformly distributed interior points.

        Returns:
            Tensor of (x, t) points with shape (n_interior, 2).
        """
        x = self._uniform(
            self.n_interior,
            self.x_min,
            self.x_max,
        )
        t = self._uniform(
            self.n_interior,
            self.t_min,
            self.t_max,
        )

        return torch.column_stack((x, t))

    def _generate_initial_points(self) -> torch.Tensor:
        """Generate points on the initial-time boundary.

        Returns:
            Tensor of (x, t) points with shape (n_initial, 2).
        """
        x = self._uniform(
            self.n_initial,
            self.x_min,
            self.x_max,
        )
        t = torch.full_like(x, self.t_min)

        return torch.column_stack((x, t))

    def _generate_boundary_points(self) -> torch.Tensor:
        """
        Generate points on the two spatial boundaries.

        Points are split approximately equally between x_min and
        x_max.

        Returns:
            Tensor of (x, t) points with shape (n_boundary, 2).
        """
        n_left = self.n_boundary // 2
        n_right = self.n_boundary - n_left

        t_left = self._uniform(
            n_left,
            self.t_min,
            self.t_max,
        )
        t_right = self._uniform(
            n_right,
            self.t_min,
            self.t_max,
        )

        left = torch.column_stack(
            (
                torch.full_like(t_left, self.x_min),
                t_left,
            )
        )

        right = torch.column_stack(
            (
                torch.full_like(t_right, self.x_max),
                t_right,
            )
        )

        return torch.cat((left, right))

    def _uniform(
        self,
        n_points: int,
        minimum: float,
        maximum: float,
    ) -> torch.Tensor:
        """
        Generate uniformly distributed values within an interval.

        Args:
            n_points: Number of values to generate.
            minimum: Lower bound of the interval.
            maximum: Upper bound of the interval.

        Returns:
            One-dimensional tensor containing uniformly distributed values.
        """
        return minimum + (maximum - minimum) * torch.rand(
            n_points, generator=self.rng, device=self.device
        )
