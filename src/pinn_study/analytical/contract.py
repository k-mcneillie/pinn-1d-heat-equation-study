# contract.py
# - define protocols for analytical solutions.

# =============================
# Import Libraries
# =============================
from typing import Protocol

import torch


class AnalyticalSolution(Protocol):
    """Protocol for analytical solutions."""

    def __init__(self) -> None:
        """Initialise the analytical solution."""
        ...

    def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluate the analytical solution.

        Args:
            coordinates: Tensor containing the input coordinates.

        Returns:
            Tensor containing the analytical solution evaluated at the
            supplied coordinates.
        """
        ...
