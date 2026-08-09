# contract.py
# - contract for generalised weighted loss

# =============================
# Import Libraries
# =============================
from typing import Protocol

from torch import Tensor


class Loss(Protocol):
    """Protocol for combining PINN constraint losses."""

    def __call__(
        self,
        losses: dict[str, Tensor],
    ) -> Tensor:
        """
        Calculate the total loss from constraint losses.

        Args:
            losses: Mapping of constraint names to scalar loss tensors.

        Returns:
            The total weighted loss.
        """
        ...
