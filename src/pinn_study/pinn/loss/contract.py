# contract.py
# - contract for generalised weighted loss

# =============================
# Import Libraries
# =============================
from typing import Protocol

from torch import Tensor

from pinn_study.pinn.loss.result import PINNLossResult


# =============================
# Loss Contract
# =============================
class Loss(Protocol):
    """Protocol for combining PINN constraint losses."""

    def __call__(
        self,
        losses: dict[str, Tensor],
    ) -> PINNLossResult:
        """Calculate the weighted loss and its components.

        Args:
            losses: Mapping of constraint names to scalar loss tensors.

        Returns:
            Weighted total loss and individual constraint losses.
        """
        ...
