# pinn_loss.py
# - PINN loss implementation

# =============================
# Import Libraries
# =============================
from torch import Tensor

from .config import LossConfig


# =============================
# PINN Loss
# =============================
class PINNLoss:
    """Combine weighted constraint losses."""

    def __init__(self, config: LossConfig) -> None:
        """
        Initialise the PINN loss.

        Args:
            config: Configuration containing constraint weights.
        """
        self.config = config

    def __call__(self, losses: dict[str, Tensor]) -> Tensor:
        """
        Calculate the weighted total loss.
        Only constraints defined in the configuration contribute to the
        total loss.

        Args:
            losses: Mapping of constraint names to scalar loss tensors.

        Returns:
            The total weighted loss.

        Raises:
            ValueError: If a configured constraint is missing from losses.
        """
        missing = set(self.config.weights) - set(losses)
        if missing:
            raise ValueError(
                f"Missing losses for configured constraints: {sorted(missing)}"
            )
        return sum(
            self.config.weights[name] * losses[name] for name in self.config.weights
        )
