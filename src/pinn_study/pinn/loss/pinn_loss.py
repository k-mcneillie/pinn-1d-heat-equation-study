# pinn_loss.py
# - generic PINN loss implementation


# =============================
# Import Libraries
# =============================
from torch import Tensor

from pinn_study.pinn.loss.config import LossConfig
from pinn_study.pinn.loss.result import PINNLossResult


# =============================
# PINN Loss
# =============================
class PINNLoss:
    """Combine weighted constraint losses."""

    def __init__(self, config: LossConfig) -> None:
        """Initialise the PINN loss.

        Args:
            config: Configuration containing constraint weights.
        """
        self.config = config

    # =============================
    # Weight Updates
    # =============================
    def update_weights(self, weights: dict[str, float]) -> None:
        """Update constraint weights.

        Args:
            weights: New constraint weights.

        Raises:
            ValueError: If a weight is negative or a constraint is unknown.
        """
        if set(weights) != set(self.config.weights):
            raise ValueError(
                "Updated weights must contain the same constraints "
                "as the configured weights."
            )

        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError("Loss weights must be non-negative.")

        self.config.weights = weights.copy()

    # =============================
    # Loss Calculation
    # =============================
    def __call__(
        self,
        losses: dict[str, Tensor],
    ) -> PINNLossResult:
        """Calculate the weighted total loss.

        Args:
            losses: Mapping of constraint names to scalar losses.

        Returns:
            Weighted total loss and individual constraint losses.

        Raises:
            ValueError: If a configured constraint is missing.
        """
        missing = set(self.config.weights) - set(losses)

        if missing:
            raise ValueError(
                f"Missing losses for configured constraints: {sorted(missing)}"
            )

        total = sum(
            self.config.weights[name] * losses[name] for name in self.config.weights
        )

        return PINNLossResult(
            total=total,
            components=losses.copy(),
        )
