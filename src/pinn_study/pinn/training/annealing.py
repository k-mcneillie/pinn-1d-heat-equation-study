# annealing.py
# - loss-weight annealing strategies

# =============================
# Import Libraries
# =============================
from dataclasses import dataclass


# =============================
# Constant Weighting
# =============================
@dataclass
class ConstantWeighting:
    """Keep loss weights constant throughout training."""

    weights: dict[str, float]

    def update(self, epoch: int) -> dict[str, float]:
        """Return unchanged weights.

        Args:
            epoch: Current epoch.

        Returns:
            Configured loss weights.
        """
        return self.weights.copy()


# =============================
# Linear Weighting
# =============================
@dataclass
class LinearWeighting:
    """Linearly transition between initial and final weights."""

    initial: dict[str, float]
    final: dict[str, float]
    epochs: int

    def update(self, epoch: int) -> dict[str, float]:
        """Calculate weights for the current epoch.

        Args:
            epoch: Current epoch.

        Returns:
            Interpolated loss weights.
        """
        progress = min(epoch / self.epochs, 1.0)

        return {
            name: self.initial[name]
            + progress * (self.final[name] - self.initial[name])
            for name in self.initial
        }
