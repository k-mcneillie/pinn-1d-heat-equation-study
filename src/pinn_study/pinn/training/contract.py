# contract.py
# - contracts for the model training subsystem

# =============================
# Import Libraries
# =============================
from typing import Protocol

from torch import Tensor


# =============================
# Training Step Contract
# =============================
# TrainingStep represents the problem-specific calculation required to
# produce one scalar loss for the current model parameters.
class TrainingStep(Protocol):
    """Contract for a single model training step."""

    def __call__(self) -> Tensor:
        """Calculate and return the current training loss."""
        ...


# =============================
# Loss Weighting Contract
# =============================
# LossWeightingStrategy controls how constraint weights evolve during
# training. The PINNLoss itself remains responsible only for aggregation.
class LossWeightingStrategy(Protocol):
    """Contract for dynamic loss weighting."""

    def update(self, epoch: int) -> dict[str, float]:
        """Return the loss weights for the current epoch."""
        ...


# =============================
# Checkpoint Contract
# =============================
class CheckpointManager(Protocol):
    """Contract for checkpoint persistence."""

    def save(
        self,
        epoch: int,
        model_state: dict,
        optimizer_state: dict,
        loss: float,
        history: list[float],
        scheduler_state: dict | None = None,
    ) -> None:
        """Save training state."""
        ...
