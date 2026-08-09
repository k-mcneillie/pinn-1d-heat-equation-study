# test_contract.py
# - tests for training contracts


# =============================
# Import Libraries
# =============================
import torch

from pinn_study.pinn.training.contract import (
    LossWeightingStrategy,
    TrainingStep,
)


# =============================
# Test Implementations
# =============================
class ValidTrainingStep:
    """Valid implementation of TrainingStep."""

    def __call__(self) -> torch.Tensor:
        """Return a scalar training loss."""
        return torch.tensor(1.0)


class ValidWeightingStrategy:
    """Valid implementation of LossWeightingStrategy."""

    def update(self, epoch: int) -> dict[str, float]:
        """Return loss weights."""
        return {"pde": 1.0}


# =============================
# Contract Tests
# =============================
def test_training_step_contract() -> None:
    """A valid training step satisfies the contract."""
    training_step: TrainingStep = ValidTrainingStep()

    loss = training_step()

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_loss_weighting_contract() -> None:
    """A valid weighting strategy satisfies the contract."""
    strategy: LossWeightingStrategy = ValidWeightingStrategy()

    weights = strategy.update(10)

    assert weights == {"pde": 1.0}
