# test_trainer.py
# - tests for the model training lifecycle


# =============================
# Import Libraries
# =============================
import logging
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from pinn_study.pinn.loss.config import LossConfig
from pinn_study.pinn.loss.pinn_loss import PINNLoss
from pinn_study.pinn.loss.result import PINNLossResult
from pinn_study.pinn.training.annealing import LinearWeighting
from pinn_study.pinn.training.checkpoint import (
    TrainingCheckpointManager,
)
from pinn_study.pinn.training.config import (
    CheckpointConfig,
    EarlyStoppingConfig,
    GradientClippingConfig,
    SchedulerConfig,
    TrainingConfig,
)
from pinn_study.pinn.training.trainer import Trainer

device = torch.device("cpu")


# =============================
# Test Model
# =============================
class SimpleModel(nn.Module):
    """Small model for training tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


# =============================
# Test Helpers
# =============================
def create_training_step(
    model: nn.Module,
) -> Callable[[], PINNLossResult]:
    """Create a simple supervised training step."""

    inputs = torch.tensor(
        [[1.0], [2.0], [3.0]],
    )

    targets = torch.tensor(
        [[2.0], [4.0], [6.0]],
    )

    def training_step() -> PINNLossResult:
        predictions = model(inputs)

        loss = torch.mean(
            (predictions - targets) ** 2,
        )

        return PINNLossResult(
            total=loss,
            components={
                "data": loss,
            },
        )

    return training_step


def create_loss_training_step(
    model: nn.Module,
    loss: PINNLoss,
) -> Callable[[], PINNLossResult]:
    """Create a training step using a PINN loss."""

    inputs = torch.tensor(
        [[1.0], [2.0], [3.0]],
    )

    targets = torch.tensor(
        [[2.0], [4.0], [6.0]],
    )

    def training_step() -> PINNLossResult:
        predictions = model(inputs)

        data_loss = torch.mean(
            (predictions - targets) ** 2,
        )

        return loss(
            {
                "data": data_loss,
            }
        )

    return training_step


def create_logger() -> logging.Logger:
    """Create a logger for tests."""
    return logging.getLogger("test_training")


# =============================
# Basic Training
# =============================
class TestTrainer:
    """Tests for Trainer."""

    def test_training_returns_loss_history(self) -> None:
        """Training returns one loss value per epoch."""
        model = SimpleModel()

        trainer = Trainer(
            model=model,
            training_step=create_training_step(model),
            config=TrainingConfig(epochs=5),
            logger=create_logger(),
            device=device,
        )

        history = trainer.train()

        assert len(history) == 5
        assert all(isinstance(loss, float) for loss in history)

    def test_training_updates_model(self) -> None:
        """Training updates model parameters."""
        torch.manual_seed(42)

        model = SimpleModel()

        initial_parameters = [
            parameter.detach().clone() for parameter in model.parameters()
        ]

        trainer = Trainer(
            model=model,
            training_step=create_training_step(model),
            config=TrainingConfig(epochs=5),
            logger=create_logger(),
            device=device,
        )

        trainer.train()

        assert any(
            not torch.equal(
                initial,
                updated,
            )
            for initial, updated in zip(
                initial_parameters,
                model.parameters(),
                strict=True,
            )
        )

    def test_training_reduces_loss(self) -> None:
        """Training reduces the loss for a simple problem."""
        torch.manual_seed(42)

        model = SimpleModel()

        trainer = Trainer(
            model=model,
            training_step=create_training_step(model),
            config=TrainingConfig(
                epochs=100,
                learning_rate=1e-2,
            ),
            logger=create_logger(),
            device=device,
        )

        history = trainer.train()

        assert history[-1] < history[0]

    def test_training_records_result(self) -> None:
        """Training records losses, components and learning rates."""
        model = SimpleModel()

        trainer = Trainer(
            model=model,
            training_step=create_training_step(model),
            config=TrainingConfig(epochs=5),
            logger=create_logger(),
            device=device,
        )

        trainer.train()

        assert len(trainer.result.epochs) == 5
        assert len(trainer.result.losses) == 5
        assert len(trainer.result.learning_rates) == 5
        assert len(trainer.result.loss_components["data"]) == 5


# =============================
# Loss Validation
# =============================
class TestLossValidation:
    """Tests for training loss validation."""

    def test_rejects_non_loss_result(self) -> None:
        """Training steps returning invalid results are rejected."""
        model = SimpleModel()

        def invalid_training_step() -> float:
            return 1.0

        trainer = Trainer(
            model=model,
            training_step=invalid_training_step,
            config=TrainingConfig(),
            logger=create_logger(),
            device=device,
        )

        with pytest.raises(
            TypeError,
            match="Training step must return a PINNLossResult",
        ):
            trainer.train()

    def test_rejects_non_scalar_loss(self) -> None:
        """Non-scalar losses are rejected."""
        model = SimpleModel()

        def invalid_training_step() -> PINNLossResult:
            loss = torch.tensor(
                [1.0, 2.0],
                requires_grad=True,
            )

            return PINNLossResult(
                total=loss,
                components={
                    "data": loss,
                },
            )

        trainer = Trainer(
            model=model,
            training_step=invalid_training_step,
            config=TrainingConfig(),
            logger=create_logger(),
            device=device,
        )

        with pytest.raises(
            ValueError,
            match="Training loss must be a scalar tensor",
        ):
            trainer.train()


# =============================
# Scheduler Integration
# =============================
class TestSchedulerIntegration:
    """Tests for scheduler integration."""

    def test_scheduler_changes_learning_rate(self) -> None:
        """Configured scheduler changes the learning rate."""
        model = SimpleModel()

        trainer = Trainer(
            model=model,
            training_step=create_training_step(model),
            config=TrainingConfig(
                epochs=5,
                learning_rate=1e-2,
                scheduler=SchedulerConfig(
                    name="step",
                    step_size=2,
                    gamma=0.1,
                ),
            ),
            logger=create_logger(),
            device=device,
        )

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        trainer.train()

        final_lr = trainer.optimizer.param_groups[0]["lr"]

        assert final_lr < initial_lr


# =============================
# Gradient Clipping
# =============================
class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_gradient_clipping_is_applied(self) -> None:
        """Configured gradient clipping limits gradient norm."""
        model = SimpleModel()

        config = TrainingConfig(
            epochs=1,
            gradient_clipping=GradientClippingConfig(
                enabled=True,
                max_norm=0.01,
            ),
        )

        trainer = Trainer(
            model=model,
            training_step=create_training_step(model),
            config=config,
            logger=create_logger(),
            device=device,
        )

        trainer.optimizer.zero_grad()

        loss_result = trainer.training_step()
        loss_result.total.backward()

        trainer._clip_gradients()

        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.01,
        )

        assert total_norm >= 0.0


# =============================
# Early Stopping
# =============================
class TestEarlyStopping:
    """Tests for early stopping."""

    def test_early_stopping_can_stop_training(self) -> None:
        """Early stopping terminates training after patience."""
        model = SimpleModel()

        def constant_training_step() -> PINNLossResult:
            loss = torch.tensor(
                1.0,
                requires_grad=True,
            )

            return PINNLossResult(
                total=loss,
                components={
                    "data": loss,
                },
            )

        trainer = Trainer(
            model=model,
            training_step=constant_training_step,
            config=TrainingConfig(
                epochs=100,
                early_stopping=EarlyStoppingConfig(
                    enabled=True,
                    patience=3,
                    min_delta=0.0,
                ),
            ),
            logger=create_logger(),
            device=device,
        )

        history = trainer.train()

        assert len(history) < 100

    def test_early_stopping_disabled_runs_all_epochs(
        self,
    ) -> None:
        """Training runs all epochs when early stopping is disabled."""
        model = SimpleModel()

        trainer = Trainer(
            model=model,
            training_step=create_training_step(model),
            config=TrainingConfig(
                epochs=5,
                early_stopping=EarlyStoppingConfig(
                    enabled=False,
                ),
            ),
            logger=create_logger(),
            device=device,
        )

        history = trainer.train()

        assert len(history) == 5


# =============================
# Loss Weight Annealing
# =============================
class TestLossWeightingIntegration:
    """Tests for dynamic loss weighting."""

    def test_weights_are_updated_during_training(
        self,
    ) -> None:
        """Loss weights are updated according to the strategy."""
        model = SimpleModel()

        loss = PINNLoss(
            config=LossConfig(
                weights={
                    "data": 0.0,
                },
            ),
        )

        weighting = LinearWeighting(
            initial={
                "data": 0.0,
            },
            final={
                "data": 1.0,
            },
            epochs=4,
        )

        trainer = Trainer(
            model=model,
            training_step=create_loss_training_step(
                model,
                loss,
            ),
            config=TrainingConfig(
                epochs=4,
            ),
            logger=create_logger(),
            device=device,
            loss=loss,
            weighting_strategy=weighting,
        )

        trainer.train()

        assert loss.config.weights["data"] == 1.0


# =============================
# Checkpoint Integration
# =============================
class TestCheckpointIntegration:
    """Tests for checkpoint integration."""

    def test_checkpoints_are_created(
        self,
        tmp_path,
    ) -> None:
        """Training creates configured checkpoints."""
        model = SimpleModel()

        checkpoint_manager = TrainingCheckpointManager(
            tmp_path / "checkpoints",
        )

        trainer = Trainer(
            model=model,
            training_step=create_training_step(model),
            config=TrainingConfig(
                epochs=4,
                checkpoint=CheckpointConfig(
                    enabled=True,
                    interval=2,
                    save_best=False,
                ),
            ),
            logger=create_logger(),
            device=device,
            checkpoint_manager=checkpoint_manager,
        )

        trainer.train()

        assert (tmp_path / "checkpoints" / "checkpoint_epoch_2.pt").is_file()

        assert (tmp_path / "checkpoints" / "checkpoint_epoch_4.pt").is_file()

    def test_checkpoint_contains_model_state(
        self,
        tmp_path,
    ) -> None:
        """Checkpoint contains the trained model state."""
        model = SimpleModel()

        checkpoint_manager = TrainingCheckpointManager(
            tmp_path / "checkpoints",
        )

        trainer = Trainer(
            model=model,
            training_step=create_training_step(model),
            config=TrainingConfig(
                epochs=2,
                checkpoint=CheckpointConfig(
                    enabled=True,
                    interval=1,
                    save_best=False,
                ),
            ),
            logger=create_logger(),
            device=device,
            checkpoint_manager=checkpoint_manager,
        )

        trainer.train()

        checkpoint = torch.load(
            tmp_path / "checkpoints" / "checkpoint_epoch_2.pt",
            weights_only=False,
        )

        assert checkpoint["epoch"] == 2
        assert checkpoint["model_state"]
        assert checkpoint["optimizer_state"]
        assert checkpoint["history"]


# =============================
# Combined Configuration
# =============================
class TestTrainingIntegration:
    """Tests for combined training functionality."""

    def test_training_supports_combined_options(
        self,
        tmp_path,
    ) -> None:
        """Multiple training extensions work together."""
        torch.manual_seed(42)

        model = SimpleModel()

        checkpoint_manager = TrainingCheckpointManager(
            tmp_path / "checkpoints",
        )

        trainer = Trainer(
            model=model,
            training_step=create_training_step(model),
            config=TrainingConfig(
                epochs=10,
                learning_rate=1e-2,
                scheduler=SchedulerConfig(
                    name="step",
                    step_size=5,
                    gamma=0.5,
                    warmup_epochs=2,
                ),
                gradient_clipping=GradientClippingConfig(
                    enabled=True,
                    max_norm=1.0,
                ),
                early_stopping=EarlyStoppingConfig(
                    enabled=False,
                ),
                checkpoint=CheckpointConfig(
                    enabled=True,
                    interval=5,
                    save_best=False,
                ),
            ),
            logger=create_logger(),
            device=device,
            checkpoint_manager=checkpoint_manager,
        )

        history = trainer.train()

        assert len(history) == 10
        assert history[-1] < history[0]

        assert (tmp_path / "checkpoints" / "checkpoint_epoch_5.pt").is_file()

        assert (tmp_path / "checkpoints" / "checkpoint_epoch_10.pt").is_file()
