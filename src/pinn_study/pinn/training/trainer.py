# trainer.py
# - reusable training loop for PINN models


# =============================
# Import Libraries
# =============================
import logging

import torch
import torch.nn as nn
from torch import Tensor

from pinn_study.pinn.loss.pinn_loss import PINNLoss

from .annealing import ConstantWeighting
from .checkpoint import TrainingCheckpointManager
from .config import TrainingConfig
from .contract import LossWeightingStrategy, TrainingStep
from .scheduler import WarmupScheduler, create_scheduler


# =============================
# Trainer
# =============================
# Trainer owns the optimisation lifecycle.
#
# It deliberately does not know how the physical problem calculates its
# residuals. The supplied TrainingStep handles that responsibility.
class Trainer:
    """Train a model using a configurable optimisation lifecycle."""

    def __init__(
        self,
        model: nn.Module,
        training_step: TrainingStep,
        config: TrainingConfig,
        logger: logging.Logger,
        loss: PINNLoss | None = None,
        weighting_strategy: LossWeightingStrategy | None = None,
        checkpoint_manager: TrainingCheckpointManager | None = None,
    ) -> None:
        """Initialise the trainer.

        Args:
            model: Neural network to optimise.
            training_step: Problem-specific loss calculation.
            config: Training configuration.
            logger: Logger used for training progress.
            loss: Optional PINN loss whose weights may be updated.
            weighting_strategy: Optional dynamic loss weighting strategy.
            checkpoint_manager: Optional checkpoint manager.
        """
        self.model = model
        self.training_step = training_step
        self.config = config
        self.logger = logger
        self.loss = loss
        self.weighting_strategy = (
            weighting_strategy
            if weighting_strategy is not None
            else ConstantWeighting({})
        )
        self.checkpoint_manager = checkpoint_manager

        self.optimizer = self._create_optimizer()
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config.scheduler,
        )
        self.warmup_scheduler = WarmupScheduler(
            self.optimizer,
            self.config.scheduler,
        )

        self.history: list[float] = []
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0

    # =============================
    # Optimiser
    # =============================
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the configured optimiser.

        Returns:
            Configured optimiser.
        """
        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }

        optimizer_class = optimizers[self.config.optimizer]

        return optimizer_class(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    # =============================
    # Training
    # =============================
    def train(self) -> list[float]:
        """Train the model.

        Returns:
            Training loss history.
        """
        self.model.train()

        for epoch in range(1, self.config.epochs + 1):
            self._update_loss_weights(epoch)

            self._apply_warmup(epoch)

            self.optimizer.zero_grad()

            loss = self.training_step()

            self._validate_loss(loss)

            loss.backward()

            self._clip_gradients()

            self.optimizer.step()

            self._step_scheduler(epoch)

            loss_value = loss.detach().item()
            self.history.append(loss_value)

            self._log_epoch(epoch, loss_value)

            self._save_checkpoint(epoch, loss_value)

            if self._should_stop(loss_value):
                self.logger.info(
                    "Early stopping triggered at epoch %d.",
                    epoch,
                )
                break

        return self.history

    # =============================
    # Loss Weighting
    # =============================
    def _update_loss_weights(self, epoch: int) -> None:
        """Update dynamic loss weights."""
        if self.loss is None:
            return

        weights = self.weighting_strategy.update(epoch)

        if weights:
            self.loss.update_weights(weights)

    # =============================
    # Warm-Up
    # =============================
    def _apply_warmup(self, epoch: int) -> None:
        """Apply learning-rate warm-up."""
        if self.config.scheduler.warmup_epochs > 0:
            self.warmup_scheduler.step(epoch)

    # =============================
    # Gradient Clipping
    # =============================
    def _clip_gradients(self) -> None:
        """Apply gradient clipping when configured."""
        if not self.config.gradient_clipping.enabled:
            return

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.gradient_clipping.max_norm,
        )

    # =============================
    # Scheduler
    # =============================
    def _step_scheduler(self, epoch: int) -> None:
        """Advance the learning-rate scheduler."""
        if self.scheduler is None:
            return

        if epoch <= self.config.scheduler.warmup_epochs:
            return

        self.scheduler.step()

    # =============================
    # Early Stopping
    # =============================
    def _should_stop(self, loss: float) -> bool:
        """Determine whether early stopping should occur."""
        config = self.config.early_stopping

        if not config.enabled:
            return False

        if loss < self.best_loss - config.min_delta:
            self.best_loss = loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return self.epochs_without_improvement >= config.patience

    # =============================
    # Checkpointing
    # =============================
    def _save_checkpoint(
        self,
        epoch: int,
        loss: float,
    ) -> None:
        """Save a checkpoint when configured."""
        config = self.config.checkpoint

        if not config.enabled:
            return

        if self.checkpoint_manager is None:
            return

        is_interval = epoch % config.interval == 0
        is_best = config.save_best and loss <= self.best_loss

        if is_interval or is_best:
            self.checkpoint_manager.save(
                epoch=epoch,
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=(
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                loss=loss,
                history=self.history.copy(),
            )

    # =============================
    # Loss Validation
    # =============================
    @staticmethod
    def _validate_loss(loss: Tensor) -> None:
        """Validate the training loss."""
        if not isinstance(loss, Tensor):
            raise TypeError("Training loss must be a torch.Tensor.")

        if loss.ndim != 0:
            raise ValueError("Training loss must be a scalar tensor.")

    # =============================
    # Logging
    # =============================
    def _log_epoch(
        self,
        epoch: int,
        loss: float,
    ) -> None:
        """Log training progress."""
        if (
            epoch % self.config.log_interval == 0
            or epoch == 1
            or epoch == self.config.epochs
        ):
            self.logger.info(
                "Epoch %d/%d | Loss: %.6e | LR: %.6e",
                epoch,
                self.config.epochs,
                loss,
                self.optimizer.param_groups[0]["lr"],
            )
