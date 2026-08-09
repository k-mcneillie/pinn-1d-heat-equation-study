# scheduler.py
# - learning-rate scheduling and warm-up

# =============================
# Import Libraries
# =============================
import torch

from .config import SchedulerConfig


# =============================
# Warm-Up Scheduler
# =============================
class WarmupScheduler:
    """Apply linear learning-rate warm-up before another scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: SchedulerConfig,
    ) -> None:
        """Initialise the scheduler.

        Args:
            optimizer: Optimiser whose learning rate will be scheduled.
            config: Scheduler configuration.
        """
        self.optimizer = optimizer
        self.config = config
        self.base_learning_rates = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        """Update the learning rate for the current epoch.

        Args:
            epoch: Current epoch.
        """
        if self.config.warmup_epochs == 0:
            return

        if epoch > self.config.warmup_epochs:
            return

        scale = epoch / self.config.warmup_epochs

        for index, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self.base_learning_rates[index] * scale


# =============================
# Scheduler Factory
# =============================
def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create the configured learning-rate scheduler.

    Args:
        optimizer: Optimiser to schedule.
        config: Scheduler configuration.

    Returns:
        Configured scheduler or None when scheduling is disabled.
    """
    if config.name == "none":
        return None

    if config.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )

    if config.name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
        )

    if config.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.warmup_epochs),
        )

    raise ValueError(f"Unsupported scheduler: {config.name}")
