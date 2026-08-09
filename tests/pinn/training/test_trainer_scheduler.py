# test_scheduler.py
# - tests for learning-rate scheduling


# =============================
# Import Libraries
# =============================
import torch

from pinn_study.pinn.training.config import SchedulerConfig
from pinn_study.pinn.training.scheduler import (
    WarmupScheduler,
    create_scheduler,
)


# =============================
# Scheduler Factory Tests
# =============================
class TestCreateScheduler:
    """Tests for create_scheduler."""

    def test_none_returns_none(self) -> None:
        """No scheduler is returned when scheduling is disabled."""
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        scheduler = create_scheduler(
            optimizer,
            SchedulerConfig(name="none"),
        )

        assert scheduler is None

    def test_creates_step_scheduler(self) -> None:
        """Step scheduler is created correctly."""
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        scheduler = create_scheduler(
            optimizer,
            SchedulerConfig(
                name="step",
                step_size=10,
                gamma=0.5,
            ),
        )

        assert isinstance(
            scheduler,
            torch.optim.lr_scheduler.StepLR,
        )

    def test_creates_exponential_scheduler(self) -> None:
        """Exponential scheduler is created correctly."""
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        scheduler = create_scheduler(
            optimizer,
            SchedulerConfig(
                name="exponential",
                gamma=0.9,
            ),
        )

        assert isinstance(
            scheduler,
            torch.optim.lr_scheduler.ExponentialLR,
        )

    def test_creates_cosine_scheduler(self) -> None:
        """Cosine scheduler is created correctly."""
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        scheduler = create_scheduler(
            optimizer,
            SchedulerConfig(
                name="cosine",
            ),
        )

        assert isinstance(
            scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )


# =============================
# Warm-Up Tests
# =============================
class TestWarmupScheduler:
    """Tests for WarmupScheduler."""

    def test_warmup_starts_below_base_learning_rate(self) -> None:
        """Warm-up starts below the configured learning rate."""
        model = torch.nn.Linear(1, 1)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        scheduler = WarmupScheduler(
            optimizer,
            SchedulerConfig(warmup_epochs=4),
        )

        scheduler.step(1)

        assert optimizer.param_groups[0]["lr"] == 2.5e-4

    def test_warmup_reaches_base_learning_rate(self) -> None:
        """Warm-up reaches the base learning rate."""
        model = torch.nn.Linear(1, 1)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        scheduler = WarmupScheduler(
            optimizer,
            SchedulerConfig(warmup_epochs=4),
        )

        scheduler.step(4)

        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_warmup_does_not_change_learning_rate_after_completion(
        self,
    ) -> None:
        """Warm-up does not alter the rate after completion."""
        model = torch.nn.Linear(1, 1)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        scheduler = WarmupScheduler(
            optimizer,
            SchedulerConfig(warmup_epochs=4),
        )

        scheduler.step(4)
        scheduler.step(5)

        assert optimizer.param_groups[0]["lr"] == 1e-3
