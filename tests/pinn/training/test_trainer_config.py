# test_config.py
# - tests for training configuration


# =============================
# Import Libraries
# =============================
import pytest
from pydantic import ValidationError

from pinn_study.pinn.training.config import (
    CheckpointConfig,
    EarlyStoppingConfig,
    GradientClippingConfig,
    SchedulerConfig,
    TrainingConfig,
)


# =============================
# Training Configuration Tests
# =============================
class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_defaults_are_applied(self) -> None:
        """Default training configuration is applied."""
        config = TrainingConfig()

        assert config.epochs == 1
        assert config.learning_rate == 1e-3
        assert config.optimizer == "adam"
        assert config.log_interval == 100

    def test_accepts_valid_configuration(self) -> None:
        """A valid training configuration is accepted."""
        config = TrainingConfig(
            epochs=100,
            learning_rate=1e-4,
            optimizer="sgd",
            log_interval=10,
        )

        assert config.epochs == 100
        assert config.learning_rate == 1e-4
        assert config.optimizer == "sgd"
        assert config.log_interval == 10

    @pytest.mark.parametrize("epochs", [0, -1])
    def test_rejects_non_positive_epochs(self, epochs: int) -> None:
        """Non-positive epoch counts are rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=epochs)

    @pytest.mark.parametrize("learning_rate", [0.0, -1e-3])
    def test_rejects_non_positive_learning_rate(
        self,
        learning_rate: float,
    ) -> None:
        """Non-positive learning rates are rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=learning_rate)

    @pytest.mark.parametrize("log_interval", [0, -1])
    def test_rejects_non_positive_log_interval(
        self,
        log_interval: int,
    ) -> None:
        """Non-positive logging intervals are rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(log_interval=log_interval)

    @pytest.mark.parametrize("optimizer", ["invalid", "adamw"])
    def test_rejects_unsupported_optimizer(
        self,
        optimizer: str,
    ) -> None:
        """Unsupported optimisers are rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(optimizer=optimizer)


# =============================
# Scheduler Configuration Tests
# =============================
class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_defaults_are_applied(self) -> None:
        """Default scheduler configuration is applied."""
        config = SchedulerConfig()

        assert config.name == "none"
        assert config.step_size == 100
        assert config.gamma == 0.1
        assert config.warmup_epochs == 0

    @pytest.mark.parametrize(
        "name",
        ["none", "step", "exponential", "cosine"],
    )
    def test_accepts_supported_schedulers(self, name: str) -> None:
        """Supported schedulers are accepted."""
        config = SchedulerConfig(name=name)

        assert config.name == name

    @pytest.mark.parametrize("warmup_epochs", [-1])
    def test_rejects_negative_warmup(
        self,
        warmup_epochs: int,
    ) -> None:
        """Negative warm-up epochs are rejected."""
        with pytest.raises(ValidationError):
            SchedulerConfig(warmup_epochs=warmup_epochs)

    def test_rejects_non_positive_step_size(self) -> None:
        """Non-positive step size is rejected."""
        with pytest.raises(ValidationError):
            SchedulerConfig(step_size=0)

    def test_rejects_non_positive_gamma(self) -> None:
        """Non-positive scheduler gamma is rejected."""
        with pytest.raises(ValidationError):
            SchedulerConfig(gamma=0.0)


# =============================
# Gradient Clipping Tests
# =============================
class TestGradientClippingConfig:
    """Tests for GradientClippingConfig."""

    def test_defaults_are_applied(self) -> None:
        """Gradient clipping is disabled by default."""
        config = GradientClippingConfig()

        assert config.enabled is False
        assert config.max_norm == 1.0

    def test_accepts_valid_configuration(self) -> None:
        """Valid gradient clipping configuration is accepted."""
        config = GradientClippingConfig(
            enabled=True,
            max_norm=0.5,
        )

        assert config.enabled is True
        assert config.max_norm == 0.5

    def test_rejects_non_positive_max_norm(self) -> None:
        """Non-positive maximum norm is rejected."""
        with pytest.raises(ValidationError):
            GradientClippingConfig(max_norm=0.0)


# =============================
# Early Stopping Tests
# =============================
class TestEarlyStoppingConfig:
    """Tests for EarlyStoppingConfig."""

    def test_defaults_are_applied(self) -> None:
        """Early stopping is disabled by default."""
        config = EarlyStoppingConfig()

        assert config.enabled is False
        assert config.patience == 10
        assert config.min_delta == 0.0

    def test_accepts_valid_configuration(self) -> None:
        """Valid early stopping configuration is accepted."""
        config = EarlyStoppingConfig(
            enabled=True,
            patience=5,
            min_delta=1e-4,
        )

        assert config.enabled is True
        assert config.patience == 5
        assert config.min_delta == 1e-4

    def test_rejects_non_positive_patience(self) -> None:
        """Non-positive patience is rejected."""
        with pytest.raises(ValidationError):
            EarlyStoppingConfig(patience=0)


# =============================
# Checkpoint Configuration Tests
# =============================
class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_defaults_are_applied(self) -> None:
        """Checkpointing is disabled by default."""
        config = CheckpointConfig()

        assert config.enabled is False
        assert config.interval == 100
        assert config.save_best is True

    def test_accepts_valid_configuration(self) -> None:
        """Valid checkpoint configuration is accepted."""
        config = CheckpointConfig(
            enabled=True,
            interval=25,
            save_best=False,
        )

        assert config.enabled is True
        assert config.interval == 25
        assert config.save_best is False

    def test_rejects_non_positive_interval(self) -> None:
        """Non-positive checkpoint intervals are rejected."""
        with pytest.raises(ValidationError):
            CheckpointConfig(interval=0)
