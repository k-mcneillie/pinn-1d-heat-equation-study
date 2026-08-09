# config.py
# - training configuration for model optimisation

# =============================
# Import Libraries
# =============================
from typing import Literal

from pydantic import BaseModel, Field


# =============================
# Scheduler Configuration
# =============================
class SchedulerConfig(BaseModel):
    """Configuration for learning-rate scheduling."""

    name: Literal[
        "none",
        "step",
        "exponential",
        "cosine",
    ] = "none"

    step_size: int = Field(default=100, gt=0)
    gamma: float = Field(default=0.1, gt=0.0)
    warmup_epochs: int = Field(default=0, ge=0)


# =============================
# Early Stopping Configuration
# =============================
class EarlyStoppingConfig(BaseModel):
    """Configuration for early stopping."""

    enabled: bool = False
    patience: int = Field(default=10, gt=0)
    min_delta: float = Field(default=0.0, ge=0.0)


# =============================
# Gradient Clipping Configuration
# =============================
class GradientClippingConfig(BaseModel):
    """Configuration for gradient clipping."""

    enabled: bool = False
    max_norm: float = Field(default=1.0, gt=0.0)


# =============================
# Checkpoint Configuration
# =============================
class CheckpointConfig(BaseModel):
    """Configuration for model checkpointing."""

    enabled: bool = False
    interval: int = Field(default=100, gt=0)
    save_best: bool = True


# =============================
# Training Configuration
# =============================
class TrainingConfig(BaseModel):
    """Configuration for model training."""

    epochs: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)

    optimizer: Literal[
        "adam",
        "sgd",
    ] = "adam"

    log_interval: int = Field(default=100, gt=0)

    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
    )

    gradient_clipping: GradientClippingConfig = Field(
        default_factory=GradientClippingConfig,
    )

    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig,
    )

    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig,
    )
