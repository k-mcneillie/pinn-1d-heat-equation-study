# result.py
# - result container for a completed model training run

# =============================
# Import Libraries
# =============================
from dataclasses import dataclass, field


# =============================
# Training Result
# =============================
@dataclass
class TrainingResult:
    """Store the results produced during model training."""

    epochs: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    loss_components: dict[str, list[float]] = field(
        default_factory=dict,
    )
