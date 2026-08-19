# training_suite.py
# - generate the complete training visualisation suite


# =============================
# Import Libraries
# =============================
from pathlib import Path

from pinn_study.pinn.training.result import TrainingResult

from .training.learning_rate import plot_learning_rate
from .training.loss_components import plot_loss_components
from .training.losses import plot_training_loss


# =============================
# Training Visualisation Suite
# =============================
def generate_training_visualisations(
    result: TrainingResult,
    output_directory: Path,
) -> None:
    """Generate all training visualisations."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_training_loss(
        result,
        output_directory / "loss.png",
    )

    plot_loss_components(
        result,
        output_directory / "loss_components.png",
    )

    plot_learning_rate(
        result,
        output_directory / "learning_rate.png",
    )
