# training.py
# - API for generating training visualisations


# =============================
# Import Libraries
# =============================
from pathlib import Path

from pinn_study.pinn.training.result import TrainingResult
from pinn_study.vis.result import VisualizationResult
from pinn_study.vis.training.learning_rate import plot_learning_rate
from pinn_study.vis.training.loss import plot_training_loss
from pinn_study.vis.training.loss_components import (
    plot_loss_components,
)


# =============================
# Training Visualisation API
# =============================
def visualise_training(
    result: TrainingResult,
    output_dir: Path,
) -> VisualizationResult:
    """Generate the complete training visualisation suite.

    Args:
        result: Recorded result from the Trainer.
        output_dir: Directory in which figures are saved.

    Returns:
        Result containing generated figure paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Path] = []

    # =============================
    # Total Loss
    # =============================
    loss_path = output_dir / "loss.png"

    plot_training_loss(
        result=result,
        path=loss_path,
    )

    figures.append(loss_path)

    # =============================
    # Loss Components
    # =============================
    if result.loss_components:
        components_path = output_dir / "loss_components.png"

        plot_loss_components(
            result=result,
            path=components_path,
        )

        figures.append(components_path)

    # =============================
    # Learning Rate
    # =============================
    if result.learning_rates:
        learning_rate_path = output_dir / "learning_rate.png"

        plot_learning_rate(
            result=result,
            path=learning_rate_path,
        )

        figures.append(learning_rate_path)

    return VisualizationResult(
        output_dir=output_dir,
        figures=figures,
    )