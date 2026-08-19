# losses.py
# - training loss visualisation


# =============================
# Import Libraries
# =============================
import matplotlib.pyplot as plt

from pinn_study.pinn.training.result import TrainingResult


# =============================
# Total Loss
# =============================
def plot_training_loss(
    result: TrainingResult,
    path,
) -> None:
    """Plot total training loss."""

    figure, axes = plt.subplots(
        figsize=(8, 5),
    )

    axes.plot(
        result.epochs,
        result.losses,
    )

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_yscale("log")
    axes.set_title("Training Loss")

    axes.grid(
        True,
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
