# loss_components.py
# - PINN loss component visualisation


# =============================
# Import Libraries
# =============================
import matplotlib.pyplot as plt

from pinn_study.pinn.training.result import TrainingResult


# =============================
# Loss Components
# =============================
def plot_loss_components(
    result: TrainingResult,
    path,
) -> None:
    """Plot individual PINN constraint losses."""

    figure, axes = plt.subplots(
        figsize=(8, 5),
    )

    for name, values in result.loss_components.items():
        axes.plot(
            result.epochs,
            values,
            label=name,
        )

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_yscale("log")
    axes.set_title("PINN Loss Components")
    axes.legend()

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