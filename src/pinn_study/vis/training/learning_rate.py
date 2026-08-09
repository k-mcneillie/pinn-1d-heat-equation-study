# learning_rate.py
# - learning-rate visualisation


# =============================
# Import Libraries
# =============================
import matplotlib.pyplot as plt

from pinn_study.pinn.training.result import TrainingResult


# =============================
# Learning Rate
# =============================
def plot_learning_rate(
    result: TrainingResult,
    path,
) -> None:
    """Plot learning rate throughout training."""

    figure, axes = plt.subplots(
        figsize=(8, 5),
    )

    axes.plot(
        result.epochs,
        result.learning_rates,
    )

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Learning rate")
    axes.set_yscale("log")
    axes.set_title("Learning Rate Schedule")

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