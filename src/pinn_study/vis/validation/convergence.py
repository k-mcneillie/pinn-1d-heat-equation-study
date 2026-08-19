# convergence.py
# - validation convergence visualisation


# =============================
# Import Libraries
# =============================
import matplotlib.pyplot as plt


# =============================
# Convergence Plot
# =============================
def plot_convergence(
    epochs: list[int],
    errors: list[float],
    path,
) -> None:
    """Plot validation error against training epoch."""

    figure, axes = plt.subplots(
        figsize=(8, 5),
    )

    axes.plot(
        epochs,
        errors,
        marker="o",
        markersize=3,
    )

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Validation error")
    axes.set_yscale("log")
    axes.set_title("Validation Convergence")

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
