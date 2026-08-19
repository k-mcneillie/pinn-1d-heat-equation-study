# distributions.py
# - data distribution visualisations


# =============================
# Import Libraries
# =============================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================
# Input Distribution
# =============================
def plot_input_distribution(
    inputs: np.ndarray,
    output_path: Path,
    *,
    title: str = "Input Distribution",
) -> Path:
    """Plot the distribution of model inputs."""

    values = np.asarray(inputs).reshape(-1)

    figure, axis = plt.subplots()
    axis.hist(values, bins=30)
    axis.set_xlabel("Input")
    axis.set_ylabel("Count")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Target Distribution
# =============================
def plot_target_distribution(
    targets: np.ndarray,
    output_path: Path,
    *,
    title: str = "Target Distribution",
) -> Path:
    """Plot the distribution of target values."""

    values = np.asarray(targets).reshape(-1)

    figure, axis = plt.subplots()
    axis.hist(values, bins=30)
    axis.set_xlabel("Target")
    axis.set_ylabel("Count")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Distribution Comparison
# =============================
def plot_distribution_comparison(
    values: dict[str, np.ndarray],
    output_path: Path,
    *,
    title: str = "Distribution Comparison",
) -> Path:
    """Compare multiple value distributions."""

    figure, axis = plt.subplots()

    for name, data in values.items():
        axis.hist(
            np.asarray(data).reshape(-1),
            bins=30,
            alpha=0.5,
            label=name,
        )

    axis.set_xlabel("Value")
    axis.set_ylabel("Count")
    axis.set_title(title)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path
