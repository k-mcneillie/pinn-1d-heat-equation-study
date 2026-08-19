# domain.py
# - domain and sampling visualisations


# =============================
# Import Libraries
# =============================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================
# Domain
# =============================
def plot_domain(
    x: np.ndarray,
    output_path: Path,
    *,
    title: str = "Model Domain",
) -> Path:
    """Plot the domain represented by input coordinates."""

    values = np.asarray(x).reshape(-1)

    figure, axis = plt.subplots()
    axis.scatter(values, np.zeros_like(values), s=10)
    axis.set_xlabel("x")
    axis.set_yticks([])
    axis.set_title(title)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Sampling Points
# =============================
def plot_sampling_points(
    x: np.ndarray,
    output_path: Path,
    *,
    title: str = "Collocation Points",
) -> Path:
    """Plot sampled collocation points."""

    values = np.asarray(x).reshape(-1)

    figure, axis = plt.subplots()
    axis.scatter(values, np.zeros_like(values), s=12)
    axis.set_xlabel("x")
    axis.set_yticks([])
    axis.set_title(title)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Special Points
# =============================
def plot_special_points(
    x: np.ndarray,
    output_path: Path,
    *,
    boundary: np.ndarray | None = None,
    initial: np.ndarray | None = None,
    title: str = "PINN Sampling Structure",
) -> Path:
    """Plot collocation, boundary and initial-condition points."""

    collocation = np.asarray(x).reshape(-1)

    figure, axis = plt.subplots()

    axis.scatter(
        collocation,
        np.zeros_like(collocation),
        label="Collocation",
        s=12,
    )

    if boundary is not None:
        boundary_values = np.asarray(boundary).reshape(-1)
        axis.scatter(
            boundary_values,
            np.ones_like(boundary_values),
            label="Boundary",
            s=20,
        )

    if initial is not None:
        initial_values = np.asarray(initial).reshape(-1)
        axis.scatter(
            initial_values,
            np.full_like(initial_values, 2.0),
            label="Initial",
            s=20,
        )

    axis.set_xlabel("x")
    axis.set_yticks([])
    axis.set_title(title)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path
