# residuals.py
# - physics residual visualisations


# =============================
# Import Libraries
# =============================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================
# PDE Residual
# =============================
def plot_pde_residual(
    x: np.ndarray,
    residual: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot the PDE residual across the domain."""

    x_values = np.asarray(x).reshape(-1)
    residual_values = np.asarray(residual).reshape(-1)

    figure, axis = plt.subplots()
    axis.plot(x_values, residual_values)

    axis.axhline(0.0, linestyle="--")
    axis.set_xlabel("x")
    axis.set_ylabel("PDE residual")
    axis.set_title("PDE Residual")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Legacy Residual Plot
# =============================

def plot_residuals(
    validation_data,
    output_path: Path,
) -> Path:
    """Plot residuals from validation data for legacy API compatibility."""
    if validation_data is None:
        figure, axis = plt.subplots()
        axis.text(
            0.5,
            0.5,
            "No validation data",
            ha="center",
            va="center",
        )
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_path, dpi=300)
        plt.close(figure)

        return output_path

    x = None
    residual = None

    if isinstance(validation_data, dict):
        x = validation_data.get("x")
        residual = validation_data.get("residual")
    else:
        x = getattr(validation_data, "x", None)
        residual = getattr(validation_data, "residual", None)

    if x is not None and residual is not None:
        return plot_pde_residual(
            x,
            residual,
            output_path,
        )

    figure, axis = plt.subplots()
    axis.text(
        0.5,
        0.5,
        "Unable to plot residuals",
        ha="center",
        va="center",
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Residual Distribution
# =============================
def plot_residual_distribution(
    residual: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot the residual distribution."""

    values = np.asarray(residual).reshape(-1)

    figure, axis = plt.subplots()
    axis.hist(values, bins=30)

    axis.set_xlabel("PDE residual")
    axis.set_ylabel("Count")
    axis.set_title("PDE Residual Distribution")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path
    if validation_data is None:
        figure, axis = plt.subplots()
        axis.text(
            0.5,
            0.5,
            "No validation data",
            ha="center",
            va="center",
        )
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_path, dpi=300)
        plt.close(figure)

        return output_path

    x = None
    residual = None

    if isinstance(validation_data, dict):
        x = validation_data.get("x")
        residual = validation_data.get("residual")
    else:
        x = getattr(validation_data, "x", None)
        residual = getattr(validation_data, "residual", None)

    if x is not None and residual is not None:
        return plot_pde_residual(
            x,
            residual,
            output_path,
        )

    figure, axis = plt.subplots()
    axis.text(
        0.5,
        0.5,
        "Unable to plot residuals",
        ha="center",
        va="center",
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path
