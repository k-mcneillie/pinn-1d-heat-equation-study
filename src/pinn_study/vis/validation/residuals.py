# residuals.py
# - residual visualisations


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

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x_values, residual_values, linewidth=1.5)
    axis.axhline(0.0, linestyle="--", color="gray")
    axis.set_xlabel("x")
    axis.set_ylabel("PDE residual")
    axis.set_title("PDE Residual")
    axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Residual Heatmap
# =============================
def plot_residual_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    residual: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot the PDE residual as a heatmap over x and t."""
    x_values = np.asarray(x).reshape(-1)
    t_values = np.asarray(t).reshape(-1)
    residual_values = np.asarray(residual)

    figure, axis = plt.subplots(figsize=(8, 5))
    mesh = axis.pcolormesh(x_values, t_values, residual_values, shading="auto")
    axis.set_xlabel("x")
    axis.set_ylabel("t")
    axis.set_title("PDE Residual Heatmap")
    figure.colorbar(mesh, ax=axis, label="Residual")

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
    t = None
    residual = None

    if isinstance(validation_data, dict):
        x = validation_data.get("x")
        t = validation_data.get("t")
        residual = validation_data.get("residual")
    else:
        x = getattr(validation_data, "x", None)
        t = getattr(validation_data, "t", None)
        residual = getattr(validation_data, "residual", None)

    if residual is not None:
        residual_values = np.asarray(residual)
        if residual_values.ndim == 2 and x is not None and t is not None:
            return plot_residual_heatmap(
                np.asarray(x),
                np.asarray(t),
                residual_values,
                output_path,
            )

        if x is not None:
            return plot_pde_residual(
                x,
                residual_values,
                output_path,
            )

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(residual_values.reshape(-1), bins=30)
        axis.set_xlabel("PDE residual")
        axis.set_ylabel("Count")
        axis.set_title("PDE Residual Distribution")
        axis.grid(True)

        figure.tight_layout()
        figure.savefig(output_path, dpi=300)
        plt.close(figure)

        return output_path

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
