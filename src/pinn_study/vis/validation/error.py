# error.py
# - prediction error visualisations


# =============================
# Import Libraries
# =============================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================
# Error Calculation
# =============================
def absolute_error(
    prediction: np.ndarray,
    analytical: np.ndarray,
) -> np.ndarray:
    """Calculate absolute prediction error."""
    return np.abs(np.asarray(prediction) - np.asarray(analytical))


def relative_error(
    prediction: np.ndarray,
    analytical: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Calculate pointwise relative error."""
    return np.abs(np.asarray(prediction) - np.asarray(analytical)) / np.maximum(np.abs(np.asarray(analytical)), epsilon)


# =============================
# Absolute Error
# =============================
def plot_absolute_error(
    x: np.ndarray,
    prediction: np.ndarray,
    analytical: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot pointwise absolute error."""
    error = absolute_error(prediction, analytical)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.asarray(x).reshape(-1), error.reshape(-1), linewidth=1.5)
    axis.set_xlabel("x")
    axis.set_ylabel("Absolute error")
    axis.set_title("Absolute Prediction Error")
    axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Error Heatmap
# =============================
def plot_error_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    prediction: np.ndarray,
    analytical: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot the absolute error over the x-t validation mesh."""
    x_values = np.asarray(x).reshape(-1)
    t_values = np.asarray(t).reshape(-1)
    error = absolute_error(prediction, analytical)

    figure, axis = plt.subplots(figsize=(8, 5))
    mesh = axis.pcolormesh(x_values, t_values, error, shading="auto")
    axis.set_xlabel("x")
    axis.set_ylabel("t")
    axis.set_title("Absolute Prediction Error Heatmap")
    figure.colorbar(mesh, ax=axis, label="|u_pred - u_exact|")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Error Distribution
# =============================
def plot_error_distribution(
    prediction: np.ndarray,
    analytical: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot the distribution of absolute errors."""
    error = absolute_error(prediction, analytical)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(error.reshape(-1), bins=30)
    axis.set_xlabel("Absolute error")
    axis.set_ylabel("Count")
    axis.set_title("Prediction Error Distribution")
    axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Legacy Error Plot
def plot_error(
    validation_data,
    output_path: Path,
) -> Path:
    """Plot a generic error figure for legacy API compatibility."""
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
    prediction = None
    analytical = None

    if isinstance(validation_data, dict):
        x = validation_data.get("x")
        t = validation_data.get("t")
        prediction = validation_data.get("prediction")
        analytical = validation_data.get("analytical")
    else:
        x = getattr(validation_data, "x", None)
        t = getattr(validation_data, "t", None)
        prediction = getattr(validation_data, "prediction", None)
        analytical = getattr(validation_data, "analytical", None)

    if prediction is not None and analytical is not None:
        prediction_values = np.asarray(prediction)
        analytical_values = np.asarray(analytical)

        if prediction_values.ndim == 2 and x is not None and t is not None:
            return plot_error_heatmap(
                np.asarray(x),
                np.asarray(t),
                prediction_values,
                analytical_values,
                output_path,
            )

        if x is not None:
            return plot_absolute_error(
                np.asarray(x),
                prediction_values,
                analytical_values,
                output_path,
            )

        return plot_error_distribution(prediction_values, analytical_values, output_path)

    figure, axis = plt.subplots()
    axis.text(
        0.5,
        0.5,
        "Unable to plot error",
        ha="center",
        va="center",
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path
