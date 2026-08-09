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

    return np.abs(
        np.asarray(prediction) - np.asarray(analytical)
    )


def relative_error(
    prediction: np.ndarray,
    analytical: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Calculate pointwise relative error."""

    return (
        np.abs(
            np.asarray(prediction) - np.asarray(analytical)
        )
        / np.maximum(np.abs(analytical), epsilon)
    )


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

    figure, axis = plt.subplots()
    axis.plot(np.asarray(x).reshape(-1), error.reshape(-1))

    axis.set_xlabel("x")
    axis.set_ylabel("Absolute error")
    axis.set_title("Absolute Prediction Error")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Relative Error
# =============================
def plot_relative_error(
    x: np.ndarray,
    prediction: np.ndarray,
    analytical: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot pointwise relative error."""

    error = relative_error(prediction, analytical)

    figure, axis = plt.subplots()
    axis.plot(np.asarray(x).reshape(-1), error.reshape(-1))

    axis.set_xlabel("x")
    axis.set_ylabel("Relative error")
    axis.set_title("Relative Prediction Error")

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

    figure, axis = plt.subplots()
    axis.hist(error.reshape(-1), bins=30)

    axis.set_xlabel("Absolute error")
    axis.set_ylabel("Count")
    axis.set_title("Prediction Error Distribution")

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

    prediction = None
    analytical = None

    if isinstance(validation_data, dict):
        prediction = validation_data.get("prediction")
        analytical = validation_data.get("analytical")
    else:
        prediction = getattr(validation_data, "prediction", None)
        analytical = getattr(validation_data, "analytical", None)

    if prediction is not None and analytical is not None:
        return plot_error_distribution(
            prediction,
            analytical,
            output_path,
        )

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

    prediction = None
    analytical = None

    if isinstance(validation_data, dict):
        prediction = validation_data.get("prediction")
        analytical = validation_data.get("analytical")
    else:
        prediction = getattr(validation_data, "prediction", None)
        analytical = getattr(validation_data, "analytical", None)

    if prediction is not None and analytical is not None:
        return plot_error_distribution(
            prediction,
            analytical,
            output_path,
        )

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
