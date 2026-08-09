# solution.py
# - analytical and predicted solution visualisations


# =============================
# Import Libraries
# =============================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================
# Solution Comparison
# =============================
def plot_solution_comparison(
    x: np.ndarray,
    prediction: np.ndarray,
    analytical: np.ndarray,
    output_path: Path,
) -> Path:
    """Compare predicted and analytical solutions."""

    x_values = np.asarray(x).reshape(-1)
    prediction_values = np.asarray(prediction).reshape(-1)
    analytical_values = np.asarray(analytical).reshape(-1)

    figure, axis = plt.subplots()

    axis.plot(
        x_values,
        analytical_values,
        label="Analytical",
    )
    axis.plot(
        x_values,
        prediction_values,
        "--",
        label="PINN",
    )

    axis.set_xlabel("x")
    axis.set_ylabel("u(x)")
    axis.set_title("PINN Solution vs Analytical Solution")
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Legacy Solution Plot
def plot_solution(
    validation_data,
    output_path: Path,
) -> Path:
    """Plot solution comparison for legacy API compatibility."""
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
    prediction = None
    analytical = None

    if isinstance(validation_data, dict):
        x = validation_data.get("x")
        prediction = validation_data.get("prediction")
        analytical = validation_data.get("analytical")
    else:
        x = getattr(validation_data, "x", None)
        prediction = getattr(validation_data, "prediction", None)
        analytical = getattr(validation_data, "analytical", None)

    if x is not None and prediction is not None and analytical is not None:
        return plot_solution_comparison(
            x,
            prediction,
            analytical,
            output_path,
        )

    figure, axis = plt.subplots()
    axis.text(
        0.5,
        0.5,
        "Unable to plot solution",
        ha="center",
        va="center",
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Legacy Solution Plot
def plot_solution(
    validation_data,
    output_path: Path,
) -> Path:
    """Plot solution comparison for legacy API compatibility."""
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
    prediction = None
    analytical = None

    if isinstance(validation_data, dict):
        x = validation_data.get("x")
        prediction = validation_data.get("prediction")
        analytical = validation_data.get("analytical")
    else:
        x = getattr(validation_data, "x", None)
        prediction = getattr(validation_data, "prediction", None)
        analytical = getattr(validation_data, "analytical", None)

    if x is not None and prediction is not None and analytical is not None:
        return plot_solution_comparison(
            x,
            prediction,
            analytical,
            output_path,
        )

    figure, axis = plt.subplots()
    axis.text(
        0.5,
        0.5,
        "Unable to plot solution",
        ha="center",
        va="center",
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path

# =============================
# Legacy Solution Plot
def plot_solution(
    validation_data,
    output_path: Path,
) -> Path:
    """Plot solution comparison for legacy API compatibility."""
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
    prediction = None
    analytical = None

    if isinstance(validation_data, dict):
        x = validation_data.get("x")
        prediction = validation_data.get("prediction")
        analytical = validation_data.get("analytical")
    else:
        x = getattr(validation_data, "x", None)
        prediction = getattr(validation_data, "prediction", None)
        analytical = getattr(validation_data, "analytical", None)

    if x is not None and prediction is not None and analytical is not None:
        return plot_solution_comparison(
            x,
            prediction,
            analytical,
            output_path,
        )

    figure, axis = plt.subplots()
    axis.text(
        0.5,
        0.5,
        "Unable to plot solution",
        ha="center",
        va="center",
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path
