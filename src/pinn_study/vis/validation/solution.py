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
    """Compare predicted and analytical solutions as a line plot."""
    x_values = np.asarray(x).reshape(-1)
    prediction_values = np.asarray(prediction).reshape(-1)
    analytical_values = np.asarray(analytical).reshape(-1)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        x_values,
        analytical_values,
        label="Analytical",
        linewidth=2,
    )
    axis.plot(
        x_values,
        prediction_values,
        "--",
        label="PINN",
        linewidth=1.5,
    )

    axis.set_xlabel("x")
    axis.set_ylabel("u(x, t)")
    axis.set_title("PINN Solution vs Analytical Solution")
    axis.legend()
    axis.grid(True)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


def plot_solution_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    prediction: np.ndarray,
    analytical: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot analytical and predicted solution heatmaps plus error and slice overlays."""
    x_values = np.asarray(x).reshape(-1)
    t_values = np.asarray(t).reshape(-1)
    predicted_values = np.asarray(prediction)
    analytical_values = np.asarray(analytical)

    if predicted_values.ndim != 2 or analytical_values.ndim != 2:
        return plot_solution_comparison(
            x_values, predicted_values, analytical_values, output_path
        )

    error_values = np.abs(predicted_values - analytical_values)
    selected_indices = np.linspace(0, len(t_values) - 1, num=5, dtype=int)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    mesh0 = axes[0, 0].pcolormesh(
        x_values,
        t_values,
        analytical_values,
        shading="auto",
    )
    axes[0, 0].set_title("Analytical solution")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("t")
    figure.colorbar(mesh0, ax=axes[0, 0], label="u(x,t)")

    mesh1 = axes[0, 1].pcolormesh(
        x_values,
        t_values,
        predicted_values,
        shading="auto",
    )
    axes[0, 1].set_title("PINN prediction")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("t")
    figure.colorbar(mesh1, ax=axes[0, 1], label="u(x,t)")

    mesh2 = axes[1, 0].pcolormesh(
        x_values,
        t_values,
        error_values,
        shading="auto",
    )
    axes[1, 0].set_title("Absolute error")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("t")
    figure.colorbar(mesh2, ax=axes[1, 0], label="|u_pred - u_exact|")

    for idx in selected_indices:
        axes[1, 1].plot(
            x_values,
            analytical_values[idx, :],
            label=f"Analytical t={t_values[idx]:.2f}",
            linewidth=1.5,
        )
        axes[1, 1].plot(
            x_values,
            predicted_values[idx, :],
            "--",
            label=f"PINN t={t_values[idx]:.2f}",
            linewidth=1.2,
        )

    axes[1, 1].set_title("Solution slices at selected times")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("u(x, t)")
    axes[1, 1].legend(fontsize="small", ncol=1)
    axes[1, 1].grid(True)

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

    if x is not None and prediction is not None and analytical is not None:
        x_values = np.asarray(x)
        prediction_values = np.asarray(prediction)
        analytical_values = np.asarray(analytical)

        if prediction_values.ndim == 2 and t is not None:
            return plot_solution_heatmap(
                x_values,
                np.asarray(t),
                prediction_values,
                analytical_values,
                output_path,
            )

        return plot_solution_comparison(
            x_values,
            prediction_values,
            analytical_values,
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
