# sensitivity.py
# - PINN solution sensitivity analysis


# =============================
# Import Libraries
# =============================
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


# =============================
# Solution Sensitivity
# =============================
def compute_solution_sensitivity(
    x: Tensor,
    prediction: Tensor,
) -> Tensor:
    """Calculate |du/dx|."""

    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True.")

    gradient = torch.autograd.grad(
        prediction,
        x,
        grad_outputs=torch.ones_like(prediction),
        create_graph=False,
        retain_graph=True,
    )[0]

    return gradient.abs()


# =============================
# Plot Solution Sensitivity
# =============================
def plot_solution_sensitivity(
    x: Tensor,
    sensitivity: Tensor,
    output_path: Path,
) -> Path:
    """Plot model-output sensitivity."""

    figure, axis = plt.subplots()

    axis.plot(
        x.detach().cpu().numpy().reshape(-1),
        sensitivity.detach().cpu().numpy().reshape(-1),
    )

    axis.set_xlabel("x")
    axis.set_ylabel("|du/dx|")
    axis.set_title("PINN Solution Sensitivity")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Legacy Sensitivity Plot
# =============================


def plot_sensitivity(
    model,
    xai_data,
    output_path: Path,
) -> Path:
    """Plot solution sensitivity for legacy API compatibility."""
    if xai_data is None:
        figure, axis = plt.subplots()
        axis.text(
            0.5,
            0.5,
            "No XAI data",
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
    if isinstance(xai_data, dict):
        x = xai_data.get("x")
        prediction = xai_data.get("prediction")
    else:
        x = getattr(xai_data, "x", None)
        prediction = getattr(xai_data, "prediction", None)

    if x is not None and prediction is not None:
        try:
            sensitivity = compute_solution_sensitivity(
                x,
                prediction,
            )
        except Exception:
            sensitivity = None

        if sensitivity is not None:
            return plot_solution_sensitivity(
                x,
                sensitivity,
                output_path,
            )

    figure, axis = plt.subplots()
    axis.text(
        0.5,
        0.5,
        "Unable to plot sensitivity",
        ha="center",
        va="center",
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Sensitivity vs Error
# =============================
def plot_sensitivity_vs_error(
    sensitivity: Tensor,
    error: Tensor,
    output_path: Path,
) -> Path:
    """Compare sensitivity against prediction error."""

    figure, axis = plt.subplots()

    axis.scatter(
        sensitivity.detach().cpu().numpy().reshape(-1),
        error.detach().cpu().numpy().reshape(-1),
        s=12,
    )

    axis.set_xlabel("Solution sensitivity")
    axis.set_ylabel("Prediction error")
    axis.set_title("Sensitivity vs Prediction Error")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path
