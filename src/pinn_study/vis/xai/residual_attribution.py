# residual_attribution.py
# - PINN-specific residual sensitivity and attribution


# =============================
# Import Libraries
# =============================
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


# =============================
# Residual Sensitivity
# =============================
def compute_residual_sensitivity(
    x: Tensor,
    residual: Tensor,
) -> Tensor:
    """Calculate sensitivity of PDE residual with respect to input.

    Args:
        x: Input coordinates with requires_grad=True.
        residual: Scalar residual value at each coordinate.

    Returns:
        Absolute residual sensitivity at each coordinate.
    """
    if x.requires_grad and residual.requires_grad:
        gradient = torch.autograd.grad(
            residual,
            x,
            grad_outputs=torch.ones_like(residual),
            create_graph=False,
            retain_graph=True,
        )[0]

        return gradient.abs()

    x_values = x.detach().reshape(-1)
    residual_values = residual.detach().reshape(-1)

    if x_values.numel() < 2:
        return torch.zeros_like(residual)

    diffs = x_values[1:] - x_values[:-1]
    diffs = torch.where(diffs == 0, torch.ones_like(diffs), diffs)
    values = (residual_values[1:] - residual_values[:-1]) / diffs
    values = torch.cat([values, values[-1:]])

    return values.to(residual.device)


# =============================
# Residual Attribution
# =============================
def compute_residual_attribution(
    sensitivity: Tensor,
) -> Tensor:
    """Normalise residual sensitivity into an attribution score."""

    values = sensitivity.abs()

    maximum = values.max()

    if maximum.item() == 0.0:
        return torch.zeros_like(values)

    return values / maximum


# =============================
# Plot Residual Sensitivity
# =============================
def plot_residual_sensitivity(
    x: Tensor,
    sensitivity: Tensor,
    output_path: Path,
) -> Path:
    """Plot residual sensitivity across the domain."""

    figure, axis = plt.subplots()

    axis.plot(
        x.detach().cpu().numpy().reshape(-1),
        sensitivity.detach().cpu().numpy().reshape(-1),
    )

    axis.set_xlabel("x")
    axis.set_ylabel("|dR/dx|")
    axis.set_title("PDE Residual Sensitivity")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Plot Residual Attribution
# =============================
def plot_residual_attribution(
    x: Tensor,
    attribution: Tensor,
    output_path: Path,
) -> Path:
    """Plot normalised residual attribution."""

    figure, axis = plt.subplots()

    axis.plot(
        x.detach().cpu().numpy().reshape(-1),
        attribution.detach().cpu().numpy().reshape(-1),
    )

    axis.set_xlabel("x")
    axis.set_ylabel("Normalised attribution")
    axis.set_title("PINN Residual Attribution")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path