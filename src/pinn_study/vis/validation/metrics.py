# metrics.py
# - numerical validation metrics


# =============================
# Import Libraries
# =============================
import torch


# =============================
# Validation Metrics
# =============================
def calculate_error_metrics(
    prediction: torch.Tensor,
    analytical: torch.Tensor,
) -> dict[str, float]:
    """Calculate standard solution-error metrics."""

    error = prediction - analytical

    absolute_error = error.abs()

    mse = torch.mean(
        error**2,
    )

    rmse = torch.sqrt(mse)

    mae = torch.mean(
        absolute_error,
    )

    max_error = torch.max(
        absolute_error,
    )

    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(analytical)

    return {
        "mse": mse.detach().item(),
        "rmse": rmse.detach().item(),
        "mae": mae.detach().item(),
        "max_error": max_error.detach().item(),
        "relative_l2": relative_l2.detach().item(),
    }
