# xai.py
# - API for generating PINN-specific XAI visualisations


# =============================
# Import Libraries
# =============================
from pathlib import Path

import matplotlib.pyplot as plt

from pinn_study.vis.result import VisualizationResult
from pinn_study.vis.xai.residual_attribution import (
    plot_residual_attribution,
)
from pinn_study.vis.xai.sensitivity import (
    plot_sensitivity,
)


# =============================
# XAI Visualisation API
# =============================
def visualise_xai(
    model,
    xai_data,
    output_dir: Path,
) -> VisualizationResult:
    """Generate the complete XAI visualisation suite.

    Args:
        model: Trained model.
        xai_data: Data required for XAI analysis.
        output_dir: Directory in which figures are saved.

    Returns:
        Result containing generated figure paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Path] = []

    attribution_path = output_dir / "residual_attribution.png"

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
        figure.savefig(attribution_path, dpi=300)
        plt.close(figure)
    else:
        x = None
        attribution = None

        if isinstance(xai_data, dict):
            x = xai_data.get("x")
            attribution = xai_data.get("attribution")

        if x is not None and attribution is not None:
            plot_residual_attribution(
                x,
                attribution,
                attribution_path,
            )
        else:
            figure, axis = plt.subplots()
            axis.text(
                0.5,
                0.5,
                "Unable to plot residual attribution",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
            figure.tight_layout()
            figure.savefig(attribution_path, dpi=300)
            plt.close(figure)

    figures.append(attribution_path)

    sensitivity_path = output_dir / "sensitivity.png"

    plot_sensitivity(
        model=model,
        xai_data=xai_data,
        output_path=sensitivity_path,
    )

    figures.append(sensitivity_path)

    return VisualizationResult(
        output_dir=output_dir,
        figures=figures,
    )
