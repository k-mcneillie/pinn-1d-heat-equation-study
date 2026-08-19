# validation.py
# - API for generating validation visualisations


# =============================
# Import Libraries
# =============================
from pathlib import Path

from pinn_study.vis.result import VisualizationResult
from pinn_study.vis.validation.error import plot_error
from pinn_study.vis.validation.residuals import plot_residuals
from pinn_study.vis.validation.solution import plot_solution


# =============================
# Validation Visualisation API
# =============================
def visualise_validation(
    validation_data,
    output_dir: Path,
) -> VisualizationResult:
    """Generate the complete validation visualisation suite.

    Args:
        validation_data: Data required for validation plots.
        output_dir: Directory in which figures are saved.

    Returns:
        Result containing generated figure paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Path] = []

    solution_path = output_dir / "solution.png"

    plot_solution(
        validation_data=validation_data,
        output_path=solution_path,
    )

    figures.append(solution_path)

    error_path = output_dir / "error.png"

    plot_error(
        validation_data=validation_data,
        output_path=error_path,
    )

    figures.append(error_path)

    residual_path = output_dir / "residuals.png"

    plot_residuals(
        validation_data=validation_data,
        output_path=residual_path,
    )

    figures.append(residual_path)

    return VisualizationResult(
        output_dir=output_dir,
        figures=figures,
    )
