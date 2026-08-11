# data.py
# - API for generating data visualisations


# =============================
# Import Libraries
# =============================
from pathlib import Path

from pinn_study.vis.data.distributions import (
    plot_distribution_comparison,
    plot_input_distribution,
    plot_target_distribution,
)
from pinn_study.vis.data.domain import (
    plot_domain,
    plot_sampling_points,
    plot_special_points,
)
from pinn_study.vis.data.sampling import (
    plot_sampling_coverage,
    plot_sampling_density,
    plot_sampling_gaps,
)
from pinn_study.vis.result import VisualizationResult


# =============================
# Data Visualisation API
# =============================
def visualise_dataset(
    *,
    dataset=None,
    inputs=None,
    targets=None,
    output_dir: Path,
    boundary=None,
    initial=None,
) -> VisualizationResult:
    """Generate the complete dataset visualisation suite.

    Args:
        dataset: Optional dataset object with inputs/targets.
        inputs: Model input coordinates.
        targets: Analytical target values.
        output_dir: Directory in which figures are saved.
        boundary: Optional boundary points.
        initial: Optional initial-condition points.

    Returns:
        Result containing generated figure paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset is not None:
        if inputs is None:
            inputs = getattr(dataset, "inputs", None)
        if targets is None:
            targets = getattr(dataset, "targets", None)
        if boundary is None:
            boundary = getattr(dataset, "boundary", None)
        if initial is None:
            initial = getattr(dataset, "initial", None)

    if inputs is None or targets is None:
        return VisualizationResult(
            output_dir=output_dir,
            figures=[],
        )

    figures: list[Path] = []

    # =============================
    # Domain
    # =============================
    domain_path = output_dir / "domain.png"

    plot_domain(
        inputs,
        domain_path,
    )

    figures.append(domain_path)

    # =============================
    # Sampling Points
    # =============================
    sampling_points_path = output_dir / "sampling_points.png"

    plot_sampling_points(
        inputs,
        sampling_points_path,
    )

    figures.append(sampling_points_path)

    # =============================
    # Special Points
    # =============================
    if boundary is not None or initial is not None:
        special_points_path = output_dir / "special_points.png"

        plot_special_points(
            inputs,
            special_points_path,
            boundary=boundary,
            initial=initial,
        )

        figures.append(special_points_path)

    # =============================
    # Input Distribution
    # =============================
    input_distribution_path = output_dir / "input_distribution.png"

    plot_input_distribution(
        inputs,
        input_distribution_path,
    )

    figures.append(input_distribution_path)

    # =============================
    # Target Distribution
    # =============================
    target_distribution_path = output_dir / "target_distribution.png"

    plot_target_distribution(
        targets,
        target_distribution_path,
    )

    figures.append(target_distribution_path)

    # =============================
    # Input / Target Comparison
    # =============================
    distribution_comparison_path = (
        output_dir / "distribution_comparison.png"
    )

    plot_distribution_comparison(
        {
            "inputs": inputs,
            "targets": targets,
        },
        distribution_comparison_path,
    )

    figures.append(distribution_comparison_path)

    # =============================
    # Sampling Density
    # =============================
    sampling_density_path = output_dir / "sampling_density.png"

    plot_sampling_density(
        inputs,
        sampling_density_path,
    )

    figures.append(sampling_density_path)

    # =============================
    # Sampling Gaps
    # =============================
    sampling_gaps_path = output_dir / "sampling_gaps.png"

    plot_sampling_gaps(
        inputs,
        sampling_gaps_path,
    )

    figures.append(sampling_gaps_path)

    # =============================
    # Sampling Coverage
    # =============================
    sampling_coverage_path = output_dir / "sampling_coverage.png"

    plot_sampling_coverage(
        inputs,
        sampling_coverage_path,
    )

    figures.append(sampling_coverage_path)

    return VisualizationResult(
        output_dir=output_dir,
        figures=figures,
    )