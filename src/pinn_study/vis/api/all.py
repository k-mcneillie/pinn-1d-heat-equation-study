# all.py
# - convenience API for complete visualisation suite


# =============================
# Import Libraries
# =============================
from pathlib import Path

from pinn_study.vis.api.experiment import visualise_experiment
from pinn_study.vis.result import VisualizationResult


# =============================
# Complete Visualisation API
# =============================
def visualise_all(
    *,
    dataset,
    training_history: dict,
    validation_data,
    model,
    xai_data,
    output_dir: Path,
) -> VisualizationResult:
    """Generate all available experiment visualisations.

    Args:
        dataset: Dataset used by the experiment.
        training_history: Recorded training history.
        validation_data: Data required for validation.
        model: Trained model.
        xai_data: Data required for XAI analysis.
        output_dir: Root output directory.

    Returns:
        Combined visualisation result.
    """
    return visualise_experiment(
        dataset=dataset,
        training_history=training_history,
        validation_data=validation_data,
        model=model,
        xai_data=xai_data,
        output_dir=output_dir,
    )