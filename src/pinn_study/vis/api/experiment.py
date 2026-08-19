# experiment.py
# - API for complete experiment visualisation


# =============================
# Import Libraries
# =============================
from pathlib import Path

from pinn_study.pinn.training.result import TrainingResult
from pinn_study.vis.api.data import visualise_dataset
from pinn_study.vis.api.training import visualise_training
from pinn_study.vis.api.validation import visualise_validation
from pinn_study.vis.api.xai import visualise_xai
from pinn_study.vis.result import VisualizationResult


# =============================
# Experiment Visualisation API
# =============================
def visualise_experiment(
    *,
    dataset,
    training_result: TrainingResult,
    validation_data,
    model,
    xai_data,
    output_dir: Path,
) -> VisualizationResult:
    """Generate the complete experiment visualisation suite.

    Args:
        dataset: Dataset used by the experiment.
        training_result: Recorded result from the Trainer.
        validation_data: Data required for validation plots.
        model: Trained model.
        xai_data: Data required for XAI analysis.
        output_dir: Root directory for generated figures.

    Returns:
        Combined visualisation result.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[Path] = []

    # =============================
    # Data Visualisations
    # =============================
    data_result = visualise_dataset(
        inputs=getattr(dataset, "inputs", None) if dataset is not None else None,
        targets=getattr(dataset, "targets", None) if dataset is not None else None,
        output_dir=output_dir / "data",
        boundary=getattr(dataset, "boundary", None) if dataset is not None else None,
        initial=getattr(dataset, "initial", None) if dataset is not None else None,
    )

    figures.extend(data_result.figures)

    # =============================
    # Training Visualisations
    # =============================
    training_visualisation = visualise_training(
        result=training_result,
        output_dir=output_dir / "training",
    )

    figures.extend(training_visualisation.figures)

    # =============================
    # Validation Visualisations
    # =============================
    validation_result = visualise_validation(
        validation_data=validation_data,
        output_dir=output_dir / "validation",
    )

    figures.extend(validation_result.figures)

    # =============================
    # XAI Visualisations
    # =============================
    xai_result = visualise_xai(
        model=model,
        xai_data=xai_data,
        output_dir=output_dir / "xai",
    )

    figures.extend(xai_result.figures)

    return VisualizationResult(
        output_dir=output_dir,
        figures=figures,
    )
