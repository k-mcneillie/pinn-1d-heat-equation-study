"""Tests for training visualisation functions."""

from pathlib import Path

from pinn_study.pinn.training.result import TrainingResult
from pinn_study.vis.training.learning_rate import plot_learning_rate
from pinn_study.vis.training.loss_components import plot_loss_components
from pinn_study.vis.training.losses import plot_training_loss


def create_result() -> TrainingResult:
    """Create a representative training result."""
    return TrainingResult(
        epochs=[1, 2, 3],
        losses=[1.0, 0.5, 0.25],
        learning_rates=[1e-3, 1e-3, 1e-4],
        loss_components={
            "pde": [0.8, 0.4, 0.2],
            "boundary": [0.2, 0.1, 0.05],
        },
    )


class TestTrainingPlots:
    """Tests for training plots."""

    def test_training_loss(self, tmp_path: Path) -> None:
        """Training loss plot is generated."""
        output = tmp_path / "loss.png"

        result = plot_training_loss(
            create_result(),
            output,
        )

        assert result is None
        assert output.is_file()
        assert output.stat().st_size > 0

    def test_loss_components(self, tmp_path: Path) -> None:
        """Loss component plot is generated."""
        output = tmp_path / "components.png"

        result = plot_loss_components(
            create_result(),
            output,
        )

        assert result is None
        assert output.is_file()

    def test_learning_rate(self, tmp_path: Path) -> None:
        """Learning-rate plot is generated."""
        output = tmp_path / "learning_rate.png"

        result = plot_learning_rate(
            create_result(),
            output,
        )

        assert result is None
        assert output.is_file()