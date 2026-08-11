# tests/pinn/training/test_result.py
# - tests for training results


# =============================
# Import Libraries
# =============================
from pinn_study.pinn.training.result import TrainingResult


# =============================
# Training Result
# =============================
class TestTrainingResult:
    """Tests for TrainingResult."""

    def test_initialises_empty(self) -> None:
        """TrainingResult starts with empty histories."""
        result = TrainingResult()

        assert result.epochs == []
        assert result.losses == []
        assert result.learning_rates == []
        assert result.loss_components == {}

    def test_stores_epoch_history(self) -> None:
        """TrainingResult stores epoch history."""
        result = TrainingResult()

        result.epochs.extend([1, 2, 3])

        assert result.epochs == [1, 2, 3]

    def test_stores_loss_history(self) -> None:
        """TrainingResult stores total loss history."""
        result = TrainingResult()

        result.losses.extend(
            [1.0, 0.75, 0.5],
        )

        assert result.losses == [
            1.0,
            0.75,
            0.5,
        ]

    def test_stores_learning_rate_history(self) -> None:
        """TrainingResult stores learning-rate history."""
        result = TrainingResult()

        result.learning_rates.extend(
            [1e-3, 5e-4, 1e-4],
        )

        assert result.learning_rates == [
            1e-3,
            5e-4,
            1e-4,
        ]

    def test_stores_loss_component_history(self) -> None:
        """TrainingResult stores individual loss histories."""
        result = TrainingResult()

        result.loss_components["pde"] = [
            1.0,
            0.8,
            0.6,
        ]

        result.loss_components["boundary"] = [
            0.5,
            0.4,
            0.3,
        ]

        assert result.loss_components["pde"] == [
            1.0,
            0.8,
            0.6,
        ]

        assert result.loss_components["boundary"] == [
            0.5,
            0.4,
            0.3,
        ]

    def test_result_instances_do_not_share_state(self) -> None:
        """Separate TrainingResult instances have independent state."""
        first = TrainingResult()
        second = TrainingResult()

        first.epochs.append(1)
        first.losses.append(1.0)
        first.learning_rates.append(1e-3)
        first.loss_components["pde"] = [1.0]

        assert second.epochs == []
        assert second.losses == []
        assert second.learning_rates == []
        assert second.loss_components == {}
