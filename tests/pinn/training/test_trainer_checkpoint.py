# test_checkpoint.py
# - tests for training checkpoint persistence


# =============================
# Import Libraries
# =============================
import torch

from pinn_study.pinn.training.checkpoint import (
    TrainingCheckpointManager,
)


# =============================
# Checkpoint Manager Tests
# =============================
class TestTrainingCheckpointManager:
    """Tests for TrainingCheckpointManager."""

    def test_creates_output_directory(self, tmp_path) -> None:
        """Checkpoint manager creates its output directory."""
        output_dir = tmp_path / "checkpoints"

        TrainingCheckpointManager(output_dir)

        assert output_dir.is_dir()

    def test_saves_checkpoint(self, tmp_path) -> None:
        """Checkpoint state is saved to disk."""
        manager = TrainingCheckpointManager(tmp_path / "checkpoints")

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        path = manager.save(
            epoch=10,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=None,
            loss=0.25,
            history=[1.0, 0.5, 0.25],
        )

        assert path.is_file()

    def test_saved_checkpoint_contains_training_state(
        self,
        tmp_path,
    ) -> None:
        """Saved checkpoint contains the expected state."""
        manager = TrainingCheckpointManager(tmp_path / "checkpoints")

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        manager.save(
            epoch=10,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=None,
            loss=0.25,
            history=[1.0, 0.5, 0.25],
        )

        path = tmp_path / "checkpoints" / "checkpoint_epoch_10.pt"

        checkpoint = torch.load(
            path,
            weights_only=False,
        )

        assert checkpoint["epoch"] == 10
        assert checkpoint["loss"] == 0.25
        assert checkpoint["history"] == [
            1.0,
            0.5,
            0.25,
        ]
        assert "model_state" in checkpoint
        assert "optimizer_state" in checkpoint
        assert "scheduler_state" in checkpoint
