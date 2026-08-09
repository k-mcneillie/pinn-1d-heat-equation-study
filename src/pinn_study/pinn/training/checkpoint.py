# checkpoint.py
# - checkpoint persistence for model training


# =============================
# Import Libraries
# =============================
from pathlib import Path

import torch


# =============================
# Checkpoint Manager
# =============================
class TrainingCheckpointManager:
    """Save and load model training state."""

    def __init__(self, output_dir: Path) -> None:
        """Initialise checkpoint manager.

        Args:
            output_dir: Directory in which checkpoints are stored.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # =============================
    # Save
    # =============================
    def save(
        self,
        epoch: int,
        model_state: dict,
        optimizer_state: dict,
        loss: float,
        history: list[float],
        scheduler_state: dict | None = None,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch.
            model_state: Model state dictionary.
            optimizer_state: Optimiser state dictionary.
            loss: Current loss.
            history: Training loss history.
            scheduler_state: Optional scheduler state.

        Returns:
            Path to the saved checkpoint.
        """
        path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state": model_state,
                "optimizer_state": optimizer_state,
                "scheduler_state": scheduler_state,
                "loss": loss,
                "history": history,
            },
            path,
        )

        return path
