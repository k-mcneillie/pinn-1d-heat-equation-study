# utils.py
# - data related utilities

# =============================
# Import Libraries
# =============================
from __future__ import annotations

from pathlib import Path

import torch

from .card import DatasetCard
from .generator import HeatEquationData


# =============================
# Utilities Functions
# =============================
def save_dataset(
    dataset: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    card: DatasetCard | None = None,
) -> Path:
    """
    Save a generated dataset.

    Args:
        dataset: Dataset tensors to save.
        output_dir: Directory where the dataset should be saved.
        card: Optional dataset card to save alongside the data.
    """

    dataset_path = output_dir / "data.pt"
    torch.save(dataset, dataset_path)

    if card is not None:
        card.save(output_dir)

    return dataset_path


def load_dataset(path: Path) -> HeatEquationData:
    """Load a heat equation dataset from disk.
    Args:
        path: Path to the saved dataset.
    Returns:
        The loaded heat equation dataset.
    Raises:
        FileNotFoundError: If the dataset does not exist.
        RuntimeError: If the dataset cannot be loaded.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
