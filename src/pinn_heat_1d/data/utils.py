# utils.py
# - data related utilities

# =============================
# Import Libraries
# =============================
from __future__ import annotations

from pathlib import Path

import torch

from .generator import HeatEquationData


# =============================
# Utilities Functions
# =============================
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
