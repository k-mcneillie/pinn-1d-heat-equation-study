# seed.py
# - Random seed utilities for reproducible experiments.

# =============================
# Import Libraries
# =============================
from __future__ import annotations

import random

import numpy as np
import torch


# =============================
# Initialise Seed
# =============================
def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.

    Args:
        seed: Seed value used by Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
