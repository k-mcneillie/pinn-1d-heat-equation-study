# result.py
# - result returned by PINN loss calculation

# =============================
# Import Libraries
# =============================
from dataclasses import dataclass

from torch import Tensor


# =============================
# PINN Loss Result
# =============================
@dataclass
class PINNLossResult:
    """Result of a PINN loss calculation."""

    total: Tensor
    components: dict[str, Tensor]
