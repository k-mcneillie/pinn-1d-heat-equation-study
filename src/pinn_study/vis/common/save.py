# save.py
# - shared figure saving utilities


# =============================
# Import Libraries
# =============================
from pathlib import Path

import matplotlib.pyplot as plt


# =============================
# Save Figure
# =============================
def save_figure(
    figure: plt.Figure,
    path: Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
    tight_layout: bool = True,
) -> Path:
    """Save a figure to disk."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if tight_layout:
        figure.tight_layout()

    figure.savefig(
        path,
        dpi=dpi,
        transparent=transparent,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path