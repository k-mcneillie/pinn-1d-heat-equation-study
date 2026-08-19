# style.py
# - shared scientific plotting configuration


# =============================
# Import Libraries
# =============================
import matplotlib.pyplot as plt


# =============================
# Figure Creation
# =============================
def create_figure(
    width: float = 8.0,
    height: float = 5.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a standard scientific figure."""

    figure, axes = plt.subplots(
        figsize=(width, height),
    )

    axes.grid(
        True,
        alpha=0.25,
    )

    return figure, axes
