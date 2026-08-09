# sampling.py
# - sampling diagnostics


# =============================
# Import Libraries
# =============================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================
# Sampling Density
# =============================
def plot_sampling_density(
    x: np.ndarray,
    output_path: Path,
    *,
    bins: int = 30,
) -> Path:
    """Plot the spatial density of sampled points."""

    values = np.sort(np.asarray(x).reshape(-1))

    figure, axis = plt.subplots()
    axis.hist(values, bins=bins)
    axis.set_xlabel("x")
    axis.set_ylabel("Sample count")
    axis.set_title("Sampling Density")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Sampling Gaps
# =============================
def plot_sampling_gaps(
    x: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot distances between consecutive samples."""

    values = np.sort(np.asarray(x).reshape(-1))
    gaps = np.diff(values)

    figure, axis = plt.subplots()
    axis.hist(gaps, bins=30)
    axis.set_xlabel("Sampling gap")
    axis.set_ylabel("Count")
    axis.set_title("Sampling Gap Distribution")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path


# =============================
# Sampling Coverage
# =============================
def plot_sampling_coverage(
    x: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot cumulative spatial coverage of samples."""

    values = np.sort(np.asarray(x).reshape(-1))

    cumulative = np.arange(1, len(values) + 1) / len(values)

    figure, axis = plt.subplots()
    axis.plot(values, cumulative)
    axis.set_xlabel("x")
    axis.set_ylabel("Cumulative sample fraction")
    axis.set_title("Sampling Coverage")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    return output_path