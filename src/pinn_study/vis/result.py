# result.py
# - result of a visualisation run


# =============================
# Import Libraries
# =============================
from dataclasses import dataclass, field
from pathlib import Path


# =============================
# Visualisation Result
# =============================
@dataclass
class VisualizationResult:
    """Record generated visualisation artefacts."""

    output_dir: Path
    figures: list[Path] = field(default_factory=list)
