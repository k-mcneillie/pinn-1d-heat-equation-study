# config.py
# - configuration for scientific visualisations


# =============================
# Import Libraries
# =============================
from pathlib import Path

from pydantic import BaseModel, Field


# =============================
# Visualisation Configuration
# =============================
class VisualisationConfig(BaseModel):
    """Configuration shared by scientific visualisations."""

    output_directory: Path = Path("figures")

    dpi: int = Field(
        default=300,
        gt=0,
    )

    figure_width: float = Field(
        default=8.0,
        gt=0.0,
    )

    figure_height: float = Field(
        default=5.0,
        gt=0.0,
    )

    file_format: str = "png"

    transparent: bool = False

    tight_layout: bool = True