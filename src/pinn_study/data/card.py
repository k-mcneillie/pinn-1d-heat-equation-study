# card.py
# - construct and save dataset card when new data is generated.

# =============================
# Import Libraries
# =============================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DatasetCard:
    """Create documentation for a generated dataset."""

    def __init__(
        self,
        *,
        generator: str,
        parameters: dict[str, Any],
        seed: int,
        device: str,
        description: str | tuple[str] | list[str],
    ) -> None:
        """
        Initialise a dataset card.

        Args:
            generator: Name of the generator used to create the dataset.
            parameters: Parameters used during dataset generation.
            seed: Random seed used for generation.
            device: Device on which generation was performed.
            description: Short description of the dataset.
        """
        self.generator = generator
        self.parameters = parameters
        self.seed = seed
        self.device = device
        self.description = description

    def save(self, output_dir: Path) -> None:
        """
        Save the dataset card and generation configuration.

        Args:
            output_dir: Directory in which the files should be saved.
        """
        self._save_markdown(output_dir / "dataset_card.md")
        self._save_config(output_dir / "dataset_config.json")

    def _save_markdown(self, path: Path) -> None:
        """
        Save the human-readable dataset card.

        Args:
            path: Path to the dataset card.
        """
        if isinstance(self.description, str):
            self.description = [self.description]

        lines = [
            "# Dataset Card",
            "",
            "## Overview",
            "",
            *self.description,
            "",
            "## Generation",
            "",
            f"- Generator: `{self.generator}`",
            f"- Seed: `{self.seed}`",
            f"- Device: `{self.device}`",
            "",
            "## Parameters",
            "",
        ]

        for name, value in self.parameters.items():
            lines.append(f"- `{name}`: `{value}`")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _save_config(self, path: Path) -> None:
        """
        Save the machine-readable generation configuration.

        Args:
            path: Path to the configuration file.
        """
        config = {
            "generator": self.generator,
            "seed": self.seed,
            "device": self.device,
            "parameters": self.parameters,
        }
        path.write_text(
            json.dumps(config, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
