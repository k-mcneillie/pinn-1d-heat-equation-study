# session.py
# - Session management utilities.

# =============================
# Import Libraries
# =============================
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch

from .logger import configure_logger
from .seed import set_seed

# =============================
# Set Standard Variables
# =============================
_DEFAULT_OUTPUT_ROOT = Path("outputs")
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


# =============================
# Session Class
# =============================
class Session:
    """
    Manage outputs and logging for a single experiment run.
    Each session creates a unique, timestamped directory under the configured
    output root. The session logger writes to both the console and a log file
    within that directory.

    Args:
        name: Experiment name.
        output_root: Root directory in which session directories are created.
    """

    def __init__(
        self,
        name: str,
        *,
        seed: int = 42,
        device: str = "cpu",
        output_root: Path = _DEFAULT_OUTPUT_ROOT,
    ) -> None:

        timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
        self.name = name
        self.seed = seed
        self.device = torch.device(device)
        self.output_dir = output_root / f"{timestamp}_{name}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.logger = configure_logger(
            name=name,
            log_file=self.output_dir / "run.log",
        )

        set_seed(seed)

        self.logger.info(f"Experiment session: {self.name} started.")
        self.logger.info("Random Seed: %d", seed)
        self.logger.info(f"Device: {self.device}")
        self.logger.info("Output directory: %s", self.output_dir)

    def path(self, filename: str) -> Path:
        """
        Return a path within the session output directory.
        Args:
            filename: Name of the file within the output directory.
        Returns:
            The path to the requested file.
        """
        return self.output_dir / filename
