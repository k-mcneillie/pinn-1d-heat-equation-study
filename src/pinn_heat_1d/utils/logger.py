# logger.py
# - Initialise logging for project.

# =============================
# Import Libraries
# =============================
from __future__ import annotations

import logging
from pathlib import Path

# =============================
# Formatting
# =============================
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =============================
# Configure Logger
# =============================
def configure_logger(
    name: str,
    log_file: Path,
    *,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger with console and file handlers.
    The logger writes messages to both standard output and the specified
    log file. If the logger already has handlers, no additional handlers
    are added.

    Args:
        name: Name of the logger.
        log_file: Path to the log file.
        level: Logging level for the logger and its handlers.

    Returns:
        A configured ``logging.Logger`` instance.

    Raises:
        OSError: If the parent directory of ``log_file`` cannot be created.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
