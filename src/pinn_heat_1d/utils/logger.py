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
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        fmt=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    log_file.parent.mkdir(parents=True, exist_ok=True)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger