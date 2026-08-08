"""Tests for logging utilities."""

from pathlib import Path

from pinn_study.utils.logger import configure_logger


def test_configure_logger_creates_log_file(tmp_path: Path) -> None:
    """Configured logger writes messages to the specified log file."""
    log_file = tmp_path / "test.log"
    logger = configure_logger(
        name="test_logger",
        log_file=log_file,
    )
    logger.info("Test message.")
    assert log_file.exists()
    assert "Test message." in log_file.read_text()
