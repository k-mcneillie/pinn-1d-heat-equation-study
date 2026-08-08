"""Tests for experiment session utilities."""

from pathlib import Path

import torch

from pinn_heat_1d.utils.session import Session


def test_session_creates_output_directory(tmp_path: Path) -> None:
    """A session creates its timestamped output directory."""
    session = Session(
        "test",
        seed=42,
        device="cpu",
        output_root=tmp_path,
    )
    assert session.output_dir.exists()
    assert session.output_dir.is_dir()


def test_session_stores_seed_and_device(tmp_path: Path) -> None:
    """A session stores the configured seed and device."""
    session = Session(
        "test",
        seed=123,
        device="cpu",
        output_root=tmp_path,
    )
    assert session.seed == 123
    assert session.device == torch.device("cpu")


def test_session_creates_logger(tmp_path: Path) -> None:
    """A session configures a logger with a run log."""
    session = Session(
        "test",
        output_root=tmp_path,
    )
    session.logger.info("Test message.")
    log_file = session.output_dir / "run.log"
    assert log_file.exists()
    assert "Test message." in log_file.read_text()


def test_session_path_returns_output_path(tmp_path: Path) -> None:
    """Session paths point inside the session output directory."""
    session = Session(
        "test",
        output_root=tmp_path,
    )
    path = session.path("model.pt")
    assert path == session.output_dir / "model.pt"
