"""Tests for dataset loading utilities."""

from pathlib import Path

import pytest
import torch

from pinn_heat_1d.data.generator import HeatEquationData
from pinn_heat_1d.data.utils import load_dataset


def test_load_dataset(tmp_path: Path) -> None:
    """A saved dataset can be loaded correctly."""
    data = HeatEquationData(
        interior=torch.rand(100, 2),
        initial=torch.rand(10, 2),
        boundary=torch.rand(10, 2),
    )
    path = tmp_path / "data.pt"
    torch.save(data, path)
    loaded = load_dataset(path)
    assert torch.equal(loaded.interior, data.interior)
    assert torch.equal(loaded.initial, data.initial)
    assert torch.equal(loaded.boundary, data.boundary)


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    """Loading a missing dataset raises FileNotFoundError."""
    path = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        load_dataset(path)
