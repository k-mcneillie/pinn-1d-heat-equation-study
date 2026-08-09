"""Tests for dataset loading utilities."""

from pathlib import Path

import pytest
import torch

from pinn_study.data.generator import HeatEquationData
from pinn_study.data.utils import load_dataset, save_dataset


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


def test_save_dataset(tmp_path):
    """Test that a dataset is saved to the output directory."""
    dataset = {
        "interior": torch.tensor([[0.1, 0.2]]),
        "initial": torch.tensor([[0.3, 0.0]]),
        "boundary": torch.tensor([[0.0, 0.5]]),
    }

    path = save_dataset(dataset, tmp_path)

    assert path == tmp_path / "data.pt"
    assert path.exists()


def test_load_dataset_preserves_data(tmp_path):
    """Test that a saved dataset can be loaded without modification."""
    dataset = {
        "interior": torch.tensor([[0.1, 0.2]]),
        "initial": torch.tensor([[0.3, 0.0]]),
        "boundary": torch.tensor([[0.0, 0.5]]),
    }

    save_dataset(dataset, tmp_path)
    loaded = load_dataset(tmp_path / "data.pt")

    assert loaded.keys() == dataset.keys()

    for name in dataset:
        assert torch.equal(loaded[name], dataset[name])


def test_load_dataset_raises_for_missing_file(tmp_path):
    """Test that loading a missing dataset raises FileNotFoundError."""
    path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError):
        load_dataset(path)
