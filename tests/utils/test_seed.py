"""Tests for random seed utilities."""
import numpy as np
import torch
from pinn_heat_1d.utils.seed import set_seed

def test_set_seed_reproduces_random_values() -> None:
    """The same seed produces the same random values."""
    set_seed(42)
    python_values = np.random.rand(5)
    torch_values = torch.rand(5)
    set_seed(42)
    assert np.array_equal(python_values, np.random.rand(5))
    assert torch.equal(torch_values, torch.rand(5))