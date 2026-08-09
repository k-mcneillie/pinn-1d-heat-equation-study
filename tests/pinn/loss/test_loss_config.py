"""Tests for PINN loss configuration."""

import pytest
from pydantic import ValidationError

from pinn_study.pinn.loss.config import LossConfig


class TestLossConfig:
    """Tests for LossConfig."""

    def test_accepts_valid_weights(self) -> None:
        """Valid constraint weights are accepted."""
        config = LossConfig(
            weights={
                "pde": 1.0,
                "initial": 1.0,
                "boundary": 0.5,
            }
        )
        assert config.weights == {
            "pde": 1.0,
            "initial": 1.0,
            "boundary": 0.5,
        }

    def test_accepts_zero_weight(self) -> None:
        """A zero loss weight is accepted."""
        config = LossConfig(
            weights={
                "pde": 1.0,
                "boundary": 0.0,
            }
        )
        assert config.weights["boundary"] == 0.0

    def test_rejects_negative_weight(self) -> None:
        """Negative loss weights are rejected."""
        with pytest.raises(ValidationError):
            LossConfig(
                weights={
                    "pde": -1.0,
                }
            )

    def test_rejects_empty_weights(self) -> None:
        """An empty weight mapping is rejected."""
        with pytest.raises(ValidationError):
            LossConfig(weights={})

    def test_rejects_empty_constraint_name(self) -> None:
        """Empty constraint names are rejected."""
        with pytest.raises(ValueError, match="Constraint names"):
            LossConfig(
                weights={
                    "": 1.0,
                }
            )
