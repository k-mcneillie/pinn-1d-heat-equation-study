"""Tests for the MLP configuration."""

import pytest
from pydantic import ValidationError

from pinn_study.pinn.architecture.mlp.config import MLPConfig


class TestMLPConfig:
    """Tests for MLPConfig."""

    def test_valid_configuration(self) -> None:
        """A valid configuration is accepted."""
        config = MLPConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            activation="tanh",
        )
        assert config.input_dim == 2
        assert config.output_dim == 1
        assert config.hidden_dims == [32, 32]
        assert config.activation == "tanh"

    @pytest.mark.parametrize(
        "field",
        ["input_dim", "output_dim"],
    )
    def test_rejects_non_positive_dimensions(self, field: str) -> None:
        """Non-positive input and output dimensions are rejected."""
        kwargs = {
            "input_dim": 2,
            "output_dim": 1,
            "hidden_dims": [32, 32],
            field: 0,
        }
        with pytest.raises(ValidationError):
            MLPConfig(**kwargs)

    def test_rejects_non_positive_hidden_dimension(self) -> None:
        """Non-positive hidden dimensions are rejected."""
        with pytest.raises(ValidationError):
            MLPConfig(
                input_dim=2,
                output_dim=1,
                hidden_dims=[32, 0],
            )

    def test_rejects_empty_hidden_dimensions(self) -> None:
        """An empty hidden-layer configuration is rejected."""
        with pytest.raises(ValidationError):
            MLPConfig(
                input_dim=2,
                output_dim=1,
                hidden_dims=[],
            )

    def test_default_activation_is_tanh(self) -> None:
        """The default activation is tanh."""
        config = MLPConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32],
        )
        assert config.activation == "tanh"
