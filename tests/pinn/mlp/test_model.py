"""Tests for the MLP model."""

import pytest
import torch

from pinn_study.pinn.architecture.mlp.config import MLPConfig
from pinn_study.pinn.architecture.mlp.model import MLPModel


class TestMLPModel:
    """Tests for MLPModel."""

    def test_forward_returns_expected_shape(self) -> None:
        """The forward pass returns the expected output shape."""
        config = MLPConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16, 16],
        )
        model = MLPModel(config)
        inputs = torch.randn(10, 2)
        outputs = model(inputs)
        assert outputs.shape == (10, 1)

    def test_forward_preserves_batch_size(self) -> None:
        """The model preserves the input batch size."""
        config = MLPConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16],
        )
        model = MLPModel(config)
        inputs = torch.randn(32, 2)
        outputs = model(inputs)
        assert outputs.shape[0] == inputs.shape[0]

    @pytest.mark.parametrize(
        "activation",
        ["tanh", "relu", "sigmoid"],
    )
    def test_supported_activations(self, activation: str) -> None:
        """Supported activation functions can be used."""
        config = MLPConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16],
            activation=activation,
        )
        model = MLPModel(config)
        inputs = torch.randn(4, 2)
        outputs = model(inputs)
        assert outputs.shape == (4, 1)

    def test_rejects_unsupported_activation(self) -> None:
        """An unsupported activation raises ValueError."""
        config = MLPConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16],
            activation="unsupported",
        )
        with pytest.raises(ValueError, match="Unsupported activation"):
            MLPModel(config)

    def test_model_is_torch_module(self) -> None:
        """The MLP is a PyTorch module."""
        config = MLPConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16],
        )
        model = MLPModel(config)
        assert isinstance(model, torch.nn.Module)

    def test_model_has_trainable_parameters(self) -> None:
        """The MLP contains trainable parameters."""
        config = MLPConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16, 16],
        )
        model = MLPModel(config)
        parameters = list(model.parameters())
        assert parameters
        assert all(parameter.requires_grad for parameter in parameters)

    def test_single_hidden_layer(self) -> None:
        """The model supports a single hidden layer."""
        config = MLPConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=[16],
        )
        model = MLPModel(config)
        outputs = model(torch.randn(8, 2))
        assert outputs.shape == (8, 1)
