# model.py
# - model definition for multi-layer perceptron

# =============================
# Import Libraries
# =============================
import torch.nn as nn
from torch import Tensor

from .config import MLPConfig


# =============================
# Multi-Layer Perceptron Model
# =============================
# Mutli-Layer Perceptrons are a class of neural network built up from
# sequential linear layers.
class MLPModel(nn.Module):
    """A configurable multilayer perceptron."""

    def __init__(self, config: MLPConfig) -> None:
        """Initialise the multilayer perceptron.

        Args:
            config: Configuration defining the network architecture.

        Raises:
            ValueError: If the configured activation is unsupported.
        """
        super().__init__()

        layers: list[nn.Module] = []
        dimensions = [
            config.input_dim,
            *config.hidden_dims,
            config.output_dim,
        ]

        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))

            if index < len(dimensions) - 2:
                layers.append(self._get_activation(config.activation))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Perform a forward pass through the network.

        Args:
            inputs: Input Tensor of shape (batch_size, input_dim).

        Returns:
            Model predictions of shape (batch_size, output_dim).
        """
        return self.network(inputs)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Return the activation function specified by name.

        Args:
            name: Activation function name.

        Returns:
            The corresponding PyTorch activation module.

        Raises:
            ValueError: If the activation is unsupported.
        """
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
        }

        try:
            return activations[name.lower()]()
        except KeyError as exc:
            raise ValueError(f"Unsupported activation function: {name}") from exc
