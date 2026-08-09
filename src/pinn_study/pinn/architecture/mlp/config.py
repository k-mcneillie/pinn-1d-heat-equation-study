# config.py
# - configuration for multi-layer perceptron

# =============================
# Import Libraries
# =============================
from pydantic import BaseModel, Field, field_validator


# =============================
# Multi-Layer Perceptron Config
# =============================
# Configs are used too...
class MLPConfig(BaseModel):
    """Configuration for a multilayer perceptron.

    Attributes:
        input_dim: Number of input features.
        output_dim: Number of output features.
        hidden_dims: Number of neurons in each hidden layer.
        activation: Activation function used between hidden layers.
    """

    input_dim: int = Field(gt=0)
    output_dim: int = Field(gt=0)
    hidden_dims: list[int] = Field(min_length=1)
    activation: str = "tanh"

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, value: list[int]) -> list[int]:
        """Validate hidden-layer dimensions.

        Args:
            value: Hidden-layer dimensions.

        Returns:
            Validated hidden-layer dimensions.

        Raises:
            ValueError: If a hidden dimension is not positive.
        """
        if any(dim <= 0 for dim in value):
            raise ValueError("All hidden dimensions must be positive.")
        return value
