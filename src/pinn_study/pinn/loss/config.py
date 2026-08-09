# config.py
# - configuration for generalised weighted loss

# =============================
# Import Libraries
# =============================
from pydantic import BaseModel, Field, field_validator


class LossConfig(BaseModel):
    """Configuration for weighting PINN constraint losses.
    Attributes:
        weights: Mapping of constraint names to their loss weights.
    """

    weights: dict[str, float] = Field(min_length=1)

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        """
        Validate that loss weights are non-negative.

        Args:
            value: Mapping of constraint names to weights.

        Returns:
            Validated loss weights.

        Raises:
            ValueError: If a constraint name is empty or a weight is negative.
        """
        if any(not name.strip() for name in value):
            raise ValueError("Constraint names must not be empty.")
        if any(weight < 0.0 for weight in value.values()):
            raise ValueError("Loss weights must be non-negative.")
        return value
