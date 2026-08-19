from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class MLflowConfig(BaseModel):
    """Configuration for MLflow experiment tracking."""

    tracking_uri: str = Field(default="file:./mlruns")
    experiment_name: str = Field(default="default")
    run_name: str | None = None
    artifact_location: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("experiment_name")
    @classmethod
    def _validate_experiment_name(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("experiment_name must not be empty.")
        return value

    @field_validator("tracking_uri")
    @classmethod
    def _validate_tracking_uri(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("tracking_uri must not be empty.")
        return value


class ExperimentConfig(BaseModel):
    """Configuration for an end-to-end experiment."""

    experiment_name: str = Field(default="pinn-heat-1d")
    run_name: str | None = Field(default="run-1")
    seed: int = Field(default=42, ge=0)
    device: str = Field(default="cpu")
    n_interior: int = Field(default=1000, gt=0)
    n_initial: int = Field(default=100, gt=0)
    n_boundary: int = Field(default=100, gt=0)
    epochs: int = Field(default=200, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    hidden_dims: list[int] = Field(default_factory=lambda: [32, 32], min_length=1)
    mlflow: bool = Field(default=False)
    tracking_uri: str = Field(default="file:./mlruns")
    alpha: float = Field(default=1.0, gt=0.0)
    output_root: Path = Field(default_factory=lambda: Path("outputs"))

    @field_validator("experiment_name")
    @classmethod
    def _validate_experiment_name(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("experiment_name must not be empty.")
        return value

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        valid_devices = {"cpu", "cuda", "mps"}
        if value not in valid_devices:
            raise ValueError(f"device must be one of {sorted(valid_devices)}")
        return value

    @field_validator("tracking_uri")
    @classmethod
    def _validate_tracking_uri(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("tracking_uri must not be empty.")
        return value

    @classmethod
    def load(cls, path: Path | str | None = None) -> ExperimentConfig:
        if path is None:
            path = Path("experiments/config.json")

        path = Path(path)
        if path.exists():
            return cls.model_validate_json(path.read_text())

        return cls()
