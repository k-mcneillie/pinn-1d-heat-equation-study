from __future__ import annotations

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
