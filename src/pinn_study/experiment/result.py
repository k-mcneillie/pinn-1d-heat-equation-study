from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MLflowResult:
    """Structured information captured from an MLflow run."""

    experiment_name: str
    run_id: str
    tracking_uri: str
    artifact_uri: str
    status: str
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
