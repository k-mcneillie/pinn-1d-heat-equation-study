from pinn_study.experiment.result import MLflowResult


def test_mlflow_result_fields() -> None:
    result = MLflowResult(
        experiment_name="experiment",
        run_id="123456",
        tracking_uri="file:./mlruns",
        artifact_uri="file:./mlruns/0",
        status="FINISHED",
        params={"lr": "0.001"},
        metrics={"loss": 0.123},
    )

    assert result.experiment_name == "experiment"
    assert result.run_id == "123456"
    assert result.tracking_uri == "file:./mlruns"
    assert result.artifact_uri.endswith("/0")
    assert result.status == "FINISHED"
    assert result.params["lr"] == "0.001"
    assert result.metrics["loss"] == 0.123
