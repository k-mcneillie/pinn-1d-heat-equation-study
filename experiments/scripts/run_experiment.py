from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import autograd

from pinn_study.analytical.heat_equation_1D import HeatEquation1D
from pinn_study.data.card import DatasetCard
from pinn_study.data.generator import HeatEquationGenerator, HeatEquationData
from pinn_study.data.utils import save_dataset
from pinn_study.experiment.config import ExperimentConfig, MLflowConfig
from pinn_study.pinn.architecture.mlp.config import MLPConfig
from pinn_study.pinn.architecture.mlp.model import MLPModel
from pinn_study.pinn.card import ModelCard
from pinn_study.pinn.loss.config import LossConfig
from pinn_study.pinn.loss.pinn_loss import PINNLoss
from pinn_study.pinn.training.checkpoint import TrainingCheckpointManager
from pinn_study.pinn.training.config import CheckpointConfig, TrainingConfig
from pinn_study.pinn.training.trainer import Trainer
from pinn_study.utils.session import Session
from pinn_study.vis.api.experiment import visualise_experiment

try:
    from pinn_study.experiment.mlflow import MLflowExperiment
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def build_dataset(session: Session, config: ExperimentConfig) -> HeatEquationData:
    generator = HeatEquationGenerator(
        n_interior=config.n_interior,
        n_initial=config.n_initial,
        n_boundary=config.n_boundary,
        seed=session.seed,
        device=session.device,
    )

    session.logger.info("Generating dataset...")
    dataset = generator.generate()

    dataset_dir = session.output_dir / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    card = DatasetCard(
        generator=generator.name,
        parameters=generator.parameters,
        seed=session.seed,
        device=str(session.device),
        description=(
            "Generated collocation points for the 1D heat equation.",
            "Includes interior, initial and boundary points.",
        ),
    )

    save_dataset(dataset, dataset_dir, card=card)
    session.logger.info("Dataset saved to %s", dataset_dir)

    return dataset


def create_model(config: ExperimentConfig, device: torch.device) -> MLPModel:
    model_config = MLPConfig(
        input_dim=2,
        output_dim=1,
        hidden_dims=config.hidden_dims,
        activation="tanh",
    )

    model = MLPModel(model_config)
    return model.to(device)


def create_training_step(
    model: MLPModel,
    dataset: HeatEquationData,
    device: torch.device,
    loss: PINNLoss,
    alpha: float,
) -> callable[[], Any]:
    pde = HeatEquation1D()
    interior = dataset.interior.to(device)
    initial = dataset.initial.to(device)
    boundary = dataset.boundary.to(device)

    def training_step() -> Any:
        interior_req = interior.detach().clone().requires_grad_(True)

        prediction_interior = model(interior_req)
        predicted_sum = prediction_interior.sum()

        gradients = autograd.grad(
            outputs=predicted_sum,
            inputs=interior_req,
            create_graph=True,
        )[0]

        derivative_x = gradients[:, 0:1]
        derivative_t = gradients[:, 1:2]

        second_derivative_x = autograd.grad(
            outputs=derivative_x.sum(),
            inputs=interior_req,
            create_graph=True,
        )[0][:, 0:1]

        residual = derivative_t - second_derivative_x
        pde_loss = residual.pow(2).mean()

        prediction_initial = model(initial)
        exact_initial = pde(
            initial[:, 0:1],
            initial[:, 1:2],
            alpha=alpha,
        )
        initial_loss = (prediction_initial - exact_initial).pow(2).mean()

        prediction_boundary = model(boundary)
        exact_boundary = torch.zeros_like(prediction_boundary)
        boundary_loss = (prediction_boundary - exact_boundary).pow(2).mean()

        result = loss(
            {
                "pde": pde_loss,
                "initial": initial_loss,
                "boundary": boundary_loss,
            }
        )

        return result

    return training_step


def run_training(
    session: Session,
    model: MLPModel,
    dataset: HeatEquationData,
    config: ExperimentConfig,
) -> Trainer:
    loss = PINNLoss(
        config=LossConfig(
            weights={
                "pde": 1.0,
                "initial": 1.0,
                "boundary": 1.0,
            }
        )
    )

    training_step = create_training_step(
        model=model,
        dataset=dataset,
        device=session.device,
        loss=loss,
        alpha=config.alpha,
    )

    checkpoint_manager = TrainingCheckpointManager(session.path("checkpoints"))
    training_config = TrainingConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        checkpoint=CheckpointConfig(enabled=True, interval=max(1, config.epochs // 5), save_best=True),
    )

    trainer = Trainer(
        model=model,
        training_step=training_step,
        config=training_config,
        logger=session.logger,
        device=session.device,
        loss=loss,
        checkpoint_manager=checkpoint_manager,
    )

    session.logger.info("Starting training for %d epochs...", config.epochs)
    trainer.train()
    session.logger.info("Training complete.")

    return trainer


def build_validation_data(
    model: MLPModel,
    alpha: float,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()

    n_x = 100
    n_t = 100
    x = torch.linspace(0.0, 1.0, steps=n_x, device=device)
    t = torch.linspace(0.0, 1.0, steps=n_t, device=device)
    X = x.unsqueeze(0).repeat(n_t, 1)
    T = t.unsqueeze(1).repeat(1, n_x)
    inputs = torch.stack([X.flatten(), T.flatten()], dim=-1).requires_grad_(True)

    with torch.enable_grad():
        prediction = model(inputs)
        gradients = autograd.grad(
            outputs=prediction.sum(),
            inputs=inputs,
            create_graph=True,
        )[0]

        derivative_x = gradients[:, 0:1]
        derivative_t = gradients[:, 1:2]

        second_derivative_x = autograd.grad(
            outputs=derivative_x.sum(),
            inputs=inputs,
            create_graph=False,
        )[0][:, 0:1]

    residual = derivative_t - second_derivative_x
    prediction = prediction.view(n_t, n_x)
    residual = residual.view(n_t, n_x)
    analytical = HeatEquation1D()(X, T, alpha=alpha)

    return {
        "x": x.cpu().numpy(),
        "t": t.cpu().numpy(),
        "X": X.cpu().numpy(),
        "T": T.cpu().numpy(),
        "prediction": prediction.detach().cpu().numpy(),
        "analytical": analytical.detach().cpu().numpy(),
        "residual": residual.detach().cpu().numpy(),
    }


def save_model(session: Session, model: MLPModel) -> Path:
    model_path = session.path("model.pt")
    torch.save(model.state_dict(), model_path)
    session.logger.info("Saved model state to %s", model_path)
    return model_path


def save_model_card(
    session: Session,
    architecture: str,
    parameters: dict[str, Any],
    training_metadata: dict[str, Any],
) -> None:
    card = ModelCard(
        name=session.name,
        description=(
            "A PINN model trained for the 1D heat equation.",
            "The architecture is a feedforward MLP trained with physics-informed losses.",
        ),
        architecture=architecture,
        parameters=parameters,
        intended_use=[
            "Demonstrate a physics-informed neural network for the 1D heat equation.",
            "Provide a baseline PINN training workflow.",
        ],
        limitations=[
            "Model is trained on a small collocation set and may not generalize broadly.",
            "Validation is performed at a single fixed final time.",
        ],
        training=training_metadata,
    )

    card_dir = session.path("model_card")
    card_dir.mkdir(parents=True, exist_ok=True)
    card.save(card_dir)
    session.logger.info("Saved model card to %s", card_dir)


def log_to_mlflow(
    session: Session,
    config: ExperimentConfig,
    trainer: Trainer,
    model_path: Path,
    dataset_dir: Path,
) -> None:
    if not config.mlflow:
        return

    if not MLFLOW_AVAILABLE:
        session.logger.warning("MLflow is not available; skipping MLflow logging.")
        return

    mlflow_config = MLflowConfig(
        tracking_uri=config.tracking_uri,
        experiment_name=config.experiment_name,
        run_name=config.run_name,
    )

    experiment = MLflowExperiment(mlflow_config)
    session.logger.info("Starting MLflow run %s", config.run_name)
    experiment.start_run()

    parameters = {
        "seed": session.seed,
        "device": str(session.device),
        "n_interior": config.n_interior,
        "n_initial": config.n_initial,
        "n_boundary": config.n_boundary,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "hidden_dims": config.hidden_dims,
        "alpha": config.alpha,
    }
    experiment.log_params(parameters)
    experiment.log_metrics({"final_loss": float(trainer.result.losses[-1])})

    experiment.log_artifact(model_path)
    experiment.log_artifacts(dataset_dir, artifact_path="dataset")
    experiment.log_artifacts(session.path("visualisations"), artifact_path="visualisations")
    experiment.end_run()
    session.logger.info("MLflow run complete.")


def main() -> None:
    config = ExperimentConfig.load()
    session = Session(
        config.experiment_name,
        seed=config.seed,
        device=config.device,
        output_root=config.output_root,
    )

    session.logger.info("Experiment configuration: %s", config.model_dump())

    dataset = build_dataset(session, config)
    model = create_model(config, session.device)
    trainer = run_training(session, model, dataset, config)

    model_path = save_model(session, model)
    save_model_card(
        session,
        architecture=model.__class__.__name__,
        parameters={
            "hidden_dims": config.hidden_dims,
            "activation": "tanh",
            "input_dim": 2,
            "output_dim": 1,
        },
        training_metadata={
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "final_loss": float(trainer.result.losses[-1]) if trainer.result.losses else 0.0,
        },
    )

    validation_data = build_validation_data(model, config.alpha, session.device)
    visualise_experiment(
        dataset=dataset,
        training_result=trainer.result,
        validation_data=validation_data,
        model=model,
        xai_data=None,
        output_dir=session.path("visualisations"),
    )

    log_to_mlflow(
        session,
        config,
        trainer,
        model_path,
        session.path("data"),
    )

    session.logger.info("Experiment complete. Outputs saved to %s", session.output_dir)


if __name__ == "__main__":
    main()
