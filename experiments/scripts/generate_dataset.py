from pinn_study.data.card import DatasetCard
from pinn_study.data.generator import HeatEquationGenerator
from pinn_study.data.utils import save_dataset
from pinn_study.utils.session import Session

session = Session("test-dataset-gen", seed=42, device="cpu")

logger = session.logger

logger.info("Starting dataset generation...")

args = {"n_interior": 100, "n_initial": 10, "n_boundary": 10}

kwargs = {
    "seed": session.seed,
    "device": session.device,
}

generator = HeatEquationGenerator(**args, **kwargs)

dataset = generator.generate()
logger.info("Dataset generated...")

card = DatasetCard(
    generator=generator.name,
    parameters=generator.parameters,
    seed=session.seed,
    device=session.device,
    description=(
        "- Collocation points generated for the 1D Heat Equation.",
        "- PINN Study",
        "- Test",
    ),
)
logger.info("Dataset card created...")

save_dataset(
    dataset,
    session.output_dir,
    card=card,
)
logger.info(f"Dataset saved to: {session.path('data.pt')}...")

logger.info("Dataset generation complete.")
