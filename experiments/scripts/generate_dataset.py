import torch

from pinn_study.data.generator import Generator
from pinn_study.utils.session import Session

session = Session("test-dataset-gen", seed=42, device="cpu")

logger = session.logger

logger.info("Starting dataset generation...")

args = {"n_interior": 100, "n_initial": 10, "n_boundary": 10}

kwargs = {
    "seed": session.seed,
    "device": session.device,
}

generator = Generator(**args, **kwargs)

data = generator.generate()
logger.info("Dataset generated...")

torch.save(data, session.path("data.pt"))
logger.info(f"Dataset saved to: {session.path('data.pt')}...")

logger.info("Dataset generation complete.")
