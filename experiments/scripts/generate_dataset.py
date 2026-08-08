from pinn_heat_1d.utils.session import Session
from pinn_heat_1d.data.generator import Generator

import torch

session = Session("test-dataset-gen", seed = 42, device = "cpu")

logger = session.logger

logger.info("Starting dataset generation...")

args = {
    "n_interior": 100,
    "n_initial": 10,
    "n_boundary": 10
}

kwargs = {
    "seed": session.seed,
    "device": session.device,
}

generator = Generator(**args, **kwargs)

data = generator.generate()
logger.info("Dataset generated...")

torch.save(data, session.path("data.pt"))
logger.info(f"Dataset saved to: {session.path("data.pt")}...")

logger.info(f"Dataset generation complete.")