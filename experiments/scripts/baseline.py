
from pinn_heat_1d.analytical import analytical_solution
from pinn_heat_1d.data.utils import load_dataset
from pinn_heat_1d.utils.session import Session

from pathlib import Path

session = Session("test-analytical-method", seed=42, device="cpu")

logger = session.logger

dataset_path = Path("outputs/20260808_215130_test-dataset-gen/data.pt")

data = load_dataset(dataset_path)
logger.info("Dataset loaded...")

x = data.initial[:, 0]
t = data.initial[:, 1]

u_exact = analytical_solution(x, t)
