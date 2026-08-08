from pathlib import Path

from pinn_study.analytical.heat_equation_1D import HeatEquation1D
from pinn_study.data.utils import load_dataset
from pinn_study.utils.session import Session

session = Session("test-analytical-method", seed=42, device="cpu")

logger = session.logger

dataset_path = Path("outputs/20260808_215130_test-dataset-gen/data.pt")

data = load_dataset(dataset_path)
logger.info("Dataset loaded...")

x = data.initial[:, 0]
t = data.initial[:, 1]

u_exact = HeatEquation1D.compute(x, t)
