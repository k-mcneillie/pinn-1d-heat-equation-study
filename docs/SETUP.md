Development Environment Setup

This document describes the local development environment for the PINN Heat Equation project.

1. Create the virtual environment

From the repository root:

python3 -m venv .venv

Activate the environment:

Linux / macOS / GitHub Codespaces

source .venv/bin/activate

Windows

.venv\Scripts\activate

Once activated, the terminal should indicate that .venv is being used.

⸻

2. Upgrade pip

Upgrade pip before installing project dependencies:

python -m pip install --upgrade pip

⸻

3. Install the project

Install the project and its development dependencies:

pip install -e ".[dev]"

The editable installation means changes to the source code are immediately available without reinstalling the package.

Development dependencies include the tools required for:

* testing
* linting
* formatting
* development

⸻

4. Verify the installation

Confirm that the package can be imported:

python -c "import pinn_study; print('pinn_study import successful')"

Run the test suite:

pytest

Run Ruff:

ruff check .

If formatting is configured in pyproject.toml, also run:

ruff format --check .

⸻

5. GitHub Codespaces

The project is designed to work with GitHub Codespaces.

When creating a Codespace, the repository is provided with a Python development environment. The project-specific virtual environment should still be created:

python3 -m venv .venv

Then:

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

VS Code should normally detect .venv automatically. If it does not, select it through:

Command Palette
→ Python: Select Interpreter
→ .venv

⸻

6. Environment variables

The project currently does not require secrets or external service credentials.

If environment variables are introduced in the future, document their names and example values in:

.env.example

Never commit a real .env file containing secrets.

The .env file should be included in .gitignore.

⸻

7. Recommended development workflow

Before making changes:

source .venv/bin/activate

After making changes:

pytest
ruff check .
ruff format --check .

Then review the changes:

git status
git diff

Commit only after the test and lint checks pass.

⸻

8. Repository structure

The project follows a source-layout structure:

pinn_study/
├── src/
│   └── pinn_study/
│       ├── data/
│       ├── pinn/
│       │   ├── architecture/
│       │   ├── loss/
│       │   └── utils/
│       ├── analytical/
│       └── utils/
│
├── tests/
│   ├── data/
│   ├── pinn/
│   ├── analytical/
│   └── utils/
│
├── outputs/
├── notebooks/
├── pyproject.toml
├── README.md
└── .gitignore

Generated datasets, logs, model outputs and experiment artefacts should be written through the project’s session/output system rather than committed directly to the source tree.

⸻

9. Reproducibility

Dataset generation uses an explicit random seed.

The active session is responsible for:

* the random seed
* the compute device
* the timestamped output directory
* logging

Generated datasets should be accompanied by their dataset card and relevant generation configuration.

⸻

10. Clean environment setup

To recreate the development environment from scratch:

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
ruff check .

A successful run of these commands indicates that the development environment is correctly configured.