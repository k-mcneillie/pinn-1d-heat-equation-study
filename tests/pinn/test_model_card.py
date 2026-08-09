# test_card.py
# - tests for model card generation

# =============================
# Import Libraries
# =============================
from pathlib import Path

from pinn_study.pinn.card import ModelCard


# =============================
# Test Fixtures
# =============================
def create_model_card() -> ModelCard:
    """Create a model card for testing."""
    return ModelCard(
        name="MLP Heat Equation PINN",
        description=[
            "A multi-layer perceptron used to approximate",
            "the solution of the one-dimensional heat equation.",
        ],
        architecture="MLP",
        parameters={
            "input_dim": 2,
            "hidden_dim": 64,
            "hidden_layers": 3,
            "output_dim": 1,
        },
        intended_use=[
            "Approximation of the one-dimensional heat equation.",
        ],
        limitations=[
            "Validated only for the configured training domain.",
        ],
        training={
            "epochs": 1000,
            "learning_rate": 0.001,
        },
    )


# =============================
# Model Card Tests
# =============================
def test_model_card_initialises() -> None:
    """Model card stores the supplied metadata."""
    card = create_model_card()

    assert card.name == "MLP Heat Equation PINN"
    assert card.architecture == "MLP"
    assert card.parameters["input_dim"] == 2


def test_model_card_saves_markdown(tmp_path: Path) -> None:
    """Model card is saved as Markdown."""
    card = create_model_card()

    path = card.save(tmp_path)

    assert path == tmp_path / "model_card.md"
    assert path.is_file()


def test_model_card_contains_model_information(tmp_path: Path) -> None:
    """Saved model card contains model information."""
    card = create_model_card()

    path = card.save(tmp_path)
    content = path.read_text(encoding="utf-8")

    assert "# Model Card" in content
    assert "MLP Heat Equation PINN" in content
    assert "MLP" in content
    assert "`input_dim`: `2`" in content


def test_model_card_contains_description(tmp_path: Path) -> None:
    """Saved model card contains the model description."""
    card = create_model_card()

    path = card.save(tmp_path)
    content = path.read_text(encoding="utf-8")

    assert "A multi-layer perceptron used to approximate" in content
    assert "the solution of the one-dimensional heat equation." in content


def test_model_card_contains_training_information(tmp_path: Path) -> None:
    """Saved model card contains training metadata."""
    card = create_model_card()

    path = card.save(tmp_path)
    content = path.read_text(encoding="utf-8")

    assert "`epochs`: `1000`" in content
    assert "`learning_rate`: `0.001`" in content


def test_model_card_contains_intended_use_and_limitations(
    tmp_path: Path,
) -> None:
    """Saved model card contains intended use and limitations."""
    card = create_model_card()

    path = card.save(tmp_path)
    content = path.read_text(encoding="utf-8")

    assert "Approximation of the one-dimensional heat equation." in content
    assert "Validated only for the configured training domain." in content
