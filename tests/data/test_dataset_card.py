"""Tests for dataset card utilities."""

import json

from pinn_study.data.card import DatasetCard


def test_dataset_card_saves_markdown(tmp_path):
    """Test that a dataset card saves a Markdown file."""
    card = DatasetCard(
        generator="Generator",
        parameters={
            "n_interior": 100,
            "n_initial": 10,
            "n_boundary": 10,
        },
        seed=42,
        device="cpu",
        description="Test dataset.",
    )

    card.save(tmp_path)

    assert (tmp_path / "dataset_card.md").exists()


def test_dataset_card_saves_config(tmp_path):
    """Test that a dataset card saves a JSON configuration."""
    card = DatasetCard(
        generator="Generator",
        parameters={
            "n_interior": 100,
            "n_initial": 10,
            "n_boundary": 10,
        },
        seed=42,
        device="cpu",
        description="Test dataset.",
    )

    card.save(tmp_path)

    assert (tmp_path / "dataset_config.json").exists()


def test_dataset_card_contains_metadata(tmp_path):
    """Test that the Markdown card contains generation metadata."""
    card = DatasetCard(
        generator="Generator",
        parameters={
            "n_interior": 100,
            "n_initial": 10,
            "n_boundary": 10,
        },
        seed=42,
        device="cpu",
        description="Test dataset.",
    )

    card.save(tmp_path)

    content = (tmp_path / "dataset_card.md").read_text(encoding="utf-8")

    assert "# Dataset Card" in content
    assert "Test dataset." in content
    assert "Generator" in content
    assert "42" in content
    assert "cpu" in content
    assert "n_interior" in content
    assert "100" in content


def test_dataset_card_config_contains_metadata(tmp_path):
    """Test that the JSON configuration contains generation metadata."""
    card = DatasetCard(
        generator="Generator",
        parameters={
            "n_interior": 100,
            "n_initial": 10,
            "n_boundary": 10,
        },
        seed=42,
        device="cpu",
        description="Test dataset.",
    )

    card.save(tmp_path)

    config_path = tmp_path / "dataset_config.json"

    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    assert config["generator"] == "Generator"
    assert config["seed"] == 42
    assert config["device"] == "cpu"
    assert config["parameters"]["n_interior"] == 100
    assert config["parameters"]["n_initial"] == 10
    assert config["parameters"]["n_boundary"] == 10
