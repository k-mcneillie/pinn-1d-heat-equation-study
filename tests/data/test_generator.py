import pytest
import torch
from pinn_heat_1d.data.generator import Generator

class TestGenerator:
    """Tests for Generator."""
    def test_generate_returns_expected_shapes(self) -> None:
        """Generated datasets have the expected shapes."""
        generator = Generator(
            n_interior=100,
            n_initial=10,
            n_boundary=10,
            seed=42,
        )
        data = generator.generate()
        assert data.interior.shape == (100, 2)
        assert data.initial.shape == (10, 2)
        assert data.boundary.shape == (10, 2)

    def test_interior_points_are_within_domain(self) -> None:
        """Interior points lie within the spatial and temporal domains."""
        generator = Generator(
            n_interior=1_000,
            n_initial=10,
            n_boundary=10,
            seed=42,
        )
        data = generator.generate()
        x = data.interior[:, 0]
        t = data.interior[:, 1]
        assert torch.all(x >= 0.0)
        assert torch.all(x <= 1.0)
        assert torch.all(t >= 0.0)
        assert torch.all(t <= 1.0)

    def test_initial_points_are_at_initial_time(self) -> None:
        """Initial-condition points all have t equal to t_min."""
        generator = Generator(
            n_interior=100,
            n_initial=100,
            n_boundary=10,
            seed=42,
        )
        data = generator.generate()
        assert torch.all(data.initial[:, 1] == 0.0)

    def test_boundary_points_are_on_spatial_boundaries(self) -> None:
        """Boundary points lie on either spatial boundary."""
        generator = Generator(
            n_interior=100,
            n_initial=10,
            n_boundary=100,
            seed=42,
        )
        data = generator.generate()
        x = data.boundary[:, 0]
        assert torch.all((x == 0.0) | (x == 1.0))

    def test_seed_produces_reproducible_data(self) -> None:
        """The same seed produces identical datasets."""
        generator_1 = Generator(
            n_interior=100,
            n_initial=10,
            n_boundary=10,
            seed=42,
        )
        generator_2 = Generator(
            n_interior=100,
            n_initial=10,
            n_boundary=10,
            seed=42,
        )
        data_1 = generator_1.generate()
        data_2 = generator_2.generate()
        assert torch.equal(data_1.interior, data_2.interior)
        assert torch.equal(data_1.initial, data_2.initial)
        assert torch.equal(data_1.boundary, data_2.boundary)
    @pytest.mark.parametrize(
        ("parameter", "value"),
        [
            ("n_interior", 0),
            ("n_initial", 0),
            ("n_boundary", 0),
        ],
    )

    def test_rejects_non_positive_point_counts(
        self,
        parameter: str,
        value: int,
    ) -> None:
        """Non-positive point counts raise ValueError."""
        kwargs = {
            "n_interior": 100,
            "n_initial": 10,
            "n_boundary": 10,
            parameter: value,
        }
        with pytest.raises(ValueError):
            Generator(**kwargs)
            
    def test_rejects_invalid_spatial_domain(self) -> None:
        """An invalid spatial domain raises ValueError."""
        with pytest.raises(ValueError):
            Generator(
                n_interior=100,
                n_initial=10,
                n_boundary=10,
                x_min=1.0,
                x_max=0.0,
            )
    def test_rejects_negative_initial_time(self) -> None:
        """A negative initial time raises ValueError."""
        with pytest.raises(ValueError):
            Generator(
                n_interior=100,
                n_initial=10,
                n_boundary=10,
                t_min=-1.0,
            )
    def test_rejects_invalid_temporal_domain(self) -> None:
        """An invalid temporal domain raises ValueError."""
        with pytest.raises(ValueError):
            Generator(
                n_interior=100,
                n_initial=10,
                n_boundary=10,
                t_min=1.0,
                t_max=1.0,
            )