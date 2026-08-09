"""Tests for the generic PINN loss."""

import pytest
import torch

from pinn_study.pinn.loss.config import LossConfig
from pinn_study.pinn.loss.contract import Loss
from pinn_study.pinn.loss.pinn_loss import PINNLoss


class TestPINNLoss:
    """Tests for PINNLoss."""

    def test_implements_loss_contract(self) -> None:
        """PINNLoss satisfies the Loss contract."""
        config = LossConfig(weights={"pde": 1.0})
        loss: Loss = PINNLoss(config)

        result = loss({"pde": torch.tensor(1.0)})

        assert torch.equal(result.total, torch.tensor(1.0))
        assert torch.equal(
            result.components["pde"],
            torch.tensor(1.0),
        )

    def test_returns_weighted_sum(self) -> None:
        """The loss returns the weighted sum of constraint losses."""
        config = LossConfig(
            weights={
                "pde": 1.0,
                "initial": 2.0,
                "boundary": 0.5,
            }
        )
        loss = PINNLoss(config)

        result = loss(
            {
                "pde": torch.tensor(2.0),
                "initial": torch.tensor(3.0),
                "boundary": torch.tensor(4.0),
            }
        )

        assert torch.equal(result.total, torch.tensor(10.0))

    def test_returns_loss_components(self) -> None:
        """The loss exposes the individual constraint losses."""
        config = LossConfig(
            weights={
                "pde": 1.0,
                "boundary": 0.5,
            }
        )
        losses = {
            "pde": torch.tensor(2.0),
            "boundary": torch.tensor(4.0),
        }

        result = PINNLoss(config)(losses)

        assert result.components == losses

    def test_supports_arbitrary_constraint_names(self) -> None:
        """The loss supports constraints without fixed names."""
        config = LossConfig(
            weights={
                "constraint_a": 2.0,
                "constraint_b": 3.0,
            }
        )
        result = PINNLoss(config)(
            {
                "constraint_a": torch.tensor(1.0),
                "constraint_b": torch.tensor(2.0),
            }
        )

        assert torch.equal(result.total, torch.tensor(8.0))

    def test_rejects_missing_constraint(self) -> None:
        """A configured constraint missing from the losses raises ValueError."""
        config = LossConfig(
            weights={
                "pde": 1.0,
                "boundary": 1.0,
            }
        )

        with pytest.raises(
            ValueError,
            match="Missing losses for configured constraints",
        ):
            PINNLoss(config)({"pde": torch.tensor(1.0)})

    def test_ignores_unconfigured_constraint(self) -> None:
        """Constraints without configured weights do not contribute."""
        config = LossConfig(weights={"pde": 1.0})

        result = PINNLoss(config)(
            {
                "pde": torch.tensor(2.0),
                "additional": torch.tensor(100.0),
            }
        )

        assert torch.equal(result.total, torch.tensor(2.0))
        assert "additional" in result.components

    def test_preserves_gradient(self) -> None:
        """The total loss remains connected to its input tensors."""
        config = LossConfig(
            weights={
                "pde": 2.0,
                "boundary": 1.0,
            }
        )

        pde_loss = torch.tensor(2.0, requires_grad=True)
        boundary_loss = torch.tensor(3.0, requires_grad=True)

        result = PINNLoss(config)(
            {
                "pde": pde_loss,
                "boundary": boundary_loss,
            }
        )

        result.total.backward()

        assert pde_loss.grad is not None
        assert boundary_loss.grad is not None
        assert torch.equal(pde_loss.grad, torch.tensor(2.0))
        assert torch.equal(boundary_loss.grad, torch.tensor(1.0))
