# tests/pinn/loss/test_result.py
# - tests for PINN loss results


# =============================
# Import Libraries
# =============================
import torch

from pinn_study.pinn.loss.result import PINNLossResult


# =============================
# PINN Loss Result
# =============================
class TestPINNLossResult:
    """Tests for PINNLossResult."""

    def test_stores_total_loss(self) -> None:
        """PINNLossResult stores the total loss."""
        total = torch.tensor(10.0)

        result = PINNLossResult(
            total=total,
            components={},
        )

        assert torch.equal(result.total, total)

    def test_stores_loss_components(self) -> None:
        """PINNLossResult stores individual loss components."""
        components = {
            "pde": torch.tensor(2.0),
            "boundary": torch.tensor(3.0),
            "initial": torch.tensor(4.0),
        }

        result = PINNLossResult(
            total=torch.tensor(9.0),
            components=components,
        )

        assert result.components == components

    def test_preserves_component_tensors(self) -> None:
        """Loss components remain connected to their tensors."""
        pde_loss = torch.tensor(
            2.0,
            requires_grad=True,
        )

        result = PINNLossResult(
            total=pde_loss,
            components={
                "pde": pde_loss,
            },
        )

        result.total.backward()

        assert pde_loss.grad is not None
        assert torch.equal(
            pde_loss.grad,
            torch.tensor(1.0),
        )

    def test_supports_empty_components(self) -> None:
        """PINNLossResult permits an empty component mapping."""
        result = PINNLossResult(
            total=torch.tensor(1.0),
            components={},
        )

        assert result.components == {}
