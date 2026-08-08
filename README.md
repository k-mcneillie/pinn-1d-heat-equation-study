# Physics-Informed Neural Networks — 1D Heat Equation

## What are Physics-Informed Neural Networks?

Standard neural networks learn from data. Given enough examples of inputs and outputs they approximate the underlying function that connects them. This works well when data is abundant but poorly when it is scarce, expensive or impossible to collect.

Physics-Informed Neural Networks take a different approach. Rather than learning purely from data, they embed the governing equations of a physical system directly into the training process as constraints. The network is penalised not just for mismatching observed data but for violating the physics. The result is a model that respects physical laws by construction rather than approximation.

## Why the heat equation?

The 1D heat equation is the simplest meaningful partial differential equation with a known analytical solution. It describes how heat distributes itself across a rod over time — governed by a second order spatial derivative and a first order time derivative. Every term is physically interpretable and the analytical solution provides an exact verification target.

Starting here is deliberate. The goal is not to solve a hard problem but to build genuine intuition for how physics constraints shape the loss landscape before approaching more complex systems. Understanding where the PINN succeeds and where it struggles on a tractable problem is essential groundwork.

## Implementation

- PyTorch for automatic differentiation and network training
- Fully connected network with sinusoidal activations following the SIREN architecture, which handles oscillatory solutions better than ReLU
- Loss function combining three terms: PDE residual enforcing the heat equation across the domain, boundary conditions fixing the temperature at the rod ends, and initial conditions fixing the temperature distribution at time zero
- Training on randomly sampled collocation points across the spatial and temporal domain rather than a fixed grid
- Comparison of the learned solution against the analytical solution across the full domain

## What this establishes

A verified, working PINN implementation. Understanding of how each loss term contributes to the solution. Intuition for where physics constraints alone are sufficient to recover correct behaviour and where additional data points are needed. The foundation for everything that follows.
