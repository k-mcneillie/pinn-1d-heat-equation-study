# PINN-Study — Physics-Informed Neural Networks Programme

An independent, self-directed research and development programme building from first principles of Physics-Informed Neural Networks (PINNs) to a research contribution: applying learned surrogate modelling to the author's own dissertation research on topological superconductivity, benchmarked against a known analytical phase diagram.

## What are Physics-Informed Neural Networks?

Standard neural networks learn from data. Given enough examples of inputs and outputs, they approximate the underlying function that connects them. This works well when data is abundant but poorly when it is scarce, expensive, or impossible to collect.

PINNs take a different approach. Rather than learning purely from data, they embed the governing equations of a physical system directly into the training process as constraints. The network is penalised not just for mismatching observed data but for violating the physics. The result is a model that respects physical laws by construction rather than approximation.

## Objectives

- Develop working understanding of PINN mechanics: loss-term balancing, collocation sampling, and architecture choices appropriate to smooth, oscillatory, and coupled solutions.
- Establish a verified baseline (1D heat equation) anchored against a closed-form analytical solution.
