# PINN-Study — Physics-Informed Neural Networks Programme

An independent, self-directed research and development programme building from first principles of Physics-Informed Neural Networks (PINNs) to a research contribution: applying learned surrogate modelling to the author's own dissertation research on topological superconductivity, benchmarked against a known analytical phase diagram.

## What are Physics-Informed Neural Networks?

Standard neural networks learn from data. Given enough examples of inputs and outputs, they approximate the underlying function that connects them. This works well when data is abundant but poorly when it is scarce, expensive, or impossible to collect.

PINNs take a different approach. Rather than learning purely from data, they embed the governing equations of a physical system directly into the training process as constraints. The network is penalised not just for mismatching observed data but for violating the physics. The result is a model that respects physical laws by construction rather than approximation.

## Objectives

- Develop working understanding of PINN mechanics: loss-term balancing, collocation sampling, and architecture choices appropriate to smooth, oscillatory, and coupled solutions.
- Establish a verified baseline (1D heat equation) anchored against a closed-form analytical solution.
- Extend to a complex-valued, coupled PDE system (time-dependent Schrödinger equation) to build the eigenvalue-problem intuition the final phase requires.
- Apply the accumulated methodology to a genuine research contribution: a data-driven surrogate for topological phase prediction in a magnetic-impurity/superconductor system, grounded in and validated against the author's own dissertation and its cited analytical results.
- Build and apply AI assurance and governance framework; tracking development through model cards, datset cards and a clean, auditable repository.

## Structure

The programme runs as three sequential phases, each with its own sub-phases, notebooks, and definition of done.

| Phase | Focus | Stage | Definition of Done (Minimum) |
|---|---|---|---|
| 1 | 1D heat equation — PINN foundations, three-term loss construction, analytical validation | Documentation | Verified PINN vs. analytical solution notebook |
| 2 | Time-dependent Schrödinger equation — complex-valued, coupled PDE, oscillatory solutions, probability-conservation check | Not started | Full train + validation notebook & discussion |
| 3 | Topological phase surrogate — ML applied to a dissertation-adjacent BdG eigenvalue problem | Not started | Validated baseline models + analytical phase-diagram comparison + interpretability discussion |

## Key Architecture Decisions

**Core**
- Heat equation PINN: PyTorch, SIREN (sinusoidal) activations, three-term loss (PDE residual + boundary condition + initial condition), random collocation sampling.
- Schrödinger PINN: dual real-valued output heads $u(x,t)$, $v(x,t)$ representing $\psi = u + iv$ (avoids complex-tensor autodiff entirely); coupled real/imaginary residual loss; added probability-conservation penalty term with no analogue in Phase 1.
- Topological surrogate: a lightweight Bogoliubov–de Gennes (BdG) Hamiltonian data generator (`scipy.sparse.linalg.eigsh`) feeding a gradient-boosted tree baseline (XGBoost/LightGBM) for classification (trivial vs. non-trivial phase) and regression (critical scattering strength $V_m^*$, near-zero-state energy splitting).

**Exploratory**
- Active-learning resampling loop for sample-efficient dataset construction in Phase 3, targeting the same computational-cost constraint the dissertation itself documents.
- Operator-learning (CNN / Fourier Neural Operator) image surrogate predicting particle-hole probability density maps directly, bypassing diagonalisation entirely.

**Other**
- Collocation sampling: random sampling across the domain, methodology held consistent from Phase 1 through Phase 2 to keep the two directly comparable.
- Validation philosophy: every phase is anchored against a closed-form or independently-derived ground truth before being considered complete — the analytical solution in Phases 1 and 2, and the analytical topological phase diagram (Carroll & Braunecker, 2021) in Phase 3.

## Research Contribution

Phase 3 compares a learned surrogate against the direct numerical diagonalisation used in the author's dissertation, across two tracks:

**3a — Data & Baseline:** BdG data generator validated against the dissertation's own known transition points → Latin-Hypercube-sampled dataset across scattering strength, spin texture, chain geometry, and edge separation → gradient-boosted-tree classifier (trivial/non-trivial) and regressors ($V_m^*$, energy splitting).

**3b — Validation & Interpretability:** comparison of the model's predicted phase boundary against the closed-form analytical phase diagram (Carroll & Braunecker, 2021) as an independent correctness check, distinct from validating against the model's own generated data; SHAP-based feature-importance analysis assessed against the dissertation's qualitative conclusions — specifically, whether the model independently recovers the dominance of spin texture and $V_m$ over raw lattice size as drivers of Majorana bound state stability.

## What this establishes

A working, incrementally verified PINN and surrogate-modelling toolchain, anchored at every stage against either a closed-form analytical solution or an independently derived analytical result, culminating in an original empirical question answered on top of the author's own dissertation: whether a cheap learned model can reproduce, and independently corroborate, the topological phase behaviour that dissertation characterised through direct numerical diagonalisation.

## References

- McNeillie, K. (2023). Numerical investigation of topological states near magnetic structures on a superconductor. (Author's dissertation, supervised by Dr. Bernd Braunecker.)
- Carroll, C.J.F. & Braunecker, B. (2021). Subgap states at ferromagnetic and spiral-ordered magnetic chains in two-dimensional superconductors. I. Continuum description. *Phys. Rev. B*, 104, 245133.
- Carroll, C.J.F. & Braunecker, B. (2021). Subgap states at ferromagnetic and spiral-ordered magnetic chains in two-dimensional superconductors. II. Topological classification. *Phys. Rev. B*, 104, 245134.
- Braunecker, B., Japaridze, G.I., Klinovaja, J. & Loss, D. (2010). Spin-selective Peierls transition in interacting one-dimensional conductors with spin-orbit interaction. *Phys. Rev. B*, 82, 045127.
- Nadj-Perge, S. et al. (2014). Observation of Majorana fermions in ferromagnetic atomic chains on a superconductor. *Science*, 346, 602–607.
- Kitaev, A.Y. (2001). Unpaired Majorana fermions in quantum wires. *Physics-Uspekhi*, 44, 131.
- Heimes, A., Kotetes, P. & Schön, G. (2014). Majorana fermions from Shiba states in an antiferromagnetic chain on top of a superconductor. *Phys. Rev. B*, 90, 060507(R).
- Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
- Sitzmann, V. et al. (2020). Implicit Neural Representations with Periodic Activation Functions. (SIREN.) *NeurIPS*.