```@meta
CurrentModule = SparseIdentification
```

# SparseIdentification.jl

Data-driven discovery of governing equations, with an extension that preserves the symplectic
structure of Hamiltonian systems.

Given measurements of a dynamical system's state, SparseIdentification recovers the equations that
generated them, by selecting a small number of terms from a library of candidate functions. For
Hamiltonian systems it goes further: rather than fitting the vector field, it identifies the scalar
**Hamiltonian**, so that the discovered dynamics is Hamiltonian *by construction* rather than by
approximation.

## Why structure preservation

A general-purpose fit knows nothing about the physics. Applied to a Hamiltonian system it produces
a model that conserves energy only approximately and drifts under long integration. Identifying
``H`` and deriving ``\dot z = J\nabla H`` makes conservation exact for every candidate model
considered, including wrong ones — the constraint is structural, not a penalty.

The payoff is measurable. On the Toda lattice, the structure-preserving method reaches a maximum
coefficient residual of **0.006** where classical SINDy on the same data reaches **0.5** — roughly
two orders of magnitude, at half the optimiser iterations. See [Toda Lattice](@ref).

## Two lines to try it

```@example index
using SparseIdentification

A = [-0.1 2.0; -2.0 -0.1]          # the system to rediscover
x = randn(2, 500); ẋ = A * x

result = identify(TrainingData(x, ẋ), SINDy(CompoundBasis(polyorder = 3); λ = 0.05))
parameters(result)[2:3, :]         # recovered: A transposed
```

## Where to go next

| | |
|:--|:--|
| **[Getting Started](@ref)** | installation, the shape of a run, both methods end to end |
| **[Hamiltonian Systems](@ref)** | notation, the symplectic form, why structure matters |
| **[Sparse Identification](@ref)** | the SINDy formulation, STLSQ, what is actually guaranteed |
| **[Hamiltonian SINDy](@ref)** | the extension, and why the problem is linear in the coefficients |
| **[Choosing λ](@ref)** | the one parameter that matters, with measured guidance |
| **[When It Fails](@ref)** | the failure modes, with evidence — read this before trusting a result |

## Provenance

The Hamiltonian method implemented here is due to Nigel Bruce Khan's master's thesis at the
Technische Universität München, supervised by Michael Kraus [Khan2023](@ref).

Claims taken from that thesis are **verified rather than transcribed**. The script
`scripts/verify_thesis_examples.jl` checks each one and reports what holds and what does not; three
of its equations need correcting, and the corrections are stated wherever this documentation
touches them ([Nonlinear Oscillator](@ref), [When It Fails](@ref)).

## Status

Working and tested: classical SINDy with sequentially thresholded least squares, polynomial and
trigonometric bases, and Hamiltonian SINDy in its flow-map formulation.

Not yet implemented, and tracked in the release notes: the linear vector-field formulation of
Hamiltonian SINDy, rational/exponential/logarithmic bases, the autoencoder variant that discovers
canonical coordinates, and the weak formulation for noisy data.
