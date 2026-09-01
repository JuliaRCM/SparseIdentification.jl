# Toda Lattice

The Toda lattice is a one-dimensional chain of particles with exponential nearest-neighbour
interaction — a classical integrable model from solid-state physics, and the thesis's second
Hamiltonian example [Khan2023](@ref). It is included here because it is the clearest demonstration
in the thesis of what the symplectic structure actually buys.

```math
H(q, p) = \sum_{n} \left( \frac{p_n^2}{2} + V(q_{n+1} - q_n) \right),
\qquad V(r) = e^{-r} + r - 1
```

## The potential is correctly normalised

``V`` is the standard normalised Toda potential, confirmed in
`scripts/verify_thesis_examples.jl`:

```
V(0) = 0.000,  V'(0) = 0.000,  V''(0) = 1.000
```

``V(0) = V'(0) = 0`` places the equilibrium at zero displacement with zero force, and ``V''(0) = 1``
normalises the harmonic stiffness — which is what makes the lattice reduce to coupled unit-frequency
harmonic oscillators at small amplitude, and the exponential nonlinearity dominate at large ones.

## What the thesis measured

A four-particle system, a library of 156 terms, 2000 samples from a standard normal distribution,
coefficients initialised to zero, BFGS with a convergence criterion of ``10^{-8}``. Trajectories
were integrated with an implicit Runge–Kutta Gauss integrator from
[GeometricIntegrators.jl](https://github.com/JuliaGNI/GeometricIntegrators.jl), time step 0.01 over
250 time units.

Table 4.4 of the thesis:

| method | noise | ``\lambda`` | max coefficient residual | SINDy cycles | iterations |
|:--|:--|:--|:--|:--|:--|
| Classical least squares | 2.5 % | 0.05 | 0.5 | 2 | – |
| Classical BFGS | 2.5 % | 0.05 | 0.504 | 2 | 346 |
| **Hamiltonian BFGS** | 2.5 % | 0.05 | **0.006** | 1 | 140 |
| Hamiltonian BFGS | 10 % | 0.05 | 0.07 | 1 | 146 |
| Hamiltonian BFGS | 2.5 % | 0.1 | **0.003** | 1 | 140 |
| Hamiltonian BFGS | 10 % | 0.1 | 0.003 | 1 | 142 |

## The result worth taking away

**Hamiltonian SINDy is roughly two orders of magnitude more accurate than classical SINDy on the
same data**, at half the iterations and one SINDy cycle instead of two.

That gap is not a tuning artefact. Classical SINDy has to discover, from data, that the vector
field happens to be a symplectic gradient — a constraint it has no way to express, so it spends its
coefficients approximating a structure it cannot represent. Hamiltonian SINDy has that structure
built into every candidate it evaluates and spends its coefficients only on the physics.

The high iteration count for classical BFGS (346 against 140) tells the same story from the other
side: minimising over a space that contains vastly many non-Hamiltonian fields is a harder
optimisation problem than minimising over one that contains none.

This is the empirical case for the whole approach, and it generalises: where you know a structural
property of the system, imposing it by construction beats fitting freely and hoping.

## Not currently reproducible here

The Toda Hamiltonian needs ``e^{-(q_{n+1} - q_n)}`` — an exponential of a **difference of state
components**. The package's current basis library offers polynomials and trigonometric functions of
individual components only, so this system cannot presently be expressed. The thesis's
implementation supported exponential, logarithmic and rational bases together with arithmetic
combinations of components; restoring that is tracked in the changelog's open issues.

Until then, the numbers above are reported from the thesis rather than recomputed, and are marked as
such. The `GeometricProblems.TodaLattice` module provides the system itself, so the data-generation
half is available in-tree already.
