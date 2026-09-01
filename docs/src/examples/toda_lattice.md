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

## Building the library

The Toda Hamiltonian needs ``e^{-(q_{n+1} - q_n)}`` — an exponential of a **difference** of state
components — which is exactly what [`Differences`](@ref) exists for. For a two-particle chain, with
``z = (q_1, q_2, p_1, p_2)`` so the positions are components 1 and 2:

```@example toda
using SparseIdentification
using Symbolics

basis = hamiltonian_basis(polyorder = 2) ⊕
        ExponentialBasis(Differences(1:2; consecutive = true); rates = (-1.0,))

z = Symbolics.variables(:z, 1:4)
φ = basis_functions(basis, z)

length(φ), φ[end]      # the interaction term is in the library
```

The interaction is formed over the *positions only*, which is what `Differences(1:2)` says: the
momenta do not interact, and including them would add terms that cannot appear in a Toda
Hamiltonian.

## Checking it against the exact field

With the coefficients of the true Hamiltonian ``H = \tfrac{1}{2}(p_1^2 + p_2^2) + e^{-(q_2 - q_1)}``
set by hand, the compiled ``J\nabla H`` must reproduce the exact vector field
``\dot z = (p_1,\, p_2,\, -e,\, e)`` with ``e = e^{-(q_2-q_1)}``:

```@example toda
hfuns = SparseIdentification.hamiltonian_functions(basis, 2)

a = zeros(hfuns.nparam)
a[findfirst(isequal(z[3]^2), φ)]          = 0.5      # ½p₁²
a[findfirst(isequal(z[4]^2), φ)]          = 0.5      # ½p₂²
a[findfirst(isequal(exp(z[1] - z[2])), φ)] = 1.0     # e^{-(q₂-q₁)}

function exact(z)
    e = exp(-(z[2] - z[1]))
    [z[3], z[4], -e, e]
end

zv  = [0.3, -0.7, 1.1, 0.4]
out = zeros(4)
hfuns.ż(out, zv, a)

out, exact(zv)
```

```@example toda
maximum(abs, out - exact(zv))
```

This is the check that the library really contains the system, before any fitting is attempted —
worth doing first, because a library that cannot represent the answer produces a confident, dense,
wrong model rather than an error.

The numbers in the table above are reported from the thesis rather than recomputed here; the
`GeometricProblems.TodaLattice` module provides the system itself for a full reproduction.
