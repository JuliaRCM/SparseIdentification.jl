# Basis Libraries

The library of candidate functions is the single most consequential choice in a SINDy run. **SINDy
cannot discover a term you did not offer it**, and every term you do offer costs conditioning and
raises the chance of a spurious fit. This page covers what is available and how to choose.

## Available bases

```@docs; canonical=false
SparseIdentification.AbstractBasis
SparseIdentification.PolynomialBasis
SparseIdentification.TrigonometricBasis
SparseIdentification.ExponentialBasis
SparseIdentification.LogarithmicBasis
SparseIdentification.RationalBasis
SparseIdentification.CompoundBasis
SparseIdentification.basis_functions
SparseIdentification.evaluate
```

## One definition, not two

A basis is defined **symbolically**, by `basis_functions(basis, z)`, and everything else is derived
from that single definition: `evaluate` compiles the expressions into a fast numerical evaluator
(cached per basis and state dimension), and the Hamiltonian methods differentiate the same
expressions to build `J∇φₖ`. There is deliberately no second, hand-written numeric path that could
drift from the symbolic one.

```@example basis
using SparseIdentification, Symbolics

z = Symbolics.variables(:z, 1:2)
basis_functions(CompoundBasis(polyorder = 2), z)
```

## Polynomials

`CompoundBasis(polyorder = n)` assembles the constant, then all monomials of degree 1 through
``n``, without repetition. For ``d`` degrees of freedom the count is ``\binom{d+n}{n}``.

```@example basis
using SparseIdentification

basis = CompoundBasis(polyorder = 3, trigonometric = 0)
x = randn(2, 5)
Θ = evaluate(x, basis)
size(Θ)      # 5 snapshots × 10 terms: 1 + 2 + 3 + 4
```

The column order is: the constant, then degree 1 in state order, then degree 2 as
``x_1^2, x_1x_2, x_2^2``, and so on.

```@example basis
Θ[:, 1] ≈ ones(5), Θ[:, 2] ≈ x[1, :], Θ[:, 4] ≈ x[1, :] .^ 2
```

Growth is combinatorial. In four variables — a two-degree-of-freedom Hamiltonian system — a cubic
library has 34 terms, a quintic one 125. The thesis routinely works with libraries of roughly 100
to 170 terms, which is a reasonable ceiling.

## Trigonometric terms

`TrigonometricBasis(n)` supplies ``\sin(k x_i)`` and ``\cos(k x_i)`` for ``1 \le k \le n`` on every
component:

```@example basis
btrig = TrigonometricBasis(2)
Θt = evaluate(x, btrig)
size(Θt)     # 5 × 8:  2 wavenumbers × {sin, cos} × 2 components
```

These are essential for pendulum-like systems, whose Hamiltonians carry ``\cos q`` and which no
polynomial library can represent. `CompoundBasis(polyorder = p, trigonometric = n)` combines both.

## Choosing a library

**Start from what you know about the system.** The thesis's framing is worth adopting: supply
enough to represent plausible dynamics, but treat the library as an expression of genuine prior
belief rather than a fishing expedition. Its nonlinear-oscillator run started from 97 candidate
functions and selected 4.

**Watch the conditioning.** Near-duplicate terms make the least-squares refit ill-conditioned.
Some redundancy is unavoidable — ``\sin`` and ``\cos`` at different wavenumbers are not orthogonal
on a finite sample — but adding both a high-degree polynomial and an exponential of the same
variable invites trouble.

**Sample where the library is distinguishable.** Terms that differ only at large amplitude cannot
be told apart from data clustered near the origin. Uniform sampling over a broad range, as used
throughout the thesis, distinguishes them better than a single trajectory that lingers in one
region.

## Arguments: applying a function to *differences*

This is the piece that makes interacting systems expressible, and it is easy to miss. Applying
``\exp`` to individual state components gives ``e^{q_1}, e^{q_2}, \dots``, which is useless for a
lattice: a Toda chain interacts through ``e^{-(q_{n+1} - q_n)}``, an exponential of a
**difference**. The same is true of a point vortex (``\log\lvert q_i - q_j\rvert``) and of the
``N``-body problem (``1/\lvert q_i - q_j\rvert``).

Every univariate basis therefore takes an argument selection:

```@docs; canonical=false
SparseIdentification.BasisArguments
SparseIdentification.StateComponents
SparseIdentification.Differences
```

`Differences(indices)` forms all pairs ``z_i - z_j`` with ``i > j``, which is what an all-to-all
interaction needs; `Differences(indices; consecutive = true)` forms only neighbouring differences,
which is what a nearest-neighbour lattice needs. The `indices` select which components take part —
for a Hamiltonian system in ``z = (q, p)`` the interaction is usually among the positions alone,
i.e. the first half.

```@example basis
# a Toda chain of three particles: interaction among the positions only
basis_functions(ExponentialBasis(Differences(1:3; consecutive = true); rates = (-1.0,)), 
                Symbolics.variables(:z, 1:6))
```

```@example basis
# a point vortex: log of every pairwise separation
basis_functions(LogarithmicBasis(Differences(1:3)), Symbolics.variables(:z, 1:6))
```

```@example basis
# an N-body gravitational term
basis_functions(RationalBasis(Differences(1:3)), Symbolics.variables(:z, 1:6))
```

## Combining bases

`⊕` concatenates bases, so a library is assembled from the pieces a system actually needs rather
than from one monolithic keyword:

```@example basis
basis = hamiltonian_basis(polyorder = 2) ⊕
        ExponentialBasis(Differences(1:2; consecutive = true); rates = (-1.0,))

nterms(basis, 4)
```

```@docs; canonical=false
SparseIdentification.hamiltonian_basis
SparseIdentification.strip_constants
```

For a Hamiltonian ansatz the constant term is dropped, because it contributes an identically-zero
column to ``J\nabla H`` and so cannot be identified. `hamiltonian_basis` omits it, and
`strip_constants` removes any term whose gradient vanishes — filtering on the gradient rather than
on the type of the term catches every such case, whatever basis it came from.

## What is still out of reach

The bases above take **scalar** arguments. A genuinely three-dimensional ``N``-body problem needs
``1/\lVert \mathbf{q}_i - \mathbf{q}_j \rVert`` — the norm of a difference of position
*vectors* — which needs a block structure over components that `Differences` does not express. In
one spatial dimension the rational basis above is exactly right; in three it is not.

Where a system's Hamiltonian is not in the span of any fixed library, an evolutionary
symbolic-regression search is the better tool, since it composes operators rather than selecting
from a list.
