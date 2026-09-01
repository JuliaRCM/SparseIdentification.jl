# Basis Libraries

The library of candidate functions is the single most consequential choice in a SINDy run. **SINDy
cannot discover a term you did not offer it**, and every term you do offer costs conditioning and
raises the chance of a spurious fit. This page covers what is available and how to choose.

## Available bases

```@docs; canonical=false
SparseIdentification.AbstractBasis
SparseIdentification.PolynomialBasis
SparseIdentification.TrigonometricBasis
SparseIdentification.CompoundBasis
SparseIdentification.evaluate
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

## Not yet implemented

The thesis's implementation supported a wider set than this package currently exposes: **rational,
exponential and logarithmic** basis functions, and arithmetic combinations of state components
(differences ``q_i - q_j``, products, quotients). These matter for the examples it treats —

- the Toda lattice needs ``e^{-(q_{n+1} - q_n)}``, i.e. an exponential of a *difference*;
- the point-vortex system needs ``\log \lVert q_i - q_j \rVert``;
- the ``N``-body problem needs ``1/\lVert q_i - q_j \rVert``.

None of these three is currently expressible. Extending the basis machinery to cover them is
tracked in the changelog's open issues. Where a system's Hamiltonian is not in the span of the
polynomial and trigonometric library, an evolutionary symbolic-regression search is the better tool
— it composes operators rather than selecting from a fixed list.
