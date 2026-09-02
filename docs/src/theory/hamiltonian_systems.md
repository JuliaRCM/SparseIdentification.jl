# Hamiltonian Systems

This page fixes the notation and states the two facts the rest of the package rests on: what a
Hamiltonian vector field is, and what "preserving the symplectic structure" buys you. Readers who
know Hamiltonian mechanics can skip to [Sparse Identification](@ref).

## From Lagrange to Hamilton

A mechanical system with configuration coordinates ``q \in \mathbb{R}^d`` and Lagrangian
``L(q, \dot q, t)`` follows the stationary points of the action, giving the Euler–Lagrange
equations

```math
\frac{d}{dt}\frac{\partial L}{\partial \dot q} - \frac{\partial L}{\partial q} = 0 .
```

Defining the conjugate momentum ``p = \partial L / \partial \dot q`` and passing to the Legendre
transform ``H(q,p,t) = p \cdot \dot q - L`` turns this second-order system in ``q`` into a
first-order system in the ``2d`` variables ``z = (q, p)``:

```math
\dot q = \frac{\partial H}{\partial p}, \qquad
\dot p = -\frac{\partial H}{\partial q} .
```

These are **Hamilton's equations**. The pair ``(q, p)`` are *canonical conjugate coordinates*, and
``H`` is the Hamiltonian — for most mechanical systems, the total energy.

## The canonical symplectic form

Hamilton's equations have a compact form that is the one this package uses throughout. Collect the
state into ``z = (q_1, \dots, q_d, p_1, \dots, p_d) \in \mathbb{R}^{2d}`` and define the canonical
symplectic matrix

```math
J = \begin{pmatrix} 0 & I_d \\ -I_d & 0 \end{pmatrix} .
```

``J`` is skew-symmetric (``J^\top = -J``) and orthogonal (``J^{-1} = J^\top = -J``). Hamilton's
equations become

```math
\dot z = J \, \nabla H(z) .
```

A vector field of this form is called a **Hamiltonian vector field**. Everything the package does
on the Hamiltonian side is organised around this single equation.

## Why the structure matters

Three consequences follow, and they are the reason to identify ``H`` rather than ``\dot z``
directly.

**Energy is conserved.** Along any solution,

```math
\frac{dH}{dt} = \nabla H \cdot \dot z = \nabla H \cdot J \nabla H = 0 ,
```

because ``J`` is skew-symmetric and ``x^\top J x = 0`` for every ``x``. Note this holds for *any*
``H`` — it is a property of the form of the equation, not of a particular system.

**The flow preserves phase-space volume** (Liouville's theorem), and more strongly preserves the
symplectic two-form. Trajectories cannot spiral into an attractor; a damped oscillator is not
Hamiltonian, which is exactly why the damped linear oscillator appears in this documentation as a
*non*-Hamiltonian example.

**The Jacobian is constrained.** Differentiating ``f(z) = J\nabla H(z)`` gives
``\partial f = J \, \nabla^2 H``, so

```math
J^{-1} \, \partial f = \nabla^2 H \quad\text{is symmetric.}
```

This is the sharp characterisation: a vector field is (locally) Hamiltonian if and only if
``J^{-1}\partial f`` is symmetric. It is also a property you can *test*, which is what
`scripts/verify_thesis_examples.jl` does — see [Structure preservation is exact](@ref).

## What this means for identification

Suppose you fit a general vector field ``\dot z = f(z)`` from data, as classical SINDy does. Nothing
in that fit knows about ``J``. The identified ``f`` will be approximately Hamiltonian if the data
is, but only approximately: the symmetry condition above will hold to within the fitting error,
and integrating the identified system will drift in energy.

If instead you parametrise ``H`` and *derive* ``f = J\nabla H``, then the symmetry condition holds
to machine precision for every candidate the fit ever considers, including wrong ones. The
constraint is structural, not a penalty term. That distinction is the subject of
[Hamiltonian SINDy](@ref).

## Non-canonical and Poisson systems

The package currently assumes **canonical** coordinates, i.e. the constant ``J`` above. Many
systems of interest — rigid-body rotation, point vortices, guiding-centre dynamics — are Poisson
rather than canonically Hamiltonian, with a state-dependent structure matrix ``J(z)`` satisfying
the Jacobi identity. Identifying those requires learning ``J(z)`` as well as ``H``, which this
package does not do; see [When It Fails](@ref) for what goes wrong if you assume canonical
structure where there is none.
