# Hamiltonian SINDy

Classical SINDy fits a vector field. If the system you are identifying is Hamiltonian, that throws
away everything [Hamiltonian Systems](@ref) established: the fitted field will conserve no energy
exactly, its Jacobian will satisfy the symmetry condition only to within the fitting error, and
long-time integration will drift.

Hamiltonian SINDy fixes this by identifying the **scalar Hamiltonian** instead of the vector
field, and deriving the dynamics from it. The method is due to Khan [Khan2023](@ref), supervised by
Michael Kraus, and this package is its reference implementation.

## The ansatz

Parametrise the Hamiltonian sparsely over a library of ``p`` scalar candidate functions
``\varphi_k : \mathbb{R}^{2d} \to \mathbb{R}``:

```math
H(z; a) = \sum_{k=1}^{p} a_k \, \varphi_k(z) ,
```

and take the vector field to be the symplectic gradient of *that*:

```math
\dot z = J \nabla H(z; a) = \sum_{k=1}^{p} a_k \, J \nabla \varphi_k(z) .
```

The constant term is omitted from the library: it contributes nothing to ``\nabla H`` and is
therefore unidentifiable. More generally, ``H`` is only ever determined **up to an additive
constant**, which is a gauge freedom you should expect to see in results.

## Structure preservation is exact

Every candidate the fit considers is a symplectic gradient field, whatever the coefficients happen
to be. Non-Hamiltonian candidates are not penalised — they are *unrepresentable*.

Concretely, for any ``a`` at all, fitted or arbitrary, the Jacobian satisfies

```math
J^{-1} \, \partial f = \nabla^2 H(\cdot\,; a) , \qquad \text{symmetric.}
```

This is verified numerically in `scripts/verify_thesis_examples.jl`, on a deliberately arbitrary
coefficient vector that was never fitted to anything:

```
6. Is the identified field Hamiltonian for ANY coefficients?
     ‖J⁻¹∂f - (J⁻¹∂f)ᵀ‖ = 0.000e+00
```

Compare this with a fit-then-project approach, which identifies ``f`` freely and afterwards
projects onto the Hamiltonian ones: there, the projection distance is an error you must monitor,
and the intermediate model is not physical.

## The key structural consequence: the problem is linear

Because ``\nabla`` is linear and each ``a_k`` enters ``H`` linearly,

```math
\dot z = \sum_{k=1}^{p} a_k \, J \nabla \varphi_k(z)
```

is **linear in the coefficient vector ``a``.**

Define the library matrix by stacking the symplectic gradients of the basis functions,

```math
\Theta_H(z) = \big[\; J\nabla\varphi_1(z) \;\big|\; J\nabla\varphi_2(z) \;\big|\; \cdots \;\big|\; J\nabla\varphi_p(z) \;\big] \in \mathbb{R}^{2d \times p} ,
```

and the model is simply ``\dot z = \Theta_H(z)\, a``.

This matters because it means **fitting a Hamiltonian against measured derivatives is an ordinary
linear least-squares problem** — one backslash and a thresholding loop — not a nonlinear
optimisation. It is verified in `scripts/verify_thesis_examples.jl`:

```
1. Is J∇H linear in the coefficients?
     ‖J∇H(αa₁+βa₂) - (α J∇H(a₁) + β J∇H(a₂))‖ = 0.000e+00
     ‖Θa - J∇H(a)‖ = 2.220e-16
```

!!! note "This differs from the thesis"
    The thesis states that in Hamiltonian SINDy "the coefficients cannot be as easily factorized
    into a linear system of equations […] the vector field usually depends linearly on the
    coefficients and thus can, in principle, be transformed into a matrix-vector product. It is
    just much more involved to do this." It therefore uses BFGS throughout, including where
    derivative data is available.

    The linearity is exact, not approximate, and the transformation is not involved: it is one
    symbolic gradient per basis function, computed once. Where ``\dot z`` is available, the linear
    formulation is both faster and deterministic.

There is one important structural difference from classical SINDy. In classical SINDy each state
component has its own coefficient column, and the refits are independent. Here there is a **single
coefficient vector shared across all ``2d`` components**, because they all come from one scalar
``H``. The regression is therefore one joint least-squares problem over the stacked residual, not
``2d`` separate ones.

## Fitting without derivative data

Derivative data is often unavailable — in many-body systems it is expensive to compute, and in
experimental settings it may not be measurable at all. The thesis's answer, and the formulation
currently implemented in this package, is to match the **flow map** instead.

Given consecutive states ``z_j`` and ``z_{j+1} = \Phi_{\Delta t}(z_j)`` separated by a short
interval, minimise

```math
\mathcal{L}(a) = \sum_j \big\lVert \Phi^a_{\Delta t}(z_j) - z_{j+1} \big\rVert^2 ,
```

where ``\Phi^a_{\Delta t}`` is one step of a numerical integrator applied to the candidate field
``J\nabla H(\cdot\,; a)``. The implementation uses the **implicit midpoint rule**, solved by a
fixed number of Picard iterations, with an explicit Euler step as the initial guess:

```math
\tilde z^{(0)} = z_j + \Delta t\, f_a(z_j), \qquad
\tilde z^{(i+1)} = z_j + \Delta t\, f_a\!\left(\tfrac{z_j + \tilde z^{(i)}}{2}\right) .
```

Implicit midpoint is itself symplectic, so the fitted model is compared against data through a
structure-preserving integrator rather than a generic one.

This formulation is **nonlinear in ``a``** — the coefficients enter through the integrator — so it
genuinely needs an optimiser. The package uses BFGS from
[GeometricOptimizers.jl](https://github.com/JuliaGNI/GeometricOptimizers.jl), with coefficients
initialised to zero.

!!! warning "The Picard iteration count is fixed, not converged"
    The number of Picard iterations is a parameter (`picard_iterations`, default 4), not a
    convergence tolerance. The step computed is therefore *an approximation to* the implicit
    midpoint step, to no stated accuracy. The thesis notes this as a limitation and attributes part
    of the residual accuracy gap to it. Treat the flow-map formulation as accurate to about two
    decimal places in the coefficients, against roughly five for the vector-field formulation.

## Which formulation to use

The package does not pick for you, because the two suit genuinely different data and silently
applying one to data meant for the other gives a wrong answer rather than an error.

| | vector-field matching | flow-map matching |
|:--|:--|:--|
| **needs** | states and derivatives ``\dot z`` | consecutive states ``z_j, z_{j+1}`` |
| **problem type** | linear least squares | nonlinear optimisation |
| **cost** | one `\` per threshold pass | BFGS over the whole trajectory set |
| **determinism** | exact and reproducible | depends on optimiser convergence |
| **accuracy** | limited by derivative noise | limited by the integrator and Picard count |

```@docs; canonical=false
SparseIdentification.HamiltonianSINDy
```

## Sparsification

The thresholding loop is the same as classical SINDy's, with one difference forced by the shared
coefficient vector: there is one support, not ``2d`` of them. Coefficients below ``\lambda`` are
zeroed, and the model is refitted over the survivors.

The thesis's guidance on ``\lambda``, borne out by its results tables, is that it should sit near
the noise amplitude — and that raising it as noise rises improves accuracy. On the nonlinear
oscillator at 10 % noise, raising ``\lambda`` from 0.05 to 0.1 halved the maximum coefficient
residual, from 0.04 to 0.02. See [Choosing λ](@ref).
