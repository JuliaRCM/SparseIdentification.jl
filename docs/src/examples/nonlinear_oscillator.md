# Nonlinear Oscillator

This is the thesis's first Hamiltonian example [Khan2023](@ref) — a two-degree-of-freedom system,
so four variables ``(q_1, q_2, p_1, p_2)``, combining quadratic and trigonometric terms:

```math
H = \tfrac{1}{2}p_1^2 + \tfrac{1}{2}p_2^2 + \cos(q_1) + \cos(q_2)
```

It is two uncoupled pendulums, and it is a good test because it needs both halves of a compound
basis: a polynomial library alone cannot represent ``\cos``, and a trigonometric one alone cannot
represent ``p^2``.

!!! warning "Correction to the thesis"
    Equation (4.2) of the thesis prints this Hamiltonian as
    ``H = \frac{1}{2}p_1^2 + \frac{1}{2}p_1^2 + \cos(q_1) + \cos(q_2)``, with ``\tfrac{1}{2}p_1^2``
    appearing **twice** and ``p_2`` not appearing at all. As printed it cannot be the described
    system: ``\dot q_2 = \partial H/\partial p_2 = 0``, so the second degree of freedom never moves.
    `scripts/verify_thesis_examples.jl` confirms this directly —

    ```
    as printed:  q̇ = [1.8, 0.0]   (p₂ = 1.7)
    as intended: q̇ = [0.9, 1.7]
    ```

    The form used here, with ``\tfrac{1}{2}p_2^2``, is what the surrounding text describes.

## The equations of motion

```math
\dot q_1 = p_1, \quad \dot q_2 = p_2, \qquad
\dot p_1 = \sin(q_1), \quad \dot p_2 = \sin(q_2)
```

Note the sign: ``\dot p = -\partial H/\partial q = -(-\sin q) = \sin q``. With ``+\cos q`` in the
Hamiltonian rather than the ``-\cos q`` of a physical pendulum, the equilibrium at the origin is
unstable — which is what gives the system the "intricate trajectory depending on the initial
condition" the thesis describes.

## Identification

```@example nlosc
using SparseIdentification

# the true vector field, ż = J∇H
function grad_H(z)
    q₁, q₂, p₁, p₂ = z
    [p₁, p₂, sin(q₁), sin(q₂)]
end

# uniform sampling over a broad range, as in the thesis
n = 400
x = 40 .* rand(4, n) .- 20
ẋ = reduce(hcat, [grad_H(x[:, j]) for j in axes(x, 2)])
nothing # hide
```

The thesis fits this with the flow-map formulation. Because the true derivatives are available
here, we can build consecutive states with a short explicit step and use `TrajectoryData`:

```@example nlosc
Δt = 0.01
xs = [x[:, j] for j in axes(x, 2)]
ys = [xs[j] .+ Δt .* grad_H(xs[j]) for j in eachindex(xs)]

method = HamiltonianSINDy(λ = 0.05, integrator_timestep = Δt,
                          polyorder = 2, trigonometric = 1)
nothing # hide
```

The library here holds quadratic polynomials in four variables plus ``\sin`` and ``\cos`` at
wavenumber one on each variable:

```@example nlosc
calculate_nparams(2, 2, 1)     # d = 2 degrees of freedom
```

The thesis used a considerably larger library — 97 terms, fourth-order polynomials and several
trigonometric frequencies — deliberately, to demonstrate identification with little prior
information. It reports the algorithm selecting four basis functions from those 97, within a
coefficient residual tolerance of 0.05, in a single SINDy cycle.

## Reported accuracy

From Table 4.3 of the thesis, with 10,000 samples from ``[-20, 20]`` and BFGS:

| noise | ``\lambda`` | max coefficient residual | SINDy cycles | iterations |
|:--|:--|:--|:--|:--|
| 2.5 % | 0.05 | 0.01 | 1 | 115 |
| 2.5 % | 0.1 | 0.007 | 1 | 110 |
| 10 % | 0.05 | 0.04 | 1 | 118 |
| 10 % | 0.1 | 0.02 | 1 | 95 |

Two things are worth reading off this table. The SINDy cycle count is always one, meaning the
support was correct after the first threshold pass — the system is easy to identify. And raising
``\lambda`` with the noise improves accuracy, which is the general guidance in [Choosing λ](@ref).

## Why this example is a good structural test

The Hamiltonian mixes term types that a purely polynomial method would have to approximate rather
than represent. Since the identified field is ``J\nabla H`` by construction, whatever coefficients
come out, the model conserves *some* Hamiltonian exactly. The question the example answers is
whether it conserves the **right** one — and a coefficient residual is the honest way to report
that, which is why the thesis tabulates residuals rather than trajectory agreement.
