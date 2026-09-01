# Getting Started

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaRCM/SparseIdentification.jl")
```

The package requires Julia 1.11 or later.

## The shape of a SINDy run

Every identification has the same four parts, whichever method you use:

1. **Data** — states, and either their derivatives or their successors one time step on.
2. **A basis** — the library of candidate functions to search.
3. **A method** — `SINDy` or `HamiltonianSINDy`, carrying the sparsification threshold.
4. **A vector field** — the identified model, callable and integrable.

## Classical SINDy

The two-dimensional damped linear oscillator is the illustrative example of the original SINDy
paper, and a good first check because the answer is known exactly:

```math
\dot x = -0.1x + 2y, \qquad \dot y = -2x - 0.1y .
```

```@example started
using SparseIdentification

# the true system
A = [-0.1  2.0
     -2.0 -0.1]

# sample the state space and evaluate the true vector field on it
x = randn(2, 500)
ẋ = A * x

# search polynomials up to degree 3
basis = CompoundBasis(polyorder = 3, trigonometric = 0)

# identify
result = identify(TrainingData(x, ẋ), SINDy(basis; λ = 0.05))
nothing # hide
```

The coefficient matrix maps library terms to state components. Rows 2 and 3 hold the linear terms,
so that block should be `A` transposed:

```@example started
parameters(result)[2:3, :]
```

and everything else should be exactly zero — not merely small:

```@example started
Ξ = parameters(result)
all(iszero, Ξ[1, :]) && all(iszero, Ξ[4:end, :])
```

Note that the data here is **sampled from the state space**, not taken along a single trajectory.
Both work; uniform sampling over a region covers the library's domain more evenly and needs no
integration, which is why the thesis uses it throughout.

## Identifying a Hamiltonian

For a Hamiltonian system, identify the Hamiltonian instead. Take the harmonic oscillator,
``H = \tfrac{1}{2}(q^2 + p^2)``, whose flow is a rotation in phase space:

```@example started
using SparseIdentification

Δt = 0.01
R  = [ cos(Δt) sin(Δt)
      -sin(Δt) cos(Δt)]          # the exact flow map over one step

x = [randn(2) for _ in 1:60]     # states
y = [R * xⱼ for xⱼ in x]         # the same states one step later

method = HamiltonianSINDy(λ = 0.05, integrator_timestep = Δt, polyorder = 2)
result = identify(TrajectoryData(x, y, Δt), method)
vf     = HamiltonianSINDyVectorField(result)
nothing # hide
```

The identified field should reproduce ``\dot z = (p, -q)`` at points it never saw:

```@example started
dz = zeros(2)
z  = [0.7, -0.3]
vf(dz, z)
dz            # should be ≈ [-0.3, -0.7]
```

Because the field is a symplectic gradient by construction, this model conserves *some* Hamiltonian
exactly, whether or not the coefficients are right.

## Data layout

Two container types, distinguished by what they mean rather than by shape:

```@docs; canonical=false
SparseIdentification.TrainingData
SparseIdentification.TrajectoryData
```

Both accept either a matrix whose columns are snapshots, or a vector of state vectors. Both check
at construction that the pieces describe the same number of snapshots of the same dimension — a
shape error here otherwise surfaces much later as a wrong answer rather than an exception.

```@docs; canonical=false
SparseIdentification.statedimension
```

## Noise belongs to the data

The estimator does not add noise. If you want to test robustness, add it when you *generate* the
data:

```@example started
using Random
Random.seed!(1234)

A  = [-0.1 2.0; -2.0 -0.1]
x  = randn(2, 500)
ẋ  = A * x
η  = 0.05
ẋ_noisy = ẋ .+ η .* randn(size(ẋ))       # noise added here, deliberately and reproducibly

basis = CompoundBasis(polyorder = 3, trigonometric = 0)
Ξ = parameters(identify(TrainingData(x, ẋ_noisy), SINDy(basis; λ = 0.05)))
Ξ[2:3, :]
```

Earlier versions of this package injected noise inside the fitting routine, which made results
irreproducible and could not be switched off. Two identical calls now give identical answers.
