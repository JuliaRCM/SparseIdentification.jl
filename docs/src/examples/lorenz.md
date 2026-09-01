# Lorenz Attractor

The Lorenz system is the standard chaotic benchmark for SINDy, and the headline example of the
original paper [BruntonProctorKutz2016](@ref):

```math
\begin{aligned}
\dot x &= \sigma (y - x) \\
\dot y &= x(\rho - z) - y \\
\dot z &= x y - \beta z
\end{aligned}
```

with the classical parameters ``\sigma = 10``, ``\rho = 28``, ``\beta = 8/3``. Seven terms in three
equations, all of degree at most two — sparse in a quadratic polynomial library, which is what
makes it a good test.

## Exact recovery

```@example lorenz
using SparseIdentification

σ, ρ, β = 10.0, 28.0, 8/3

x = 10 .* randn(3, 2000)
ẋ = similar(x)
for j in axes(x, 2)
    ẋ[:, j] .= lorenz(x[:, j], (σ, β, ρ), 0.0)
end

basis = CompoundBasis(polyorder = 2, trigonometric = 0)
Ξ = VectorField(SINDy(λ = 0.1), basis, TrainingData(x, ẋ)).coefficients

count(!iszero, Ξ)     # the true model has 7 terms
```

The library order for three degrees of freedom is
``[1,\; x,\; y,\; z,\; x^2,\; xy,\; xz,\; y^2,\; yz,\; z^2]``, so the expected non-zeros are:

```@example lorenz
truth = zeros(10, 3)
truth[2, 1] = -σ;  truth[3, 1] = σ            # ẋ = σ(y − x)
truth[2, 2] = ρ;   truth[3, 2] = -1.0
truth[7, 2] = -1.0                            # ẏ = x(ρ − z) − y  =  ρx − y − xz
truth[4, 3] = -β;  truth[6, 3] = 1.0          # ż = xy − βz

maximum(abs, Ξ - truth)
```

Recovery is exact to machine precision on clean data. This is asserted in the package's test suite
at a tolerance of `1e-10`.

## Why sampling beats a trajectory here

It is tempting to generate the data by integrating the Lorenz system and using the trajectory. That
works, but it conditions the problem badly: a Lorenz trajectory lies on a strange attractor of
fractal dimension about 2.06 embedded in three dimensions, so the sampled states occupy a set of
nearly zero volume. The library columns evaluated on that set are far closer to collinear than they
would be on a spread of states.

Sampling the state space directly, as above, spans the region properly. The cost is that you need
the true vector field to generate ``\dot X`` — fine for a benchmark, not available for real data.
With real trajectory data, the derivative has to be estimated numerically, and that is where the
weak formulation earns its keep (see [Sparse Identification](@ref)).

## Noise

Lorenz is chaotic, so trajectory comparison is not a meaningful accuracy measure over any
appreciable time — two models differing in the twelfth decimal place diverge visibly within a few
Lyapunov times. Judge the identification by **coefficient recovery and by the support**, not by how
long the trajectories stay together.

```@example lorenz
using Random
Random.seed!(7)

ẋ_noisy = ẋ .+ 0.5 .* randn(size(ẋ))
Ξn = VectorField(SINDy(λ = 0.5), basis, TrainingData(x, ẋ_noisy)).coefficients

println("active terms: ", count(!iszero, Ξn), "   (true: 7)")
println("max coefficient error: ", maximum(abs, Ξn - truth))
```

The support survives noise well; the coefficient values degrade gracefully. That asymmetry —
structure recovered before precision — is characteristic of thresholded methods and is the main
practical reason to use them.
