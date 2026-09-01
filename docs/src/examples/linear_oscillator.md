# Linear 2D Oscillator

The two-dimensional damped linear oscillator is the illustrative example of the original SINDy
paper [BruntonProctorKutz2016](@ref) and the first example of the thesis. It is the right first
test because the answer is known exactly and the recovery should be exact too.

```math
\dot x = -0.1\,x + 2\,y, \qquad \dot y = -2\,x - 0.1\,y
```

The system is a stable spiral: decay rate ``0.1``, angular frequency ``2``. Note it is **not**
Hamiltonian — energy decays — which is why it belongs to classical SINDy and not to the Hamiltonian
path.

## Exact recovery on clean data

```@example linear
using SparseIdentification

A = [-0.1  2.0
     -2.0 -0.1]

x = randn(2, 500)
ẋ = A * x

basis = CompoundBasis(polyorder = 3, trigonometric = 0)
Ξ = VectorField(SINDy(λ = 0.05), basis, TrainingData(x, ẋ)).coefficients

Ξ[2:3, :]        # the linear block: should equal A'
```

```@example linear
maximum(abs, Ξ[2:3, :] - A')
```

Every other coefficient is exactly zero, not merely small — the thresholding sets them to zero and
the refit never reintroduces them:

```@example linear
all(iszero, Ξ[1, :]) && all(iszero, Ξ[4:end, :])
```

## Robustness to noise

The thesis's headline claim for this example is that the algorithm is highly resistant to noise. It
used 144 samples drawn uniformly from ``[-20, 20]``, with Gaussian noise at 10 % of the sampled
range, a library of 21 polynomial terms up to fifth order, and ``\lambda = 0.05``; it reports the
correct basis functions recovered with one spurious constant, accurate to one decimal place.

Reproducing the structure of that experiment:

```@example linear
using Random
Random.seed!(42)

A = [-0.1 2.0; -2.0 -0.1]

# uniform sampling over a broad range, as in the thesis
x = 40 .* rand(2, 144) .- 20
ẋ = A * x

# noise at 10% of the sampled range
η = 0.10 * 20
ẋ_noisy = ẋ .+ η .* randn(size(ẋ))

basis = CompoundBasis(polyorder = 5, trigonometric = 0)
Ξ = VectorField(SINDy(λ = 0.05), basis, TrainingData(x, ẋ_noisy)).coefficients

println("active terms: ", count(!iszero, Ξ), "  (true model has 4)")
println("linear block:\n", Ξ[2:3, :])
println("max error on the linear block: ", maximum(abs, Ξ[2:3, :] - A'))
```

The linear terms come back close to their true values while the higher-order terms are thresholded
away. Typically one spurious term survives — which is precisely what the thesis reports for this
experiment: "the discovered coefficient values […] are the correct basis functions with only one
extra constant bias coefficient showing up […] accurate within one decimal place of their actual
value."

The exact figures depend on the random draw. What matters is that the *support* is nearly
recovered under noise at 10 % of the sampled range, and that the surviving spurious term is small:
structure is recovered before precision, which is the characteristic behaviour of thresholded
methods.

## Sampling the state space rather than a trajectory

Both this page and the thesis sample states uniformly over a region rather than integrating a
single trajectory. That differs from the original MATLAB implementation, and the thesis is explicit
about why: it is faster, since no trajectory has to be computed, and the sampling region can be
chosen to cover the domain where the library terms are distinguishable.

A trajectory of a stable spiral spends most of its time near the origin, where a fifth-order
polynomial library is very poorly conditioned — every high-degree term is nearly zero and nearly
collinear with every other. Uniform sampling avoids that.
