# Choosing λ

``\lambda`` is the one parameter that matters most, and the one with no default that is right for
every problem. This page collects what is actually known about setting it, including the measured
evidence from Khan [Khan2023](@ref).

## What λ does

After each least-squares fit, every coefficient with ``|\Xi_{ij}| < \lambda`` is set to exactly
zero and removed from the library for subsequent refits. It is a hard threshold on **coefficient
magnitude**, not a penalty weight — it does not appear in any objective. Consequences:

- ``\lambda`` too small: spurious terms survive, the model overfits noise, and the identified
  equations are dense and uninterpretable.
- ``\lambda`` too large: genuine terms are cut, and the refit compensates by distorting the ones
  that remain.

The failure is asymmetric. Too small gives you a model that is right but cluttered; too large gives
you a model that is confidently wrong.

## Match λ to the noise

The working rule from the thesis is that ``\lambda`` should sit near the noise amplitude, and
should be **raised as noise rises**. Its nonlinear oscillator results (Table 4.3, 10,000 samples
drawn uniformly from ``[-20, 20]``, BFGS):

| noise | ``\lambda`` | max coefficient residual | iterations |
|:--|:--|:--|:--|
| 2.5 % | 0.05 | 0.01 | 115 |
| 2.5 % | 0.1 | 0.007 | 110 |
| 10 % | 0.05 | 0.04 | 118 |
| 10 % | 0.1 | **0.02** | 95 |

At 10 % noise, doubling ``\lambda`` halved the residual *and* reduced the iteration count, because
more spurious coefficients were removed before the optimiser had to work around them. The same
pattern appears in the Toda lattice results (Table 4.4), where the best case is the pairing of low
noise with the higher threshold.

The thesis reports that improvement stops around ``\lambda = 0.2``, which is roughly half the
smallest true coefficient — the point past which the threshold starts eating signal.

That last observation generalises into the useful bound:

!!! tip "The ceiling on λ"
    ``\lambda`` must stay below the smallest coefficient you hope to recover. If you have any prior
    estimate of that magnitude, half of it is a reasonable upper limit.

## Sweep it

There is no way to know the right value in advance, so sweep and look at where the answer stops
changing. A stable support across a range of ``\lambda`` is the signal you want:

```@example lambda
using SparseIdentification

A = [-0.1 2.0; -2.0 -0.1]
x = randn(2, 500)
ẋ = A * x .+ 0.02 .* randn(2, 500)
basis = CompoundBasis(polyorder = 3, trigonometric = 0)
data = TrainingData(x, ẋ)

for λ in (0.001, 0.01, 0.05, 0.2, 0.5)
    Ξ = VectorField(SINDy(; λ), basis, data).coefficients
    println("λ = ", rpad(λ, 6), "  active terms: ", count(!iszero, Ξ))
end
```

The count plateaus at the correct number over a broad middle range, and collapses once ``\lambda``
exceeds the true coefficients. That plateau is where to sit.

## Number of passes

`nloops` caps the thresholding iterations. Zhang and Schaeffer [ZhangSchaeffer2019](@ref) prove
termination within ``p`` steps, where ``p`` is the library size, so the cap is a safety net rather
than a tuning parameter — in practice the support stops changing within two or three passes. The
thesis reports one SINDy cycle for most of its Hamiltonian examples.

The package's test suite asserts that running with 200 loops gives bit-identical results to running
with 10, which is the practical form of the termination guarantee.

## λ cannot rescue badly scaled coefficients

A single global threshold implicitly assumes the true coefficients are of comparable magnitude.
When they are not, no value of ``\lambda`` works: any threshold that removes noise also removes the
genuinely small terms. This is not a tuning problem and is not fixable by sweeping — see
[When It Fails](@ref) for the worked case.
