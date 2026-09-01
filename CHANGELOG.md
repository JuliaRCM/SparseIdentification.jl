# Release Notes

All notable changes to SparseIdentification.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually
changed, so that a compat-only bump can be told apart from a rename or a change in results.

This file was started on 2026-08-31 and deliberately holds no entries for anything before it.
Nothing has been released yet — there are no tags, and `Project.toml` stands at `0.1.0` — so the
development history that predates this file is in `git log` alone. It is named as a gap rather
than reconstructed, because a changelog assembled after the fact loses exactly the reasoning that
makes it worth keeping.

## [Unreleased] — targeting 0.1.0

### Bug Fixes

- **The package loads again.** The blocker was not what was recorded here previously: the
  `PreallocationTools`/`ForwardDiff` extension and `LinearSolve`'s `MKL_jll` reach are real
  upstream bugs, but the load actually aborted on `SimpleNonlinearSolve` 2.7.0 failing to
  precompile on Julia 1.13. All of it arrived through `DifferentialEquations`, which is no longer
  a dependency, so the whole class of failure is gone rather than worked around.

- **`hamilGrad_func_builder` could never have run.** It wrapped `build_function` in
  `Symbolics.inject_registered_module_functions`, a name that exists in *no* released Symbolics —
  not in the pinned 5.36.0 and not in 7.x. Every Hamiltonian identification would have hit an
  `UndefVarError` the moment it was reached. The wrapper is gone; the builder is now exercised by
  the test suite.

- **Sequentially thresholded regression discarded every refit after the first.**
  `coeffs[biginds]` is a copy, so `b .= result.minimizer` wrote into the copy and the coefficient
  vector never saw it. The loop therefore only zeroed small coefficients and kept the values from
  the initial dense fit — precisely what the thresholding loop exists to correct.

- **The Hamiltonian basis and the optimiser disagreed on how many coefficients there are.**
  `calculate_nparams` takes the number of degrees of freedom `d` and works in `2d` phase-space
  variables; the regression passed it the full state dimension instead. For two degrees of freedom
  with cubic and trigonometric terms the optimiser searched **212** coefficients for a basis that
  reads **58**, so 154 of them were inert and the thresholding statistics were computed over them.

- **`TrigonometricBasis` could not be evaluated.** It built a two-element `Vector` of matrices and
  `hcat`ed it onto a matrix, and it read its input in the opposite orientation to
  `PolynomialBasis`, so it could neither run alone nor be combined with polynomials.

- **`TrainingData` had no constructor.** A third field had been added without one, so all but one
  of the scripts in `scripts/` were calling a two-argument form that no longer existed. Shapes are
  now validated at construction.

- **Identification is deterministic.** `sparsify` added fresh `randn` noise to its own targets on
  every call, so two identical calls disagreed and the noise could not be switched off. Noise is a
  property of the data, not of the estimator.

- `sparsify_hamiltonian_dynamics` was exported but never defined anywhere.

### Breaking Changes

- **Minimum Julia is now 1.11**, raised from 1.10 because `SimpleSolvers` requires it. This is the
  second floor in use across the tree, for dependencies that need it.

- **`DifferentialEquations`, `ODE`, `Optim`, `Plots`, `Distributions`, `Zygote`, `ThreadsX`,
  `ParallelUtilities`, `Distances` and `DelimitedFiles` are no longer dependencies.** Optimisation
  moves to `GeometricOptimizers` and `SimpleSolvers`; `Symbolics` goes from 5 to 7. Six of those
  ten were never used by any code that ran, and three existed only for a `pmapreduce` call that
  was commented out. Loading the package went from pulling the entire SciML stack to about eleven
  seconds.

- **`OptimSolver` is now `OptimizerSolver`**, since "Optim" no longer names anything the package
  uses. `solve` is now a method on `SimpleSolvers.solve` rather than a separate function.

- **`SINDy(lambda = …, noise_level = …)` is now `SINDy(λ = …)`.** The noise level is gone with the
  noise injection.

- **`HamiltonianSINDy` no longer takes the analytical vector field**, which was a mandatory
  positional argument — the method required the very thing it was supposed to identify. It takes
  `TrajectoryData`, and the regression targets come from data.

- **`TrainingData` is split by meaning**: `TrainingData(x, ẋ)` for matching a vector field,
  `TrajectoryData(x, y, Δt)` for matching a flow map. One struct previously served both with a
  field whose meaning depended on the method and which nothing validated.

- **Removed**: `poolDataLIST` (built equation labels by string concatenation and wrote to `stdout`;
  symbolic printing replaces it), `hamiltonian_basis_maker.jl` (duplicated `hamiltonian_poly` and
  hardcoded a size formula valid only at order 3), `hamiltonianGenerator.jl` and
  `sparsify_dynamics.jl` (dead), `autoencoder.jl` (syntactically invalid, and using a Flux API
  removed in 2019), and two notebooks duplicating the same code.

### New Features

- **A test suite.** It previously checked `A \ y` two ways and nothing else — no basis, no SINDy,
  no Hamiltonian path — which is why none of the defects above were caught. It now covers basis
  evaluation against closed forms, `TrainingData` validation, both solvers, and Aqua. Each bug
  above has a regression test.

- **Recovery tests against known coefficients**: the linear 2D oscillator
  (`ẋ = -0.1x + 2y`, `ẏ = -2x - 0.1y`) and Lorenz-63 (`σ = 10`, `ρ = 28`, `β = 8/3`) from Brunton,
  Proctor & Kutz (2016), both recovered to `1e-10` on clean data with every other coefficient
  exactly zero; and the harmonic oscillator recovered through the Hamiltonian path.

## Open Issues

- **The autoencoder variant of the method is not implemented.** Nigel Khan's thesis describes both
  a *Hamiltonian-SINDy* and an *Auto-Encoder-Hamiltonian-SINDy* algorithm, the latter identifying
  canonical conjugate coordinates alongside the dynamics. Only the first exists here. The file that
  was to become it never worked and has been removed.

- **Matching the vector field directly is not implemented yet.** `J∇H` is linear in the
  coefficients, so fitting against measured `ż` is an ordinary linear sparse regression — far
  cheaper than the flow-map fit, which needs an optimiser. Only the flow-map form exists.

- **The implicit midpoint step in the flow-map loss uses a fixed four Picard iterations** rather
  than a convergence test, so the step it computes is not the implicit midpoint step to any stated
  tolerance.

- **The package does not yet follow the JuliaGNI API.** `SparsificationMethod` is a free-standing
  abstract type rather than a `GeometricBase.AbstractMethod`, the accessor verbs are not extended,
  and an identified system does not convert to a `GeometricEquations` problem.

- **The scripts in `scripts/` have not been ported** and still call the old API and `Plots`.
