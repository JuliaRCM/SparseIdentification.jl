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

- **The package now follows the JuliaGNI API.** Identification mirrors integration:

  ```julia
  result = identify(problem, method)      # was: VectorField(method, basis, data)
  ```

  `TrainingData` and `TrajectoryData` are `GeometricBase.AbstractProblem`s and the methods are
  `GeometricBase.AbstractMethod`s, so the ecosystem's accessors work on them — `nsamples`,
  `timestep`, `datatype`, `arrtype`, `parameters`, `functions`, `basis`, `name`, `description`,
  `reference`. The basis moved into the method, `SINDy(basis; λ)`, which is what the source's own
  `# TODO: Add basis as field of SINDy method` asked for. Results are typed (`SINDyResult`,
  `HamiltonianSINDyResult`) rather than bare coefficient arrays.

- **The loop closes.** An identified system converts to a `GeometricEquations` problem —
  `ODEProblem(result, timespan, timestep, ics...)` and
  `HODEProblem(result, timespan, timestep, q₀, p₀)` — so it integrates with `GeometricIntegrators`.
  Conversely `TrainingData(solution)` and `TrajectoryData(solution)` build training data from a
  `GeometricSolution`. These extend the ecosystem's own constructors rather than inventing names,
  as `EulerLagrange` does. The Hamiltonian side generates `v = ∂H/∂p` and `f = -∂H/∂q` separately,
  which is what a `HODEProblem` needs, alongside the combined field the regression uses.

- **`issymplectic` and `isenergypreserving` now state the difference between the methods** as a
  property rather than as prose: both are `true` for `HamiltonianSINDy` and `false` for `SINDy`.
  Those two, with `isexplicit`, `isimplicit`, `issymmetric` and `isstifflyaccurate`, are extended
  but **not exported** — `GeometricIntegratorsBase` defines its own generics of those six names
  instead of extending `GeometricBase`'s stubs, so exporting them would make
  `using SparseIdentification` alongside `using GeometricIntegrators` resolve the name to neither.
  Reach them qualified.

- **`src/lorenz.jl` is gone.** `GeometricProblems.LorenzAttractor` covers it; a package in this
  ecosystem should not carry its own copy of a standard test problem.

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

- **Documentation.** Theory (Hamiltonian mechanics and the symplectic form, the SINDy formulation
  and what STLSQ actually guarantees, the Hamiltonian extension), usage (getting started, basis
  libraries, choosing `λ`), four worked examples, and a page on failure modes. Every code block in
  it executes during the build, so an example that stops working fails the docs job.

- **`scripts/verify_thesis_examples.jl`.** Claims taken from Khan's thesis are checked rather than
  transcribed. It confirms five and finds three that do not hold as printed:

  - **Eq. (4.2)**, the nonlinear oscillator, prints `½p₁² + ½p₁²` — the same term twice, with `p₂`
    absent. As printed `q̇₂ = ∂H/∂p₂ = 0`, so the second degree of freedom of a system the text
    describes as two-dimensional never moves. It must read `½p₁² + ½p₂²`.
  - **Eq. (4.4)**, the point vortex, sums over all `i` and `j` including `i = j`, where
    `log|qᵢ - qⱼ| = log 0`. Worse, it identifies `p` with the vortex strength, which makes
    `ṗ = -∂H/∂q ≠ 0` — vortex strengths are constants of the motion. The conjugate pair is the two
    spatial coordinates of each vortex, with the strengths as fixed parameters.
  - The **magnitudes quoted for the N-body conditioning argument** are `1e-24` and `1e37`;
    recomputed they are `1e-25` and `1e45`. The argument survives — it rests on the ratio, which is
    70 orders of magnitude and larger than claimed — but the numbers as printed do not reproduce.

  It also confirms the two facts the package's design rests on: `J∇H` is exactly linear in the
  coefficients, and the identified field is Hamiltonian for *any* coefficients, fitted or not.

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

- **The scripts in `scripts/` have not been ported** and still call the old API and `Plots`.
