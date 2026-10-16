# Known issues

What is known to be wrong in SparseIdentification.jl and is not fixed yet. An entry leaves this
file when its fix merges, and the CHANGELOG entry of the fix names its ID.

### K1 · The autoencoder variant of the method is not implemented.

- location: —
- evidence: Nigel Khan's thesis describes both a *Hamiltonian-SINDy* and an
  *Auto-Encoder-Hamiltonian-SINDy* algorithm, the latter identifying canonical conjugate
  coordinates alongside the dynamics. Only the first exists here. The file that was to become it
  never worked and has been removed.
- kind: defect
- found: 2026-09-01

### K2 · A norm of a difference of position vectors is not expressible.

- location: —
- evidence: `Differences` forms scalar differences `zᵢ - zⱼ`, which is exactly right in one
  spatial dimension. A three-dimensional N-body problem needs `1/‖𝐪ᵢ - 𝐪ⱼ‖`, a norm over a block
  of components, which needs a block structure the current argument selection does not carry.
- kind: defect
- found: 2026-09-02

### K3 · Matching the vector field directly is not implemented yet.

- location: —
- evidence: `J∇H` is linear in the coefficients, so fitting against measured `ż` is an ordinary
  linear sparse regression — far cheaper than the flow-map fit, which needs an optimiser. Only the
  flow-map form exists.
- kind: defect
- found: 2026-09-01

### K4 · The implicit midpoint step in the flow-map loss uses a fixed four Picard iterations

- location: —
- evidence: rather than a convergence test, so the step it computes is not the implicit midpoint
  step to any stated tolerance.
- kind: defect
- found: 2026-09-01

### K5 · The flow-map loss allocates three vectors per snapshot per evaluation.

- location: —
- evidence: `loss_kernel` builds its midpoint, iterate and gradient buffers on every call, so an
  optimiser run costs `3 × nsamples × niterations` allocations. They cannot simply be hoisted:
  their element type follows the coefficients, which the optimiser passes as dual numbers, so a
  fix needs buffers keyed on that type. This is the dominant allocation site left in the package.
- kind: defect
- found: 2026-09-02

### K6 · `SINDyVectorField` allocates 320 B per right-hand-side call

- location: `test/`
- evidence: — 160 B for the library row `evaluate` returns and 192 B for
  `yPool * coefficients` — against 0 B for `HamiltonianSINDyVectorField`. This is the
  `ODEProblem(result, …) → integrate` path, so it is the hot loop for anyone integrating an
  identified system. `evaluate` is also not inferable, because `EVALUATOR_CACHE`'s value type is
  `Any`; the function barrier in `_tabulate` keeps the batch path fast, so the cost falls on the
  single-state path alone. Nothing in `test/` pins either figure.
- kind: defect
- found: 2026-09-03

### K7 · The scripts in `scripts/` have not been ported

- location: `scripts/`
- evidence: and still call the old API and `Plots`.
- kind: defect
- found: 2026-09-01

### K9 · Four of the eight docstring doctests draw random numbers without a seed.

- location: `src/methods/method.jl:85`, `src/methods/sindy.jl:21`, `src/trainingdata.jl:35`,
  `src/trainingdata.jl:68`
- evidence: these four `jldoctest` blocks call `randn` with no `Random.seed!`. Their printed output
  is a size, a term count or a `Bool`, never a drawn value, so they pass; an output that shows a
  drawn value is not reproducible.
- kind: missing test
- found: 2026-09-26

### K10 · `TrajectoryData`'s `Δt` is not the step the fit integrates with.

- location: `src/methods/hamiltonian_sindy.jl:154`
- evidence: `grep -n 'integrator_timestep\|problem\.Δt' src/methods/hamiltonian_sindy.jl` prints
  five lines, the field in four places and its one use inside the loss —
  `3:    HamiltonianSINDy(; λ, polyorder, trigonometric, integrator_timestep, nloops, picard_iterations)`,
  `37:    integrator_timestep::T`,
  `44:            integrator_timestep::T = DEFAULT_INTEGRATOR_TIMESTEP,`,
  `48:        new{T, BT}(basis, λ, integrator_timestep, nloops, picard_iterations)` and
  `154:        mapreduce(z -> loss_kernel(z..., a, method.integrator_timestep), +, zip(xs, ys))` — and
  no line reading `problem.Δt`; `src/trainingdata.jl:77` declares that field, and
  `src/trainingdata.jl:130` is `GeometricBase.timestep(problem::TrajectoryData) = problem.Δt`. The
  fit therefore runs at the method's step while `timestep` reports the field it ignores: on four
  states, `timestep(TrajectoryData(x, y, 0.02))` prints `0.02` and
  `HamiltonianSINDy(integrator_timestep = 0.01, polyorder = 2).integrator_timestep` prints `0.01`,
  while `identify` of the one under the other prints nothing and returns a result that carries no
  step of its own — `timestep` of it is a `MethodError`. Nothing in the package compares the two. A
  caller who reaches the fit through an entry point that checks the step is protected; nothing else
  is.
- kind: defect
- found: 2026-10-11

### K11 · `_check_timestep`'s tolerance is not inferable.

- location: `src/reducedcomplexitymodeling.jl:136`
- evidence: `rtol = √eps(float(mapreduce(typeof, promote_type, steps)))` computes a type from
  runtime values, so `rtol` is `Any` and the closure it feeds is not inferable wherever `steps` has
  an abstract element type. On an eleven-point time vector (ten steps), measured after a warm call
  through the one-argument barrier `g(A) = (SparseIdentification._check_timestep(A, M); nothing)`,
  `@allocated g(t)` prints `144 B` on `Vector{Float64}`, `96 B` on `Vector{Float32}` and `800 B` on
  `Vector{AbstractFloat}`, `Vector{Any}` and `Vector{Real}`, while on the same five
  `Base.return_types(SparseIdentification._check_timestep, (typeof(t), typeof(M)))` prints
  `Any[Nothing]` — the local and its closure are not inferable, the return is. This does not bind
  §L, which binds a method a time step, a solver iteration or a right-hand-side evaluation calls;
  `_check_timestep` runs once, on about ten steps.
- kind: defect
- found: 2026-10-16

### K12 · `_check_timestep` materialises the step vector it only traverses.

- location: `src/reducedcomplexitymodeling.jl:135`
- evidence: `steps = diff(time)` builds all `N - 1` differences eagerly although they are traversed
  only for a reduction and a comparison — `mapreduce(typeof, promote_type, steps)`, `all(…, steps)`
  and, on the failing path, the `minimum` and `maximum` of the message. Nothing needs the vector;
  an iterator over the differences would serve both.
- kind: defect
- found: 2026-10-16

### K13 · The comment over `_state_pairs` describes the flattening wrongly.

- location: `src/reducedcomplexitymodeling.jl:83`
- evidence: the comment at `:83-86` says that flattening RCM's `d×M×N` ensemble column-major "would
  pair every member's last state with the next member's first — `M - 1` pairs whose two states are
  not a step of anything apart". A probe on a `2×2×3` array, `A = reshape(collect(1:12), 2, 2, 3)`,
  prints the flattened states `(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)` while the member
  trajectories are `(1, 2), (5, 6), (9, 10)` and `(3, 4), (7, 8), (11, 12)`: flattening the two
  axes first interleaves the members at each time, so the pairs it adds are not the `M - 1`
  end-to-start ones the comment names. The code at `:88` is right; the comment is not.
- kind: docs
- found: 2026-10-16
