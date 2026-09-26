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
