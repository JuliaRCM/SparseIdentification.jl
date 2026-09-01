# When It Fails

A method's failure modes are more useful than its successes, because they tell you whether to reach
for it at all. This page collects the ones that are understood, with the evidence.

## Badly scaled coefficients defeat any threshold

This is the sharpest limitation, and the one worth checking before anything else.

The ``N``-body Hamiltonian is

```math
H = \sum_i \frac{p_i^2}{2 m_i} \;-\; \sum_{i<j} \frac{G m_i m_j}{\lVert q_j - q_i \rVert} ,
```

which contains two families of coefficients: ``1/(2m_i)`` on the kinetic terms and ``G m_i m_j`` on
the potential terms. For an Earth–Sun system, with ``G = 6.6743 \times 10^{-11}``,
``m_\oplus = 5.972 \times 10^{24}\,\mathrm{kg}`` and ``m_\odot = 1.989 \times 10^{30}\,\mathrm{kg}``
(computed in `scripts/verify_thesis_examples.jl`):

```
1/(2m_earth)    = 8.372e-26   → order 1e-25
G m_earth m_sun = 7.928e+44   → order 1e45
ratio           = 9.469e+69   → 70 orders of magnitude apart
```

**Seventy orders of magnitude.** Sparsification rests on a single global ``\lambda`` separating
"real" coefficients from "noise" ones. No scalar can sit between ``10^{-25}`` and ``10^{45}`` in a
way that keeps both: any ``\lambda`` large enough to denoise the potential terms annihilates the
kinetic ones, and any ``\lambda`` small enough to keep the kinetic terms keeps everything.

Non-dimensionalising does not save it. Scaling the masses shifts both families together; the
*ratio* is what defeats the threshold, and the ratio is scale-invariant.

The thesis reports exactly this outcome for its solar-system example: the optimiser found the
momentum terms correctly and zeroed the position terms, so positions integrated correctly and
momenta were completely wrong. It took 3651 seconds to reach that answer.

!!! note "A correction"
    The thesis quotes these magnitudes as "on the order of ``10^{-24}``" and "on the order of
    ``10^{37}``". Recomputed, they are ``10^{-25}`` and ``10^{45}``. The discrepancy does not
    affect the argument — the conclusion follows from the ratio, which is larger than stated, not
    smaller — but the numbers as printed do not reproduce.

**What to do instead.** Rescale variables so the expected coefficients are of comparable size, if
you can; identify subsystems separately; or use a formulation where the disparity does not appear.
The thesis suggests the Newtonian or Lagrangian formulations, whose equations of motion group the
terms differently.

## The answer must be in the library

SINDy selects from candidates; it does not invent them. A polynomial library cannot represent
``\cos(q)``, ``\log \lVert q_i - q_j \rVert`` or ``1/\lVert q_i - q_j \rVert``, and no amount of
data or tuning will change that.

This is not a soft limitation that shows up as reduced accuracy — it shows up as a confident,
dense, wrong model, because the fit will use whatever it has to approximate what it cannot express.

The package supplies polynomial, trigonometric, exponential, logarithmic and rational bases, and —
crucially — lets each be applied to *differences* of state components rather than to components
alone, which is what interacting systems need. What remains out of reach is a **norm** of a
difference of position *vectors*, ``1/\lVert \mathbf{q}_i - \mathbf{q}_j\rVert``, so a genuinely
three-dimensional ``N``-body problem is still not expressible. See [Basis Libraries](@ref).

## Coordinates matter as much as the library

A system that is sparse in one coordinate system is generally dense in another. The original SINDy
paper's documented failure cases are both of this kind. This is the motivation for the
autoencoder variants — learning a coordinate transformation and the dynamics jointly — which the
thesis develops as *Auto-Encoder-Hamiltonian-SINDy* and which **this package does not implement**.

The thesis is candid about how hard that coupling is: because neural networks are universal
approximators, the encoder can reach coordinates in which the data reconstructs beautifully but
which are *not* canonically conjugate, and the SINDy layer then confidently identifies the wrong
basis functions while the total loss decreases. Its prototype works only when initialised at or
near the correct coordinates.

## Canonical structure is assumed

The implementation hard-codes the canonical ``J``. Systems that are Poisson but not canonically
Hamiltonian — rigid bodies, point vortices, guiding-centre motion — have a state-dependent
structure matrix ``J(z)``, and assuming a constant one identifies the wrong object.

!!! warning "The thesis's point-vortex example does not hold up"
    Its Equation (4.4) gives the point-vortex Hamiltonian as
    ``H = -\frac{1}{4\pi}\sum_{i=1}^{N}\sum_{j=1}^{N} p_i p_j \log\lvert q_i - q_j \rvert``,
    described as "``p`` represents the strength of the vorticity, and ``q`` is the distance from
    its center". Two independent problems, both checked in
    `scripts/verify_thesis_examples.jl`:

    1. The double sum runs over all ``i`` and ``j`` including ``i = j``, where
       ``\log\lvert q_i - q_i\rvert = \log 0 = -\infty``. The sum must exclude the diagonal.
    2. If ``p`` is the vortex strength, then ``\dot p = -\partial H/\partial q`` is non-zero — the
       run gives ``\dot p = [0.4341, -0.4341]`` — so the strengths would evolve in time. **Point
       vortex strengths are constants of the motion.**

    In the correct formulation the strengths ``\Gamma_i`` are fixed parameters and the conjugate
    pair is the *two spatial coordinates of each vortex*: ``\sqrt{\Gamma_i}\,x_i`` and
    ``\sqrt{\Gamma_i}\,y_i``, with
    ``H = -\frac{1}{4\pi}\sum_{i<j}\Gamma_i\Gamma_j\log\lVert r_i - r_j\rVert``.
    The results reported for this example should be read with that in mind.

## Fixed Picard iterations bound the flow-map accuracy

The flow-map formulation solves its implicit midpoint step with a fixed iteration count rather than
to a tolerance, so the step is an approximation of unstated accuracy. The thesis measures the
consequence directly: the vector-field formulation reaches coefficients accurate to about five
decimal places, the flow-map formulation about two, and it attributes the gap to the integrator and
the injected noise together. It also costs far more — 1124 seconds against 16 on the same problem.

Use the flow-map formulation when you genuinely lack derivative data, not by default.

## Sparsity is a modelling assumption

Everything above presumes the governing equations *are* sparse in the basis supplied. For turbulent
flows, strongly coupled many-body systems, or empirical models with no compact closed form, that
premise simply fails, and a method built on it will return a model that is sparse and wrong rather
than reporting that it cannot help.
