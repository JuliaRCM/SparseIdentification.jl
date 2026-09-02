
# ─────────────────────────────────────────────────────────────────────────────────────────────
# Bridges into and out of the JuliaGNI ecosystem.
#
# Out: an identified system becomes a `GeometricEquations` problem, so it integrates with
#      `GeometricIntegrators` and feeds everything downstream of that.
# In:  a `GeometricSolution` becomes training data, so the loop
#      `problem → integrate → identify → problem` closes.
#
# These add methods to `GeometricEquations.ODEProblem` and `HODEProblem` rather than inventing
# new constructor names, which is what `EulerLagrange` does for the same reason: a caller who
# knows the ecosystem already knows these names.
# ─────────────────────────────────────────────────────────────────────────────────────────────

"""
    ODEProblem(result::SINDyResult, timespan, timestep, ics...; kwargs...)

The identified system as a `GeometricEquations.ODEProblem`, ready for `integrate`.

# Examples

```julia
result = identify(TrainingData(x, ẋ), SINDy(CompoundBasis(polyorder = 3)))
prob   = ODEProblem(result, (0.0, 10.0), 0.01, x₀)

using GeometricIntegrators
sol = integrate(prob, ExplicitMidpoint())
```
"""
function GeometricEquations.ODEProblem(result::SINDyResult, timespan, timestep, ics...;
        kwargs...)
    GeometricEquations.ODEProblem(
        SINDyVectorField(result), timespan, timestep, ics...; kwargs...)
end

"""
    HODEProblem(result::HamiltonianSINDyResult, timespan, timestep, q₀, p₀; kwargs...)

The identified Hamiltonian system as a `GeometricEquations.HODEProblem`, ready for `integrate`
with a symplectic integrator.

The problem carries the identified Hamiltonian, so the energy behaviour of the integrated solution
can be checked directly with `GeometricSolutions.compute_invariant_error`.

# Examples

```julia
result = identify(TrajectoryData(x, y, Δt), HamiltonianSINDy(λ = 0.05, polyorder = 2))
prob   = HODEProblem(result, (0.0, 10.0), 0.01, q₀, p₀)

using GeometricIntegrators
sol = integrate(prob, ImplicitMidpoint())
```
"""
function GeometricEquations.HODEProblem(result::HamiltonianSINDyResult, timespan, timestep,
        ics...; parameters = (a = result.coefficients,), kwargs...)
    funs = result.hamiltonian
    GeometricEquations.HODEProblem(funs.v, funs.f, funs.H, timespan, timestep, ics...;
        parameters = parameters, kwargs...)
end

"""
    TrainingData(solution::GeometricSolution)

Training data from an integrated solution, pairing each stored state with the vector field
evaluated there.

This closes the loop: integrate a known problem, identify it from the solution, and compare.
"""
function TrainingData(solution::GeometricSolutions.GeometricSolution)
    prob = solution.problem
    funs = GeometricEquations.functions(prob)
    params = GeometricEquations.parameters(prob)

    # A solution's data series are `OffsetArray`s indexed from 0. `collect` on the index range
    # gives a plain 1-based vector of those indices, which is what the comprehensions below index
    # with; a comprehension over the range itself would produce another OffsetVector.
    idx = collect(eachindex(solution.q))
    ts = [solution.t[n] for n in idx]

    if _ispartitioned(solution)
        # A partitioned (Hamiltonian) solution stores q and p separately; the state is z = (q, p)
        # and the vector field is assembled from v = q̇ and f = ṗ.
        qs = [collect(solution.q[n]) for n in idx]
        ps = [collect(solution.p[n]) for n in idx]
        zs = [vcat(q, p) for (q, p) in zip(qs, ps)]

        żs = map(zip(ts, qs, ps)) do (t, q, p)
            v = zero(q)
            f = zero(p)
            funs.v(v, t, q, p, params)
            funs.f(f, t, q, p, params)
            vcat(v, f)
        end

        return TrainingData(zs, żs)
    end

    qs = [collect(solution.q[n]) for n in idx]
    ẋ = map(zip(ts, qs)) do (t, q)
        out = zero(q)
        funs.v(out, t, q, params)
        out
    end

    TrainingData(qs, ẋ)
end

"""
    _ispartitioned(solution)

Whether `solution` stores a conjugate momentum alongside the position.

`GeometricSolutions` defines `hasproperty` on the type of the data series, so this is answered at
compile time rather than by probing the object.
"""
_ispartitioned(solution) = hasproperty(solution, :p)

"""
    TrajectoryData(solution::GeometricSolution)

Consecutive states from an integrated solution, for flow-map matching.

Pairs each stored state with its successor, taking `Δt` from the solution's own time step.
"""
function TrajectoryData(solution::GeometricSolutions.GeometricSolution)
    # `collect` the 0-based index range into a plain 1-based vector before comprehending over it
    # — otherwise the result is an OffsetVector and the slicing below loses a state.
    idx = collect(eachindex(solution.q))
    length(idx) ≥ 2 ||
        throw(ArgumentError("need at least two stored states to form a trajectory pair"))

    ts = [solution.t[n] for n in idx]

    # For a partitioned (Hamiltonian) solution the state is z = (q, p), not q alone — taking only
    # the positions would hand a Hamiltonian method half a phase space.
    zs = if _ispartitioned(solution)
        [vcat(collect(solution.q[n]), collect(solution.p[n])) for n in idx]
    else
        [collect(solution.q[n]) for n in idx]
    end

    TrajectoryData(zs[1:(end - 1)], zs[2:end], ts[2] - ts[1])
end
