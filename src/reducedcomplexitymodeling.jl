# ─────────────────────────────────────────────────────────────────────────────────────────────
# Bridge into `ReducedComplexityModeling`.
#
# In:  a `TrainingData` whose space is an `IntrinsicSpace` and whose data are canonical `(q, p)`
#      states, tagged `StateData` on a `CanonicalHamiltonianSystem`, becomes the `TrajectoryData`
#      the Hamiltonian flow-map fit takes. That is the same entry point the docstring of
#      `HamiltonianSINDy` describes, reached from the data type the rest of the ecosystem uses.
# Out: nothing. This package adds no constructor to `TrainingData` and no accessor for it, so a
#      caller converts nothing by hand and there is no second public name for RCM's type.
#
# The space, the data type and the system type of a `TrainingData` are type parameters, so which
# data this package accepts is decided by dispatch and not by a check at run time: the accepted
# combination is a method, every other one reaches the refusal method below, and a combination
# added later cannot be fitted in silence because no method accepts it.
#
# The space is the load-bearing one. `ObservableSpace` marks what a measurement or a simulation
# produced, `IntrinsicSpace` the coordinates in which the structure of the system is manifest. The
# fit assumes its states are canonically conjugate and has no way to tell, so data that are not
# canonical are refused where they enter, and the map to canonical coordinates stays with whoever
# produced it.
# ─────────────────────────────────────────────────────────────────────────────────────────────

import ReducedComplexityModeling as RCM
using ReducedComplexityModeling: IntrinsicSpace

"""
    identify(problem::ReducedComplexityModeling.TrainingData, method::HamiltonianSINDy; kwargs...)

Identify the Hamiltonian of a `ReducedComplexityModeling.TrainingData` whose states are the
canonical `(q, p)` of the system it declares.

A `ReducedComplexityModeling` `TrainingData` is tagged with three type parameters, and this method
accepts one combination of them: `<:IntrinsicSpace`, `StateData`, `CanonicalHamiltonianSystem`.
Each says something the flow-map fit needs. `IntrinsicSpace` says the coordinates are the ones in
which the structure of the system is manifest; `StateData` with a canonical Hamiltonian system
says the stored variables are `q` then `p`, which is the order the fit assumes. A *subtype* of
`IntrinsicSpace` is accepted, since a package names its own kind of canonicalising map that way.

The data are the states of one or more trajectories, with time along the last dimension, as
`ReducedComplexityModeling` lays them out. A `d×N` matrix is one trajectory of `N` states and
`data[:, n]` is paired with `data[:, n+1]`; a `d×M×N` array holds `M` trajectories, `data[:, i, n]`
being member `i` at time `n`, and consecutive times are paired *within* a member — never one
member's last state with the next member's first, which is not a flow-map pair at all. Any other
number of dimensions, and fewer than two times, raise an `ArgumentError`.

The states of a pair must be one step of `method.integrator_timestep` apart, to `√eps` relative:
the flow-map loss integrates at the method's step and never reads the step of the data, so a
`TrainingData` built at another step would be fitted for an interval it does not describe.

Every other `TrainingData` — on an `ObservableSpace`, or on another data or system type — raises an
`ArgumentError` naming all three and telling the caller to map the data to canonical `(q, p)` and
tag it `IntrinsicSpace`, rather than being fitted in silence. This package never guesses that map:
it belongs to whatever produced it.

# Examples

```julia
using ReducedComplexityModeling: IntrinsicSpace, TrainingData
using GeometricBase: CanonicalHamiltonianSystem, StateData

data   = TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(time, states)
method = HamiltonianSINDy(λ = 0.05, integrator_timestep = 0.01, polyorder = 2)

result = identify(data, method)
```
"""
function identify(
        problem::RCM.TrainingData{<:IntrinsicSpace, StateData,
            CanonicalHamiltonianSystem},
        method::HamiltonianSINDy; kwargs...)
    identify(_trajectory_data(problem, method), method; kwargs...)
end

"""
    identify(problem::ReducedComplexityModeling.TrainingData, method::SparsificationMethod)

The refusal of a `ReducedComplexityModeling` `TrainingData` this package does not accept: the space
is not an `IntrinsicSpace`, the data are not `StateData`, the system is not a
`CanonicalHamiltonianSystem`, or the method is not the flow-map one.

The `ArgumentError` names the method, the space, the data type and the system type, and says what
to do instead — map the data to canonical `(q, p)` and tag them `IntrinsicSpace` — so that the
caller learns what is missing rather than reading a `MethodError` from several layers down.
"""
function identify(problem::RCM.TrainingData{ST, DT, SY}, method::SparsificationMethod;
        kwargs...) where {ST, DT, SY}
    throw(ArgumentError(
        "$(nameof(typeof(method))) cannot be applied to a TrainingData{$ST, $DT, $SY}. " *
        "HamiltonianSINDy fits the flow map of canonical `(q, p)` states, so it takes " *
        "`TrainingData{<:IntrinsicSpace, StateData, CanonicalHamiltonianSystem}` and nothing " *
        "else: map the data to canonical `(q, p)` and tag them `IntrinsicSpace`."))
end

# ── the conversion, private ──────────────────────────────────────────────────────────────────

# One trajectory as the pairs the fit matches: `y[j]` is the state one time step after `x[j]`. The
# states are views of the caller's array, which the fit only reads, so a `d×N` matrix is not copied
# to be paired with a shifted copy of itself.
function _state_pairs(data::AbstractArray{<:Any, 2})
    n = size(data, 2)
    ([view(data, :, j) for j in 1:(n - 1)], [view(data, :, j + 1) for j in 1:(n - 1)])
end

# RCM's ensemble layout: the middle axis is the member and the last one is time, so consecutive
# times are paired inside a member. Flattening the two axes first would pair every member's last
# state with the next member's first — `M - 1` pairs whose two states are not a step of anything
# apart, mixed into a fit that has no way to reject them.
function _state_pairs(data::AbstractArray{<:Any, 3})
    m, n = size(data, 2), size(data, 3)
    ([view(data, :, i, j) for i in 1:m for j in 1:(n - 1)],
        [view(data, :, i, j + 1) for i in 1:m for j in 1:(n - 1)])
end

"""
    _trajectory_data(problem, method)

The `TrajectoryData` the flow-map fit is applied to, built from the states of `problem`.

Private: this is the bridge's own conversion, not a constructor of `TrajectoryData` for RCM's type,
so the two packages stay free to change their types independently.
"""
function _trajectory_data(problem::RCM.TrainingData, method::HamiltonianSINDy)
    data = problem.data

    ndims(data) in (2, 3) || throw(ArgumentError(
        "the states of a TrainingData for $(nameof(typeof(method))) are one trajectory " *
        "(`d×N`) or several (`d×M×N`), with time along that last dimension, so `ndims(data)` " *
        "is 2 or 3, got $(ndims(data))"))

    times = size(data, ndims(data))
    times ≥ 2 || throw(ArgumentError(
        "a trajectory needs at least two times to form a state pair, this TrainingData has " *
        "$times"))

    Δt = _check_timestep(problem.time, method)

    x, y = _state_pairs(data)
    TrajectoryData(x, y, Δt)
end

# `sparsify` integrates the flow map at `method.integrator_timestep` and never reads the step of the
# data (K10), so data at another step would be fitted for the wrong interval without a word. The two
# steps are compared to `√eps(T)`: a time vector built by accumulation is at the step to round-off,
# and a different step is not at it at all. The data's step is the one reported.
function _check_timestep(time::AbstractVector{TT}, method::HamiltonianSINDy) where {TT}
    Δt = method.integrator_timestep
    steps = diff(time)
    rtol = √eps(float(TT))

    all(step -> isapprox(step, Δt; rtol = rtol), steps) || throw(ArgumentError(
        "the time steps of this TrainingData are not the step $(nameof(typeof(method))) " *
        "integrates the flow map at: `diff(time)` runs from $(minimum(steps)) to " *
        "$(maximum(steps)), while `integrator_timestep` is $Δt. The fit reads the method's " *
        "step and not the data's, so this pair would be fitted for the wrong interval."))

    return first(steps)
end
