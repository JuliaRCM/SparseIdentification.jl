import ReducedComplexityModeling as RCM
using ReducedComplexityModeling: IntrinsicSpace
using GeometricBase: CanonicalHamiltonianSystem, StateData

"""
    identify(problem::ReducedComplexityModeling.TrainingData, method::HamiltonianSINDy; kwargs...)

Identify the Hamiltonian of a `ReducedComplexityModeling.TrainingData{<:IntrinsicSpace, StateData,
CanonicalHamiltonianSystem}`, whose states are canonical `(q, p)`, `q` first.

The states are along the last dimension: a `d×N` matrix is one trajectory, a `d×M×N` array holds
`M` of them, and consecutive times are paired within each member. The times must be one
`method.integrator_timestep` apart, to `√eps` of their own floating type, since the fit integrates
at that step and never reads the data's. Every other `TrainingData` raises an `ArgumentError` that
names its space, data type and system type.

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
        problem::RCM.TrainingData{<:IntrinsicSpace, StateData, CanonicalHamiltonianSystem},
        method::HamiltonianSINDy; kwargs...)
    identify(_trajectory_data(problem, method), method; kwargs...)
end

function identify(::RCM.TrainingData{ST, DT, SY}, ::HamiltonianSINDy;
        kwargs...) where {ST, DT, SY}
    throw(ArgumentError(
        "HamiltonianSINDy cannot be applied to a TrainingData{$ST, $DT, $SY}: it fits the " *
        "flow map of canonical `(q, p)` states, so map the data to canonical `(q, p)` and " *
        "tag them `IntrinsicSpace`."))
end

# A `d×M×N` ensemble is paired within each member; reshaping it to `d×(M·N)` first would
# interleave the members and pair member `i` with member `i + 1` at the same time. The states are
# views of the caller's array, which the fit only reads.
function _state_pairs(data::AbstractArray{<:Any, 3})
    m, n = size(data, 2), size(data, 3)
    ([view(data, :, i, j) for i in 1:m for j in 1:(n - 1)],
        [view(data, :, i, j + 1) for i in 1:m for j in 1:(n - 1)])
end
function _state_pairs(data::AbstractMatrix)
    _state_pairs(reshape(data, size(data, 1), 1, size(data, 2)))
end
function _state_pairs(data::AbstractArray)
    throw(ArgumentError(
        "the states of a TrainingData are `d×N` or `d×M×N`, so `ndims(data)` is 2 or 3, " *
        "got $(ndims(data))"))
end

# The `TrajectoryData` the flow-map fit is applied to. Private, so that the two packages stay free
# to change their types independently.
function _trajectory_data(problem::RCM.TrainingData, method::HamiltonianSINDy)
    x, y = _state_pairs(problem.data)
    isempty(x) && throw(ArgumentError(
        "the states of shape $(size(problem.data)) form no state pair: a fit needs " *
        "at least two times and at least one member"))

    # Every interval is compared, relative to `√eps` of the step's own floating type: a time
    # vector stored as `Any[...]` has an element type with no `eps`.
    Δt = method.integrator_timestep
    steps = diff(problem.time)
    all(s -> isapprox(s, Δt; rtol = √eps(float(typeof(s)))), steps) || throw(ArgumentError(
        "`diff(time)` runs from $(minimum(steps)) to $(maximum(steps)), but HamiltonianSINDy " *
        "integrates at `integrator_timestep = $Δt` and never reads the step of the data"))

    TrajectoryData(x, y, Δt)
end
