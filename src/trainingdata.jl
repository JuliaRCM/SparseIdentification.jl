
"""
    IdentificationProblem

Supertype of the data an identification method is applied to.

An identification problem plays the role that a `GeometricEquations` problem plays for an
integrator: it is the thing a method is applied to. [`identify`](@ref) is the verb, so that

```julia
identify(problem, method)
```

reads as `GeometricIntegrators`' `integrate(problem, method)` does.

Two concrete problems, distinguished by what the data *means* rather than by its shape:
[`TrainingData`](@ref) for matching a vector field, [`TrajectoryData`](@ref) for matching a flow
map.
"""
abstract type IdentificationProblem <: GeometricBase.AbstractProblem end

"""
    TrainingData(x, ẋ)
    TrainingData(solution::GeometricSolution)

States and the corresponding time derivatives, for methods that match a vector field.

`x` and `ẋ` are either matrices whose columns are snapshots, or vectors of state vectors. Both must
describe the same number of snapshots of the same dimension; that is checked at construction,
because the alternative is a shape error surfacing much later as a wrong answer.

# Examples

```jldoctest
julia> data = TrainingData(randn(2, 20), randn(2, 20));

julia> nsamples(data), statedimension(data)
(20, 2)
```
"""
struct TrainingData{XT, DT} <: IdentificationProblem
    x::XT
    ẋ::DT

    function TrainingData(x::XT, ẋ::DT) where {XT, DT}
        _check_matching_shapes(x, ẋ, "x", "ẋ")
        new{XT, DT}(x, ẋ)
    end
end

"""
    TrajectoryData(x, y, Δt)
    TrajectoryData(solution::GeometricSolution)

Consecutive states, for methods that match a flow map: `y[j]` is the state one step of size `Δt`
after `x[j]`.

Use this where the derivatives are not available and only sampled trajectories are — matching the
flow map avoids differentiating the data, at the cost of a nonlinear regression.

# Examples

```jldoctest
julia> data = TrajectoryData([randn(2) for _ in 1:8], [randn(2) for _ in 1:8], 0.01);

julia> nsamples(data), statedimension(data), timestep(data)
(8, 2, 0.01)
```
"""
struct TrajectoryData{XT, YT, TT <: Number} <: IdentificationProblem
    x::XT
    y::YT
    Δt::TT

    function TrajectoryData(x::XT, y::YT, Δt::TT) where {XT, YT, TT <: Number}
        _check_matching_shapes(x, y, "x", "y")
        Δt > 0 || throw(ArgumentError("time step Δt must be positive, got $Δt"))
        new{XT, YT, TT}(x, y, Δt)
    end
end

function _check_matching_shapes(a::AbstractMatrix, b::AbstractMatrix, na, nb)
    size(a) == size(b) ||
        throw(DimensionMismatch("$na has size $(size(a)) but $nb has size $(size(b))"))
end

function _check_matching_shapes(a::AbstractVector{<:AbstractVector},
        b::AbstractVector{<:AbstractVector}, na, nb)
    length(a) == length(b) ||
        throw(DimensionMismatch("$na has $(length(a)) snapshots but $nb has $(length(b))"))
    for (j, (aⱼ, bⱼ)) in enumerate(zip(a, b))
        length(aⱼ) == length(bⱼ) || throw(DimensionMismatch(
            "$na[$j] has length $(length(aⱼ)) but $nb[$j] has length $(length(bⱼ))"))
    end
end

function _check_matching_shapes(a, b, na, nb)
    throw(ArgumentError("$na and $nb must both be matrices of snapshots or both be vectors of " *
                        "state vectors, got $(typeof(a)) and $(typeof(b))"))
end

# ── the ecosystem's accessors ────────────────────────────────────────────────────────────────

"""
    nsamples(problem)

The number of snapshots in an [`IdentificationProblem`](@ref).
"""
GeometricBase.nsamples(problem::TrainingData) = _nsamples(problem.x)
GeometricBase.nsamples(problem::TrajectoryData) = _nsamples(problem.x)

_nsamples(x::AbstractMatrix) = size(x, 2)
_nsamples(x::AbstractVector{<:AbstractVector}) = length(x)

"""
    statedimension(problem)

The dimension of a single state.
"""
statedimension(problem::TrainingData) = _statedimension(problem.x)
statedimension(problem::TrajectoryData) = _statedimension(problem.x)

_statedimension(x::AbstractMatrix) = size(x, 1)
_statedimension(x::AbstractVector{<:AbstractVector}) = length(first(x))

GeometricBase.timestep(problem::TrajectoryData) = problem.Δt

GeometricBase.datatype(problem::TrainingData) = eltype(_first_state(problem.x))
GeometricBase.datatype(problem::TrajectoryData) = eltype(_first_state(problem.x))

_first_state(x::AbstractMatrix) = x
_first_state(x::AbstractVector{<:AbstractVector}) = first(x)

GeometricBase.arrtype(problem::TrainingData) = typeof(problem.x)
GeometricBase.arrtype(problem::TrajectoryData) = typeof(problem.x)

"""
    states(problem)

The state snapshots, as a vector of state vectors.
"""
states(problem::TrainingData) = _as_vector_of_states(problem.x)
states(problem::TrajectoryData) = _as_vector_of_states(problem.x)

_as_vector_of_states(x::AbstractMatrix) = [x[:, j] for j in axes(x, 2)]
_as_vector_of_states(x::AbstractVector{<:AbstractVector}) = x

function Base.show(io::IO, problem::TrainingData)
    print(io, "TrainingData with ", nsamples(problem), " snapshots of dimension ",
        statedimension(problem))
end

function Base.show(io::IO, problem::TrajectoryData)
    print(io, "TrajectoryData with ", nsamples(problem), " state pairs of dimension ",
        statedimension(problem), " at Δt = ", problem.Δt)
end
