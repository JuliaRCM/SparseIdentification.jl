
"""
    TrainingData(x, ẋ)

States and the corresponding time derivatives, for methods that match a vector field.

`x` and `ẋ` are either matrices whose columns are snapshots, or vectors of state vectors. Both
must describe the same number of snapshots of the same dimension; that is checked here, because
the alternative is a shape error surfacing much later as a wrong answer.
"""
struct TrainingData{XT, DT}
    x::XT
    ẋ::DT

    function TrainingData(x::XT, ẋ::DT) where {XT, DT}
        _check_matching_shapes(x, ẋ, "x", "ẋ")
        new{XT, DT}(x, ẋ)
    end
end

"""
    TrajectoryData(x, y, Δt)

Consecutive states, for methods that match a flow map: `y[j]` is the state one step of size `Δt`
after `x[j]`.

Use this where the derivatives are not available and only sampled trajectories are — matching the
flow map avoids differentiating the data, at the cost of a nonlinear regression.
"""
struct TrajectoryData{XT, YT, TT <: Number}
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
        length(aⱼ) == length(bⱼ) ||
            throw(DimensionMismatch("$na[$j] has length $(length(aⱼ)) but $nb[$j] has length $(length(bⱼ))"))
    end
end

function _check_matching_shapes(a, b, na, nb)
    throw(ArgumentError("$na and $nb must both be matrices of snapshots or both be vectors of " *
                        "state vectors, got $(typeof(a)) and $(typeof(b))"))
end

"""
    nsnapshots(data)

The number of snapshots in `data`.
"""
nsnapshots(data::TrainingData) = _nsnapshots(data.x)
nsnapshots(data::TrajectoryData) = _nsnapshots(data.x)

_nsnapshots(x::AbstractMatrix) = size(x, 2)
_nsnapshots(x::AbstractVector{<:AbstractVector}) = length(x)

"""
    statedimension(data)

The dimension of a single state in `data`.
"""
statedimension(data::TrainingData) = _statedimension(data.x)
statedimension(data::TrajectoryData) = _statedimension(data.x)

_statedimension(x::AbstractMatrix) = size(x, 1)
_statedimension(x::AbstractVector{<:AbstractVector}) = length(first(x))
