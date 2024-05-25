"""
A structure to hold training data.

# Fields
- `x::AT`: Initial condition data.
- `ẋ::AT`: Time derivative data.
- `y::AT`: Noisy data at the next time step.
"""
struct TrainingData{AT<:AbstractArray}
    x::AT
    ẋ::AT
    y::AT

    TrainingData(x::AT, ẋ::AT, y::AT) where {AT} = new{AT}(x, ẋ, y)
    TrainingData(x::AT, ẋ::AT) where {AT} = new{AT}(x, ẋ)
end
