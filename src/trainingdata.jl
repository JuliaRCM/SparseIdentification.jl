
struct TrainingData{AT<:AbstractArray}
    x::AT # initial condition
    ẋ::AT # initial condition
    y::AT # noisy data at next time step

    TrainingData(x::AT, ẋ::AT, y::AT) where {AT} = new{AT}(x, ẋ, y)
    TrainingData(x::AT, ẋ::AT) where {AT} = new{AT}(x, ẋ)
end
