
struct TrainingData{AT<:AbstractArray}
    x::AT # initial condition
    ẋ::AT # initial condition
    y::AT # noisy data at next time step
end

