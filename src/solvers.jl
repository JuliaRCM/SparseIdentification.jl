
using Optim


abstract type AbstractSolver end

abstract type NonlinearSolver <: AbstractSolver end


struct JuliaLeastSquare <: AbstractSolver end

function solve(Θ, ẋ, ::JuliaLeastSquare)
    Θ \ ẋ
end


struct OptimSolver <: NonlinearSolver 
    method

    function OptimSolver(method = BFGS())
        new(method)
    end
end

function optimize(loss, x₀, solver::OptimSolver)
    result = Optim.optimize(loss, x₀, solver.method, Optim.Options(iterations=483))
    println(result)
    return result.minimizer
end


function solve(Θ, ẋ, solver::NonlinearSolver)
    x₀ = zeros(size(Θ,2), size(ẋ,2))
    loss(x) = mapreduce( y -> y^2, +, ẋ .- Θ * x )
    optimize(loss, x₀, solver)
end
