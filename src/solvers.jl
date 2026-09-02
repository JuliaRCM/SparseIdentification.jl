
"""
    JuliaLeastSquare()

Solve the regression `Θ x = ẋ` by the ordinary least-squares solution `Θ \\ ẋ`.

This is the right solver whenever the model is linear in the coefficients, which is the case for
every method in this package that fits a fixed library of candidate functions.
"""
struct JuliaLeastSquare <: AbstractSolver end

solve(Θ, ẋ, ::JuliaLeastSquare) = Θ \ ẋ

"""
    OptimizerSolver(; algorithm = BFGS(), linesearch = Backtracking())

Minimise a nonlinear least-squares loss with [`GeometricOptimizers`](https://github.com/JuliaGNI/GeometricOptimizers.jl).

Only needed where the model is *not* linear in its coefficients — in this package that is the
flow-map formulation of [`HamiltonianSINDy`](@ref), where the coefficients enter through an
integrator. Prefer [`JuliaLeastSquare`](@ref) wherever the problem is linear.
"""
struct OptimizerSolver{AT, LT} <: AbstractSolver
    algorithm::AT
    linesearch::LT

    function OptimizerSolver(; algorithm = BFGS(), linesearch = Backtracking())
        new{typeof(algorithm), typeof(linesearch)}(algorithm, linesearch)
    end
end

"""
    minimize(loss, x₀, solver::OptimizerSolver)

Minimise `loss` starting from `x₀`, returning the minimiser. `x₀` is not modified.
"""
function minimize(loss, x₀::AbstractVector, solver::OptimizerSolver)
    x = copy(x₀)
    state = OptimizerState(solver.algorithm, x)
    optimizer = Optimizer(x, loss; algorithm = solver.algorithm, linesearch = solver.linesearch)
    solve!(x, state, optimizer)
    return x
end

function solve(Θ, ẋ::AbstractVector, solver::OptimizerSolver)
    loss(x) = sum(abs2, ẋ .- Θ * x)
    minimize(loss, zeros(size(Θ, 2)), solver)
end

# The optimizer works on a vector, so a matrix right-hand side is solved column by column. Each
# column is an independent least-squares problem — the coefficients are not coupled across
# columns — so this is the same solution the matrix form would give, not an approximation of it.
function solve(Θ, ẋ::AbstractMatrix, solver::OptimizerSolver)
    reduce(hcat, solve(Θ, view(ẋ, :, j), solver) for j in axes(ẋ, 2))
end
