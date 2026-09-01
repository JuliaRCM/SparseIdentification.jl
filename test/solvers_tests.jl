using SparseIdentification
using Test

# `OptimizerSolver` is compared at 1e-6, not at the 1e-12 the direct solve reaches. Its
# stopping criterion is on the gradient, so it bounds the residual and not the distance to the
# minimiser; on these problems that lands a few times 1e-9 away. The test asserts the two solvers
# agree, which is the property — a tighter number would only be pinning one run's luck.
const OPTIMIZER_ATOL = 1e-6

@testset "Least squares solvers" begin
    n = 10
    m = 5

    A = rand(n, m)
    y = rand(n)
    x = A \ y

    @test solve(A, y, JuliaLeastSquare()) == x
    @test solve(A, y, OptimizerSolver())≈x atol=OPTIMIZER_ATOL
end

@testset "Matrix right-hand side" begin
    A = rand(12, 4)
    Y = rand(12, 3)

    @test solve(A, Y, JuliaLeastSquare()) ≈ A \ Y
    @test solve(A, Y, OptimizerSolver())≈A \ Y atol=OPTIMIZER_ATOL
end

@testset "minimize" begin
    # `minimize` must not modify the starting point it is handed.
    f(x) = sum(abs2, x .- [1.0, -2.0, 0.5])
    x₀ = zeros(3)
    x = minimize(f, x₀, OptimizerSolver())

    @test x≈[1.0, -2.0, 0.5] atol=1e-6
    @test x₀ == zeros(3)
end
