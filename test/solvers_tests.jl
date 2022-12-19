using SparseIdentification
using Test


@testset "Least Squares Solvers" begin

    n = 10
    m = 5
    
    A = rand(n,m)
    y = rand(n)
    x = A \ y
    
    @test solve(A, y, JuliaLeastSquare()) == x
    @test solve(A, y, OptimSolver()) ≈ x  atol=1E-10

end
