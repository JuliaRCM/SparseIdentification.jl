using SparseIdentification
using Test

@testset "Polynomial basis" begin
    # Two degrees of freedom, degrees 0 through 3: 1 + 2 + 3 + 4 = 10 terms.
    basis = CompoundBasis(polyorder = 3, trigonometric = 0)
    x = randn(2, 7)
    Θ = evaluate(x, basis)

    @test size(Θ) == (7, 10)

    # Degree 0 is the constant column, degree 1 reproduces the state.
    @test all(Θ[:, 1] .== 1)
    @test Θ[:, 2] ≈ x[1, :]
    @test Θ[:, 3] ≈ x[2, :]

    # Degree 2 is q₁², q₁q₂, q₂² in that order.
    @test Θ[:, 4] ≈ x[1, :] .^ 2
    @test Θ[:, 5] ≈ x[1, :] .* x[2, :]
    @test Θ[:, 6] ≈ x[2, :] .^ 2
end

@testset "Trigonometric basis" begin
    # Regression test. `evaluate` used to build a two-element Vector of matrices and `hcat` it
    # onto a matrix, and it read `data` in the opposite orientation to PolynomialBasis, so it
    # could neither run on its own nor be combined with polynomials.
    basis = TrigonometricBasis(2)
    x = randn(2, 5)
    Θ = evaluate(x, basis)

    # 2 wavenumbers × {sin, cos} × 2 degrees of freedom
    @test size(Θ) == (5, 8)

    @test Θ[:, 1] ≈ sin.(x[1, :])
    @test Θ[:, 2] ≈ sin.(x[2, :])
    @test Θ[:, 3] ≈ cos.(x[1, :])
    @test Θ[:, 4] ≈ cos.(x[2, :])
    @test Θ[:, 5] ≈ sin.(2 .* x[1, :])
    @test Θ[:, 7] ≈ cos.(2 .* x[1, :])
end

@testset "Compound basis" begin
    # A compound basis must be able to mix the two, which the orientation mismatch prevented.
    basis = CompoundBasis(polyorder = 2, trigonometric = 1)
    x = randn(2, 6)
    Θ = evaluate(x, basis)

    # 1 + 2 + 3 polynomial terms, then 1 wavenumber × {sin, cos} × 2 dof
    @test size(Θ) == (6, 10)
    @test all(isfinite, Θ)
end

@testset "calculate_nparams" begin
    # `d` is the number of degrees of freedom, so the phase space has 2d variables.
    @test calculate_nparams(1, 3, 0) == binomial(2 + 3, 3) - 1
    @test calculate_nparams(2, 3, 0) == binomial(4 + 3, 3) - 1

    # Each wavenumber contributes sin and cos on each of the 2d variables.
    @test calculate_nparams(1, 3, 2) == calculate_nparams(1, 3, 0) + 2 * 2 * 2
    @test calculate_nparams(2, 3, 2) == calculate_nparams(2, 3, 0) + 2 * 2 * 4
end
