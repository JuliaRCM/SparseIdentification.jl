using SparseIdentification
using Test

@testset "Linear 2D oscillator" begin
    # ẋ = -0.1x + 2y,  ẏ = -2x - 0.1y — the illustrative example of Brunton, Proctor & Kutz
    # (PNAS 2016). On clean data the coefficients must come back exactly.
    A = [-0.1 2.0
         -2.0 -0.1]

    x = randn(2, 500)
    ẋ = A * x

    basis = CompoundBasis(polyorder = 3, trigonometric = 0)
    vectorfield = VectorField(SINDy(λ = 0.05), basis, TrainingData(x, ẋ))
    Ξ = vectorfield.coefficients

    # Rows 2 and 3 are the linear terms; `Ξ` maps library terms to state components, so the
    # linear block is the transpose of `A`.
    @test Ξ[2:3, :]≈A' atol=1e-12

    # Everything else is thresholded to exactly zero, not merely small.
    @test all(iszero, Ξ[1, :])
    @test all(iszero, Ξ[4:end, :])
end

@testset "Lorenz 63" begin
    # σ = 10, ρ = 28, β = 8/3 — the headline example of the SINDy paper.
    σ, ρ, β = 10.0, 28.0, 8 / 3

    x = 10 .* randn(3, 2000)
    ẋ = similar(x)
    for j in axes(x, 2)
        ẋ[:, j] .= lorenz(x[:, j], (σ, β, ρ), 0.0)
    end

    basis = CompoundBasis(polyorder = 2, trigonometric = 0)
    vectorfield = VectorField(SINDy(λ = 0.1), basis, TrainingData(x, ẋ))
    Ξ = vectorfield.coefficients

    # Library order for 3 dof: [1, x, y, z, x², xy, xz, y², yz, z²]
    truth = zeros(10, 3)
    truth[2, 1] = -σ                     # ẋ = σ(y - x)
    truth[3, 1] = σ
    truth[2, 2] = ρ                      # ẏ = x(ρ - z) - y
    truth[3, 2] = -1.0
    truth[7, 2] = -1.0                   # -xz
    truth[4, 3] = -β                     # ż = xy - βz
    truth[6, 3] = 1.0                    # xy

    @test Ξ≈truth atol=1e-10
end

@testset "Thresholding is exact and terminating" begin
    # Zhang & Schaeffer (2019): STLSQ terminates in at most `n` iterations, where `n` is the
    # number of library terms. Running with far more loops than that must give the same answer
    # as running with `n`.
    A = [-0.1 2.0
         -2.0 -0.1]
    x = randn(2, 300)
    ẋ = A * x
    basis = CompoundBasis(polyorder = 3, trigonometric = 0)
    data = TrainingData(x, ẋ)

    few = VectorField(SINDy(λ = 0.05, nloops = 10), basis, data).coefficients
    many = VectorField(SINDy(λ = 0.05, nloops = 200), basis, data).coefficients

    @test few == many
end

@testset "Identification is deterministic" begin
    # The estimator used to add fresh `randn` noise to its own targets on every call, so two
    # identical calls disagreed. Noise belongs to the data, not to the algorithm.
    A = [-0.1 2.0
         -2.0 -0.1]
    x = randn(2, 200)
    data = TrainingData(x, A * x)
    basis = CompoundBasis(polyorder = 3, trigonometric = 0)

    a = VectorField(SINDy(λ = 0.05), basis, data).coefficients
    b = VectorField(SINDy(λ = 0.05), basis, data).coefficients

    @test a == b
end
