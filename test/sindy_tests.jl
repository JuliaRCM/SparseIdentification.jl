using Random
using SparseIdentification
using Test

# The assertions below compare recovered coefficients to the truth at 1e-12 and 1e-10, which is
# what clean data earns — but only for a well-conditioned draw. Seeding fixes the draw for a given
# Julia version, so repeated runs agree and a failure means the estimator changed rather than that
# the sample was unlucky. The stream is not guaranteed across Julia versions, so this pins the
# tolerances rather than the exact matrix.
Random.seed!(1234)

"The Lorenz system, written out so the expected coefficients below are explicit."
function lorenz_rhs(y, σ, β, ρ)
    [σ * (y[2] - y[1]),
        y[1] * (ρ - y[3]) - y[2],
        y[1] * y[2] - β * y[3]]
end

@testset "Linear 2D oscillator" begin
    # ẋ = -0.1x + 2y,  ẏ = -2x - 0.1y — the illustrative example of Brunton, Proctor & Kutz
    # (PNAS 2016). On clean data the coefficients must come back exactly.
    A = [-0.1 2.0
         -2.0 -0.1]

    x = randn(2, 500)
    ẋ = A * x

    basis = CompoundBasis(polyorder = 3, trigonometric = 0)
    result = identify(TrainingData(x, ẋ), SINDy(basis; λ = 0.05))
    Ξ = parameters(result)

    # Rows 2 and 3 are the linear terms; `Ξ` maps library terms to state components, so the
    # linear block is the transpose of `A`.
    @test Ξ[2:3, :]≈A' atol=1e-12

    # Everything else is thresholded to exactly zero, not merely small.
    @test all(iszero, Ξ[1, :])
    @test all(iszero, Ξ[4:end, :])

    @test nterms(result) == 4
end

@testset "Lorenz 63" begin
    # σ = 10, ρ = 28, β = 8/3 — the headline example of the SINDy paper.
    σ, ρ, β = 10.0, 28.0, 8 / 3

    x = 10 .* randn(3, 2000)
    ẋ = reduce(hcat, [lorenz_rhs(x[:, j], σ, β, ρ) for j in axes(x, 2)])

    basis = CompoundBasis(polyorder = 2, trigonometric = 0)
    Ξ = parameters(identify(TrainingData(x, ẋ), SINDy(basis; λ = 0.1)))

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
    data = TrainingData(x, A * x)
    basis = CompoundBasis(polyorder = 3, trigonometric = 0)

    few = parameters(identify(data, SINDy(basis; λ = 0.05, nloops = 10)))
    many = parameters(identify(data, SINDy(basis; λ = 0.05, nloops = 200)))

    @test few == many
end

@testset "Identification is deterministic" begin
    # The estimator draws no randomness of its own, so two identical calls agree exactly. Noise
    # belongs to the data, not to the algorithm.
    A = [-0.1 2.0
         -2.0 -0.1]
    x = randn(2, 200)
    data = TrainingData(x, A * x)
    method = SINDy(CompoundBasis(polyorder = 3, trigonometric = 0); λ = 0.05)

    @test parameters(identify(data, method)) == parameters(identify(data, method))
end

@testset "Training data given as vectors of states" begin
    # `TrainingData` accepts both shapes, and `TrainingData(solution)` produces the vector one, so
    # `identify` must give the same answer for both.
    A = [-0.1 2.0
         -2.0 -0.1]
    x = randn(2, 200)
    basis = CompoundBasis(polyorder = 3, trigonometric = 0)
    method = SINDy(basis; λ = 0.05)

    from_matrix = parameters(identify(TrainingData(x, A * x), method))
    from_vectors = parameters(identify(
        TrainingData([x[:, j] for j in axes(x, 2)], [(A * x)[:, j] for j in axes(x, 2)]), method))

    @test from_vectors == from_matrix
end

@testset "Identified vector field evaluates" begin
    A = [-0.1 2.0
         -2.0 -0.1]
    x = randn(2, 300)
    result = identify(TrainingData(x, A * x),
        SINDy(CompoundBasis(polyorder = 3, trigonometric = 0); λ = 0.05))

    vf = SINDyVectorField(result)
    dz = zeros(2)
    z = [0.7, -0.3]

    # GeometricEquations calls a vector field as v(v, t, q, params).
    vf(dz, 0.0, z, nothing)
    @test dz≈A * z atol=1e-10
end
