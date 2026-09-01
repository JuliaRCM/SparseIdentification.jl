using SparseIdentification
using Test

@testset "TrainingData" begin
    x = randn(4, 20)
    ẋ = randn(4, 20)
    data = TrainingData(x, ẋ)

    @test nsnapshots(data) == 20
    @test statedimension(data) == 4

    # The struct used to have three fields and no constructor at all, so every caller in
    # `scripts/` was passing two arguments to something that needed three.
    @test_throws DimensionMismatch TrainingData(randn(4, 20), randn(4, 19))
    @test_throws DimensionMismatch TrainingData(randn(4, 20), randn(3, 20))
    @test_throws ArgumentError TrainingData(randn(4, 20), [randn(4) for _ in 1:20])
end

@testset "TrainingData from vectors of states" begin
    x = [randn(4) for _ in 1:12]
    ẋ = [randn(4) for _ in 1:12]
    data = TrainingData(x, ẋ)

    @test nsnapshots(data) == 12
    @test statedimension(data) == 4

    @test_throws DimensionMismatch TrainingData(x, [randn(4) for _ in 1:11])
    @test_throws DimensionMismatch TrainingData(x, [randn(3) for _ in 1:12])
end

@testset "TrajectoryData" begin
    x = [randn(2) for _ in 1:8]
    y = [randn(2) for _ in 1:8]
    data = TrajectoryData(x, y, 0.01)

    @test nsnapshots(data) == 8
    @test statedimension(data) == 2
    @test data.Δt == 0.01

    @test_throws ArgumentError TrajectoryData(x, y, 0.0)
    @test_throws ArgumentError TrajectoryData(x, y, -0.01)
    @test_throws DimensionMismatch TrajectoryData(x, [randn(2) for _ in 1:7], 0.01)
end
