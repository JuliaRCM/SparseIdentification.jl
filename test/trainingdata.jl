using Random
using SparseIdentification
using Test

Random.seed!(1234)

@testset "SINDyData" begin
    x = randn(4, 20)
    ẋ = randn(4, 20)
    data = SINDyData(x, ẋ)

    @test nsamples(data) == 20
    @test statedimension(data) == 4

    # Shapes are validated at construction, so a mismatch surfaces here rather than as a wrong
    # answer several layers down.
    @test_throws DimensionMismatch SINDyData(randn(4, 20), randn(4, 19))
    @test_throws DimensionMismatch SINDyData(randn(4, 20), randn(3, 20))
    @test_throws ArgumentError SINDyData(randn(4, 20), [randn(4) for _ in 1:20])
end

@testset "SINDyData from vectors of states" begin
    x = [randn(4) for _ in 1:12]
    ẋ = [randn(4) for _ in 1:12]
    data = SINDyData(x, ẋ)

    @test nsamples(data) == 12
    @test statedimension(data) == 4

    @test_throws DimensionMismatch SINDyData(x, [randn(4) for _ in 1:11])
    @test_throws DimensionMismatch SINDyData(x, [randn(3) for _ in 1:12])
end

@testset "TrajectoryData" begin
    x = [randn(2) for _ in 1:8]
    y = [randn(2) for _ in 1:8]
    data = TrajectoryData(x, y, 0.01)

    @test nsamples(data) == 8
    @test statedimension(data) == 2
    @test timestep(data) == 0.01

    @test_throws ArgumentError TrajectoryData(x, y, 0.0)
    @test_throws ArgumentError TrajectoryData(x, y, -0.01)
    @test_throws DimensionMismatch TrajectoryData(x, [randn(2) for _ in 1:7], 0.01)
end
