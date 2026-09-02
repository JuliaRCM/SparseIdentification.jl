using GeometricBase
using GeometricEquations
using GeometricIntegrators
using GeometricProblems.HarmonicOscillator
using Random
using SparseIdentification
using Test

Random.seed!(1234)

# The six `is*` traits are extended but not exported, because GeometricIntegratorsBase defines
# its own generics of the same names. Reach them qualified.
using SparseIdentification: isexplicit, isimplicit, issymplectic, isenergypreserving

@testset "Methods are GeometricBase methods" begin
    sindy = SINDy(CompoundBasis(polyorder = 2); λ = 0.05)
    ham = HamiltonianSINDy(λ = 0.05, polyorder = 2)

    @test sindy isa GeometricBase.AbstractMethod
    @test ham isa GeometricBase.AbstractMethod

    for m in (sindy, ham)
        @test name(m) isa String
        @test description(m) isa String
        @test reference(m) isa String
        @test isexplicit(m) isa Bool
        @test isimplicit(m) isa Bool
        @test issymplectic(m) isa Bool
        @test isenergypreserving(m) isa Bool
    end

    # The substantive trait: only the Hamiltonian method preserves structure by construction.
    @test issymplectic(ham)
    @test isenergypreserving(ham)
    @test !issymplectic(sindy)
    @test !isenergypreserving(sindy)

    # The regression is a direct solve for SINDy and goes through an integrator for the
    # Hamiltonian flow-map formulation.
    @test isexplicit(sindy) && !isimplicit(sindy)
    @test isimplicit(ham) && !isexplicit(ham)
end

@testset "Problems are GeometricBase problems" begin
    data = TrainingData(randn(2, 20), randn(2, 20))
    traj = TrajectoryData([randn(2) for _ in 1:8], [randn(2) for _ in 1:8], 0.01)

    @test data isa GeometricBase.AbstractProblem
    @test traj isa GeometricBase.AbstractProblem
    @test data isa IdentificationProblem

    @test nsamples(data) == 20
    @test nsamples(traj) == 8
    @test statedimension(data) == 2
    @test timestep(traj) == 0.01
    @test datatype(data) == Float64
    @test arrtype(data) <: AbstractMatrix

    # An unimplemented (problem, method) pair names what is missing rather than raising a
    # MethodError from several layers down.
    @test_throws ArgumentError identify(data, HamiltonianSINDy(λ = 0.05, polyorder = 2))
    @test_throws ArgumentError identify(traj, SINDy(CompoundBasis(polyorder = 2); λ = 0.05))
end

@testset "Round trip: problem → integrate → identify → problem → integrate" begin
    # The loop that makes the package compose with the ecosystem. Integrate a known harmonic
    # oscillator, identify it from the solution, turn the result back into a problem, and
    # integrate that.
    ref = hodeproblem()
    sol = integrate(ref, ImplicitMidpoint())

    @test sol isa GeometricSolutions.GeometricSolution

    # `TrajectoryData` from a solution pairs each stored state with its successor.
    traj = TrajectoryData(sol)
    @test nsamples(traj) == length(eachindex(sol.t)) - 1
    @test statedimension(traj) == 2   # z = (q, p)

    # The harmonic oscillator needs only quadratic terms.
    method = HamiltonianSINDy(λ = 0.01, integrator_timestep = timestep(traj), polyorder = 2)
    result = identify(traj, method)

    @test parameters(result) isa AbstractVector
    @test degreesoffreedom(result) == 1

    # The round trip has to *identify*, not merely run: the pairing in `TrajectoryData(solution)`
    # is what decides the recovered field, and asserting only that the output is finite would
    # leave an off-by-one in that pairing — which its own comment warns about — passing.
    # H = p²/2m + k q²/2 with m = 1, so ż = J∇H = (p, -k q).
    k = parameters(ref).k
    vf = HamiltonianSINDyVectorField(result)
    dz = zeros(2)
    for _ in 1:5
        z = randn(2)
        vf(dz, z)
        @test dz≈[z[2], -k * z[1]] atol=1e-3
    end

    # Back to a GeometricEquations problem, and integrable.
    prob = HODEProblem(result, (0.0, 1.0), 0.01, [0.5], [0.0])
    @test prob isa GeometricEquations.HODEProblem

    idsol = integrate(prob, ImplicitMidpoint())
    @test idsol isa GeometricSolutions.GeometricSolution
    @test all(n -> all(isfinite, idsol.q[n]) && all(isfinite, idsol.p[n]),
        eachindex(idsol.q))
end

@testset "TrainingData from a GeometricSolution" begin
    ref = odeproblem()
    sol = integrate(ref, ExplicitEuler())

    data = TrainingData(sol)
    @test data isa TrainingData
    @test nsamples(data) == length(eachindex(sol.t))
    @test statedimension(data) == 2
end
