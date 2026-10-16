using Random
using SparseIdentification
using Test

using SparseIdentification: hamilGrad_func_builder

Random.seed!(1234)

@testset "Hamiltonian gradient builder" begin
    # The builder generates J∇H symbolically and compiles it, and the compiled function is
    # actually called here — a builder that only ever gets constructed proves nothing.
    for (d, polyorder, trig) in ((1, 3, 0), (1, 2, 1), (2, 3, 0), (2, 2, 1))
        fθ = hamilGrad_func_builder(d, polyorder, trig)
        nparam = calculate_nparams(d, polyorder, trig)

        out = zeros(2d)
        z = randn(2d)
        fθ(out, z, randn(nparam))

        @test length(out) == 2d
        @test all(isfinite, out)
    end
end

@testset "Basis size matches what the vector field consumes" begin
    # `calculate_nparams` takes the number of degrees of freedom `d`, and the phase space has
    # `2d` variables. Passing the full state dimension where `d` is expected inflates the count,
    # and the optimiser then searches coefficients the vector field never reads.
    for d in 1:3, polyorder in 2:3, trig in 0:1
        fθ = hamilGrad_func_builder(d, polyorder, trig)
        nparam = calculate_nparams(d, polyorder, trig)

        # A coefficient vector of exactly `nparam` entries must drive the field, and every entry
        # must matter — if the basis read fewer, the last coefficient would be inert.
        out₀ = zeros(2d)
        out₁ = zeros(2d)
        z = ones(2d) .+ 0.3
        a = ones(nparam)

        fθ(out₀, z, a)
        a[end] += 1.0
        fθ(out₁, z, a)

        @test out₀ != out₁
    end
end

@testset "Harmonic oscillator" begin
    # H = (q² + p²)/2, so ż = J∇H = (p, -q).
    H(z) = (z[1]^2 + z[2]^2) / 2
    grad_H(z) = [z[2], -z[1]]

    Δt = 0.01
    method = HamiltonianSINDy(λ = 0.05, integrator_timestep = Δt, polyorder = 2)

    # Consecutive states along the exact flow: a rotation by Δt.
    R = [cos(Δt) sin(Δt); -sin(Δt) cos(Δt)]
    x = [randn(2) for _ in 1:60]
    y = [R * xⱼ for xⱼ in x]

    result = identify(TrajectoryData(x, y, Δt), method)
    vectorfield = HamiltonianSINDyVectorField(result)

    # The identified field must reproduce ż = (p, -q) at points it never saw.
    dz = zeros(2)
    for _ in 1:10
        z = randn(2)
        vectorfield(dz, z)
        @test dz≈grad_H(z) atol=1e-3
    end

    # The identified Hamiltonian itself: the basis is q, p, q², qp, p², and H = (q² + p²)/2 is ½ on
    # the two squares and zero elsewhere. A flipped sign of `f` or swapped `v` and `f` fit the same
    # field with `(p² - q²)/2` or `-qp`, which only these assertions see. The midpoint map of H
    # rotates by `2atan(Δt/2)`, not `Δt`, so the squares come out ½(1 + Δt²/12) from exact pairs.
    a = parameters(result)
    @test a[3]≈0.5 atol=1e-4
    @test a[5]≈0.5 atol=1e-4
    @test all(iszero, a[[1, 2, 4]])
    @test nterms(result) == 2
end

@testset "Data at another step than the method's are refused" begin
    # `sparsify` integrates at `integrator_timestep` and never reads the data's step, so a fit of
    # data at another step would describe the wrong interval.
    x = [randn(2) for _ in 1:5]
    y = [randn(2) for _ in 1:5]
    method = HamiltonianSINDy(integrator_timestep = 0.01, polyorder = 2)
    @test_throws r"^ArgumentError: the step of the data is 0.02" identify(
        TrajectoryData(x, y, 0.02), method)
    @test_throws r"^ArgumentError: the step of the data is" identify(
        TrajectoryData(x, y, (1 + 1e-4) * 0.01), method)
end

@testset "An odd state dimension is rejected" begin
    x = [randn(3) for _ in 1:5]
    y = [randn(3) for _ in 1:5]
    method = HamiltonianSINDy(polyorder = 2)
    @test_throws ArgumentError identify(TrajectoryData(x, y, 0.01), method)
end
