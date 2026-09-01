using SparseIdentification
using Test

using SparseIdentification: hamilGrad_func_builder, sparsify

@testset "Hamiltonian gradient builder" begin
    # The builder generates J∇H symbolically and compiles it. Under Symbolics 7 this needs no
    # `inject_registered_module_functions` wrapper — that name exists in no released Symbolics,
    # so the call used to fail as soon as it was reached.
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
    # Regression test for the parameter-count mismatch. `calculate_nparams` takes the number of
    # degrees of freedom `d`, and the phase space has `2d` variables. Passing the full state
    # dimension where `d` was expected made the optimiser search 212 coefficients for a basis
    # that reads 58 of them.
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
end

@testset "An odd state dimension is rejected" begin
    x = [randn(3) for _ in 1:5]
    y = [randn(3) for _ in 1:5]
    method = HamiltonianSINDy(polyorder = 2)
    @test_throws ArgumentError identify(TrajectoryData(x, y, 0.01), method)
end
