using Random
using SparseIdentification
using Symbolics
using Test

using SparseIdentification: strip_constants

Random.seed!(1234)

@testset "Arguments" begin
    z = Symbolics.variables(:z, 1:4)

    @test length(SparseIdentification.basis_arguments(StateComponents(), z)) == 4

    # All pairs i > j over the named components.
    all_pairs = SparseIdentification.basis_arguments(Differences(1:3), z)
    @test length(all_pairs) == 3                       # (2,1), (3,1), (3,2)

    # Consecutive only: a nearest-neighbour lattice.
    consec = SparseIdentification.basis_arguments(Differences(1:4; consecutive = true), z)
    @test length(consec) == 3

    @test_throws ArgumentError Differences([1])        # need two to make a difference
    @test_throws ArgumentError Differences([1, 1, 2])  # duplicated index

    # An index below 1 names no state component. Rejecting it at construction gives an error that
    # says so, where deferring to the indexing gives a `BoundsError` from two layers down.
    @test_throws ArgumentError Differences([0, 1])
    @test_throws ArgumentError Differences([-1, 2])

    # Referencing a component the state does not have is caught, not silently truncated.
    @test_throws ArgumentError SparseIdentification.basis_arguments(Differences(1:6), z)
end

@testset "Bases built from the same parameters are equal and hash alike" begin
    # This is what makes a basis usable as the evaluator cache's key. `Differences` holds a
    # `Vector{Int}`, so the default identity comparison makes two structurally identical bases
    # distinct and every `evaluate` recompiles.
    mk() = ExponentialBasis(Differences(1:3; consecutive = true); rates = (-1.0,))

    @test mk() == mk()
    @test hash(mk()) == hash(mk())
    @test isequal(mk(), mk())

    # Different parameters must still compare unequal.
    @test mk() != ExponentialBasis(Differences(1:3); rates = (-1.0,))
    @test mk() != ExponentialBasis(Differences(1:3; consecutive = true); rates = (1.0,))
    @test PolynomialBasis(2) != PolynomialBasis(3)
    @test CompoundBasis(polyorder = 2) == CompoundBasis(polyorder = 2)

    # And the cache must therefore hit: repeated calls with a freshly built basis add no entries.
    data = randn(6, 20)
    evaluate(data, mk())
    n = length(SparseIdentification.EVALUATOR_CACHE)
    for _ in 1:5
        evaluate(data, mk())
    end
    @test length(SparseIdentification.EVALUATOR_CACHE) == n
end

@testset "Exponential basis" begin
    # The Toda lattice interacts through exp(-(qₙ₊₁ - qₙ)).
    basis = ExponentialBasis(Differences(1:3; consecutive = true); rates = (-1.0,))
    @test nterms(basis, 6) == 2

    # Three components, two snapshots.
    data = [0.0 1.0
            1.0 3.0
            3.0 6.0]
    Θ = evaluate(data, basis)

    @test size(Θ) == (2, 2)
    @test Θ[1, 1] ≈ exp(-(1.0 - 0.0))
    @test Θ[1, 2] ≈ exp(-(3.0 - 1.0))
    @test Θ[2, 1] ≈ exp(-(3.0 - 1.0))
    @test Θ[2, 2] ≈ exp(-(6.0 - 3.0))

    # Several rates multiply the term count.
    @test nterms(ExponentialBasis(; rates = (-1.0, 1.0)), 3) == 6
end

@testset "Logarithmic basis" begin
    # A point-vortex Hamiltonian is built from log|qᵢ - qⱼ|.
    basis = LogarithmicBasis(Differences(1:3))
    @test nterms(basis, 6) == 3

    data = reshape([1.0, 3.0, 7.0], 3, 1)
    Θ = evaluate(data, basis)

    @test size(Θ) == (1, 3)
    @test sort(vec(Θ)) ≈ sort([log(2.0), log(6.0), log(4.0)])

    # `abs` inside the logarithm means the sign of the difference does not matter.
    flipped = reshape([7.0, 3.0, 1.0], 3, 1)
    @test sort(vec(evaluate(flipped, basis))) ≈ sort(vec(Θ))
end

@testset "Rational basis" begin
    # An N-body gravitational Hamiltonian is built from 1/|qᵢ - qⱼ|.
    basis = RationalBasis(Differences(1:3))
    @test nterms(basis, 6) == 3

    data = reshape([1.0, 3.0, 7.0], 3, 1)
    Θ = evaluate(data, basis)

    @test size(Θ) == (1, 3)
    @test sort(vec(Θ)) ≈ sort([1 / 2, 1 / 6, 1 / 4])

    # `abs` inside the power means the terms are positive and the sign of the difference does not
    # matter — a reciprocal distance, not a reciprocal signed difference.
    @test all(>(0), Θ)
    flipped = reshape([7.0, 3.0, 1.0], 3, 1)
    @test sort(vec(evaluate(flipped, basis))) ≈ sort(vec(Θ))

    @test nterms(RationalBasis(; powers = (1, 2)), 3) == 6
end

@testset "Composition" begin
    poly = CompoundBasis(polyorder = 2)                   # 1 + 2 + 3 = 6 terms in 2 dof
    expo = ExponentialBasis(; rates = (-1.0,))            # 2 terms in 2 dof
    combined = poly ⊕ expo

    @test nterms(combined, 2) == nterms(poly, 2) + nterms(expo, 2)

    x = randn(2, 5)
    @test evaluate(x, combined) ≈ hcat(evaluate(x, poly), evaluate(x, expo))
end

@testset "Constants are stripped from a Hamiltonian basis" begin
    z = Symbolics.variables(:z, 1:2)

    # `CompoundBasis` includes the constant; a Hamiltonian ansatz cannot identify it, because it
    # contributes an identically-zero column to J∇H.
    withconst = CompoundBasis(polyorder = 2)
    @test length(basis_functions(withconst, z)) == length(strip_constants(withconst, z)) + 1

    # `hamiltonian_basis` omits it in the first place.
    @test length(strip_constants(hamiltonian_basis(polyorder = 2), z)) ==
          length(basis_functions(hamiltonian_basis(polyorder = 2), z))

    # A basis with nothing identifiable in it is rejected rather than producing a singular fit.
    @test_throws ArgumentError SparseIdentification.hamiltonian_functions(PolynomialBasis(0), 1)
end

@testset "Hamiltonian identification with a custom basis" begin
    # H = ½p² + ½q², built from an explicitly supplied basis rather than a polyorder keyword.
    Δt = 0.01
    R = [cos(Δt) sin(Δt); -sin(Δt) cos(Δt)]
    x = [randn(2) for _ in 1:60]
    y = [R * xⱼ for xⱼ in x]

    basis = hamiltonian_basis(polyorder = 2)
    method = HamiltonianSINDy(basis; λ = 0.05, integrator_timestep = Δt)
    result = identify(TrajectoryData(x, y, Δt), method)

    vf = HamiltonianSINDyVectorField(result)
    dz = zeros(2)
    for _ in 1:5
        z = randn(2)
        vf(dz, z)
        @test dz≈[z[2], -z[1]] atol=1e-3
    end
end

@testset "The Hamiltonian path accepts the logarithmic and rational bases" begin
    # These two are the bases whose stated purpose is the Hamiltonian path — the point vortex and
    # the N-body problem — and both put `abs` inside, which `hamiltonian_functions` has to
    # differentiate symbolically to build `ż = J∇H`. Compiling and evaluating them is what pins
    # that, independently of whether a fit recovers anything.
    for args in (Differences(1:2), StateComponents())
        for b in (LogarithmicBasis(args), RationalBasis(args))
            basis = hamiltonian_basis(polyorder = 2) ⊕ b
            H = hamiltonian_functions(basis, 1)

            @test H.nparam == length(basis_functions(basis, Symbolics.variables(:z, 1:2)))

            # Away from the singularity at a vanishing argument, the field is finite.
            ż = zeros(2)
            H.ż(ż, [0.7, 1.3], 0.1 .* collect(1:(H.nparam)))
            @test all(isfinite, ż)
        end
    end
end

@testset "A Toda-type basis is expressible and identifiable" begin
    # Two particles with a Toda interaction: H = ½(p₁² + p₂²) + exp(-(q₂ - q₁)). Expressing it
    # needs an exponential of a *difference* of positions, not of a single state component.
    exact_H(z) = (z[3]^2 + z[4]^2) / 2 + exp(-(z[2] - z[1]))

    # ż = J∇H
    function grad_H(z)
        e = exp(-(z[2] - z[1]))
        [z[3], z[4], -e, e]           # (q̇₁, q̇₂, ṗ₁, ṗ₂)
    end

    basis = hamiltonian_basis(polyorder = 2) ⊕
            ExponentialBasis(Differences(1:2; consecutive = true); rates = (-1.0,))

    z = Symbolics.variables(:z, 1:4)
    φ = basis_functions(basis, z)

    # The interaction term is present in the library, which is the point.
    @test any(isequal(exp(z[1] - z[2])), φ)

    # And the compiled Hamiltonian machinery accepts it.
    hfuns = SparseIdentification.hamiltonian_functions(basis, 2)
    @test hfuns.nparam == length(φ)
    @test hfuns.d == 2

    # Evaluating J∇H with the coefficients of the exact Hamiltonian reproduces the true field.
    # Coefficient order follows the basis: [q₁,q₂,p₁,p₂, q₁²,q₁q₂,q₁p₁,q₁p₂,q₂²,…, exp(q₁-q₂)]
    a = zeros(hfuns.nparam)
    idx_p1sq = findfirst(isequal(z[3]^2), φ)
    idx_p2sq = findfirst(isequal(z[4]^2), φ)
    idx_exp = findfirst(isequal(exp(z[1] - z[2])), φ)
    @test all(!isnothing, (idx_p1sq, idx_p2sq, idx_exp))

    a[idx_p1sq] = 0.5
    a[idx_p2sq] = 0.5
    a[idx_exp] = 1.0

    out = zeros(4)
    for _ in 1:5
        zv = randn(4)
        hfuns.ż(out, zv, a)
        @test out≈grad_H(zv) atol=1e-10
    end
end
