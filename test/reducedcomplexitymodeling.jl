using LinearAlgebra
using ReducedComplexityModeling
using SparseIdentification
using Test

# Both packages are `using`ed, which is what makes the unqualified names below a check: with two
# exported `TrainingData`, every reference to it would be an `UndefVarError` instead. The testset
# "The name `TrainingData` is unambiguous" reads it as one.
using GeometricBase: CanonicalHamiltonianSystem, NoncanonicalHamiltonianSystem, StateData,
                     VectorFieldData

# A package names its own kind of canonicalising map by subtyping `IntrinsicSpace`, so the bridge
# has to accept a subtype and not the abstract type itself.
struct CanonicalCoordinates <: IntrinsicSpace end

# ── the model ────────────────────────────────────────────────────────────────────────────────
#
# H = (q² + p²)/2, so J∇H = (p, -q) and the Hamiltonian is quadratic: ½ on q² and ½ on p², zero on
# every other candidate term of the polynomial basis of order 2.
#
# The pairs are generated with the same implicit-midpoint step, taken by the same fixed number of
# Picard iterations, that the flow-map loss integrates with. They are then exactly the pairs the
# loss can reproduce, and the difference between the two fits is about the coordinates and not
# about the truncation of either.

const Δt = 0.01

function midpoint_step(z, iterations)
    z̃ = z .+ Δt .* [z[2], -z[1]]
    for _ in 1:iterations
        z̄ = (z .+ z̃) ./ 2
        z̃ = z .+ Δt .* [z̄[2], -z̄[1]]
    end
    z̃
end

function canonical_trajectory(z₀, n, iterations)
    zs = [z₀]
    for _ in 1:(n - 1)
        push!(zs, midpoint_step(zs[end], iterations))
    end
    zs
end

# The observation y = (q, p·e^q). Its Jacobian determinant is e^q, which is not constant, so the
# map is not a canonical transformation and no Hamiltonian in y reproduces the observed flow.
observe(z) = [z[1], z[2] * exp(z[1])]

# The loss the regression minimises, at the coefficients a fit returned:
# `Σⱼ ‖Φ_Δt(zⱼ; a) − zⱼ₊₁‖²` with `Φ_Δt` the implicit-midpoint step the loss uses. Read through the
# public vector field of the result, so the fit's own loss is what is compared.
function flow_map_loss(result, x, y, iterations)
    vectorfield = HamiltonianSINDyVectorField(result)
    f = zeros(eltype(first(x)), length(first(x)))
    total = 0.0
    for (x₀, x₁) in zip(x, y)
        vectorfield(f, x₀)
        x̃ = x₀ .+ Δt .* f
        for _ in 1:iterations
            vectorfield(f, (x₀ .+ x̃) ./ 2)
            x̃ = x₀ .+ Δt .* f
        end
        total += sum(abs2, x₁ .- x̃)
    end
    total
end

function canonical_data(states)
    TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
        (0:(length(states) - 1)) .* Δt, reduce(hcat, states))
end

# The `ArgumentError` a refused pair raises, or `nothing` where the pair is fitted instead.
function refusal(problem, method)
    try
        identify(problem, method)
        nothing
    catch err
        err
    end
end

message(err) = err isa ArgumentError ? sprint(showerror, err) : ""

# ── the checks ───────────────────────────────────────────────────────────────────────────────

@testset "The name `TrainingData` is unambiguous" begin
    # Each package owns one name: `ReducedComplexityModeling`'s `TrainingData` carries data on a
    # solution space, with the space it lives in as a type parameter, and `SparseIdentification`'s
    # `SINDyData` carries states with their derivatives, for the derivative-based fit. Two modules
    # exporting one name make every unqualified reference to it an `UndefVarError`, so the two
    # types are named apart and each is read here by its owner.
    @test TrainingData === ReducedComplexityModeling.TrainingData
    @test SINDyData === SparseIdentification.SINDyData
    @test parentmodule(TrainingData) === ReducedComplexityModeling
end

@testset "The observation is not a canonical coordinate pair" begin
    # The Jacobian of the observation, by central differences — not from its formula, so the test
    # can see a change in the map.
    function jacobian(z; h = 1e-6)
        d₁ = (observe(z .+ h .* [1.0, 0.0]) .- observe(z .- h .* [1.0, 0.0])) ./ (2h)
        d₂ = (observe(z .+ h .* [0.0, 1.0]) .- observe(z .- h .* [0.0, 1.0])) ./ (2h)
        hcat(d₁, d₂)
    end

    z₁, z₂ = [0.4, -0.7], [1.9, 0.3]
    det₁ = det(jacobian(z₁))
    det₂ = det(jacobian(z₂))

    # The determinant is `e^q`, and it is not the same number at the two states. A canonical
    # transformation has one determinant for the whole phase space, so no canonical `(q, p)`
    # reproduces this observation, however many states the fit is given.
    @test det₁ ≈ exp(z₁[1]) rtol = 1e-6
    @test det₂ ≈ exp(z₂[1]) rtol = 1e-6
    @test det₁ != det₂
end

@testset "Observable-space data is refused, and the refusal names it" begin
    zs = canonical_trajectory([0.5, -0.2], 8, 4)
    data = TrainingData{ObservableSpace, StateData, CanonicalHamiltonianSystem}(
        (0:(length(zs) - 1)) .* Δt, reduce(hcat, observe.(zs)))
    method = HamiltonianSINDy(λ = 0.05, integrator_timestep = Δt, polyorder = 2)

    # The observed states are tagged for what they are: the space, not the numbers, is what the
    # fit cannot check for itself.
    err = refusal(data, method)

    @test err isa ArgumentError
    msg = message(err)
    @test startswith(err.msg, "HamiltonianSINDy cannot be applied")   # the method
    @test occursin("ObservableSpace", msg)                      # the space
    @test occursin("StateData", msg)                            # the data type
    @test occursin("CanonicalHamiltonianSystem", msg)           # the system type
    @test occursin("map the data to canonical", msg)            # and what to do instead
    @test occursin("tag them `IntrinsicSpace`", msg)
end

@testset "The composed fit recovers H, the plain fit does not" begin
    method = HamiltonianSINDy(λ = 0.3, integrator_timestep = Δt, polyorder = 2)
    zs = canonical_trajectory([1.5, 0.2], 61, method.picard_iterations)
    x, y = zs[1:(end - 1)], zs[2:end]

    # The composed fit: the canonical trajectory, tagged as canonical, paired by the bridge.
    composed = identify(canonical_data(zs), method)
    a = parameters(composed)

    # The basis is q, p, q², qp, p² in that order — the constant is stripped — so H = (q² + p²)/2
    # is ½ on the two squares and zero on everything else. λ sits above the magnitude the first
    # pass gives the three spurious terms (~0.16) and below ½, which is what makes the support come
    # out sparse; the two squares land ~1e-13 from ½, five orders inside the 1e-6 asserted here.
    @test a[3] ≈ 0.5 atol = 1e-6
    @test a[5] ≈ 0.5 atol = 1e-6
    @test all(iszero, a[[1, 2, 4]])
    @test nterms(composed) == 2

    # The plain fit: the same trajectory, observed in the coordinates above, paired by hand
    # because no `TrajectoryData` is built from an observation that is not canonical.
    observed = observe.(zs)
    plain = identify(TrajectoryData(observed[1:(end - 1)], observed[2:end], Δt), method)

    loss_composed = flow_map_loss(composed, x, y, method.picard_iterations)
    loss_plain = flow_map_loss(plain, observed[1:(end - 1)], observed[2:end],
        method.picard_iterations)

    # Each loss is the quantity its own regression minimises, on the same trajectory: the composed
    # fit reaches machine zero and the plain fit cannot approach it, because the observed flow is
    # not the symplectic gradient of any scalar in those coordinates.
    @test loss_plain ≥ 1e3 * loss_composed
    @test loss_plain > 0
end

@testset "The time step of the data is the step the fit integrates at" begin
    method = HamiltonianSINDy(λ = 0.3, integrator_timestep = Δt, polyorder = 2)
    zs = canonical_trajectory([0.4, 0.9], 11, method.picard_iterations)
    time = (0:(length(zs) - 1)) .* Δt
    data = TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
        time, reduce(hcat, zs))

    # The data are one `Δt` apart and the method integrates at `Δt`, so the conversion reports
    # that step: `sparsify` reads `method.integrator_timestep` and never the data's, and a fit can
    # then fail because of the step rather than because of the coordinates.
    @test all(step -> isapprox(step, method.integrator_timestep; rtol = √eps(Float64)),
        diff(data.time))
    converted = SparseIdentification._trajectory_data(data, method)
    @test timestep(converted) == method.integrator_timestep

    # A time vector whose own step is the method's only to `√eps` — here it differs in the ninth
    # significant figure, well inside the tolerance, so the fit is accepted — still carries the
    # method's step into the conversion, and not the first step of the data.
    near = TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
        (1 + 1e-9) .* time, reduce(hcat, zs))
    @test timestep(SparseIdentification._trajectory_data(near, method)) ==
          method.integrator_timestep

    # Data at another step are refused rather than fitted for the wrong interval...
    other = HamiltonianSINDy(λ = 0.3, integrator_timestep = 2Δt, polyorder = 2)
    err = refusal(data, other)
    @test err isa ArgumentError

    # ... and so is one interior irregular interval, which a check that reads only the first step —
    # or only one step — accepts: every interval has to be at the step, because a pair one and a
    # half steps apart describes a flow map that no Hamiltonian reproduces.
    irregular = copy(collect(time))
    irregular[3] += Δt / 2
    @test first(diff(irregular)) == Δt    # the first interval is at the step, the third is not
    @test refusal(
        TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
            irregular, reduce(hcat, zs)),
        method) isa ArgumentError

    # The tolerance is `√eps` of the time values and not a loose bound: a step 1e-4 relative away
    # is refused, where a tolerance of a few percent would take it.
    close = HamiltonianSINDy(λ = 0.3, integrator_timestep = (1 + 1e-4) * Δt, polyorder = 2)
    @test refusal(data, close) isa ArgumentError

    # ... while times that accumulate round-off are at `Δt` to `√eps` and are accepted, which is
    # what the relative comparison is for.
    accumulated = TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
        cumsum(fill(Δt, length(zs))), reduce(hcat, zs))
    @test identify(accumulated, method) isa HamiltonianSINDyResult

    # A time vector stored in a container whose element type is abstract holds `Float64` values
    # like any other: the tolerance comes from the values, while `AbstractFloat` and `Any` are not
    # types with an `eps` of their own.
    for storage in (AbstractFloat, Any)
        stored = TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
            Vector{storage}(time), reduce(hcat, zs))
        @test identify(stored, method) isa HamiltonianSINDyResult
    end

    # The tolerance comes from the values' own floating type, not from `Float64`'s: each step of a
    # `Float32` time vector sits a relative ~2e-8 from `Δt`, which `√eps(Float32)` takes and
    # `√eps(Float64)` would refuse, so the fit is accepted here and only here.
    @test refusal(
        TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
            Float32.(time), reduce(hcat, zs)),
        method) === nothing
end

@testset "An ensemble pairs consecutive times within each member" begin
    method = HamiltonianSINDy(λ = 0.3, integrator_timestep = Δt, polyorder = 2)
    n = 41
    members = (canonical_trajectory([1.5, 0.2], n, method.picard_iterations),
        canonical_trajectory([-0.5, -1.0], n, method.picard_iterations))

    # RCM's ensemble layout: the middle axis is the member, the last one is time.
    data = TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
        (0:(n - 1)) .* Δt, stack((reduce(hcat, member) for member in members); dims = 2))

    @test size(data.data) == (2, 2, n)

    result = identify(data, method)
    a = parameters(result)

    # Only the 40 pairs of the two trajectories are fitted, and each of them is a flow-map pair.
    # Pairing across members — one member's last state with the next member's first — would fit
    # pairs no Hamiltonian reproduces, and neither square would come out at ½.
    @test a[3] ≈ 0.5 atol = 1e-6
    @test a[5] ≈ 0.5 atol = 1e-6
    @test all(iszero, a[[1, 2, 4]])
    @test nterms(result) == 2
end

@testset "The layout guards" begin
    method = HamiltonianSINDy(λ = 0.3, integrator_timestep = Δt, polyorder = 2)
    zs = canonical_trajectory([0.4, 0.9], 5, method.picard_iterations)
    time = (0:(length(zs) - 1)) .* Δt

    # Each layout guard is read here by its own phrase, so that dropping one and letting another
    # refusal answer in its place fails rather than passing as the same `ArgumentError`.

    # A vector of states is not one trajectory of a `TrainingData`: the states are along the last
    # dimension, and here that is the time axis itself, so one number per time is all there is.
    err = refusal(
        TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
            time, fill(1.0, length(time))),
        method)
    @test err isa ArgumentError
    @test occursin("is 2 or 3, got 1", message(err))

    # Four dimensions name no layout the bridge knows.
    err = refusal(
        TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
            time, fill(1.0, 2, 2, 2, length(time))),
        method)
    @test err isa ArgumentError
    @test occursin("is 2 or 3, got 4", message(err))

    # Fewer than two times cannot form a pair, in either layout.
    single = TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
        [0.0], reduce(hcat, [zs[1]]))
    single_ensemble = TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
        [0.0], reshape(repeat(zs[1], 2), 2, 2, 1))
    err = refusal(single, method)
    @test err isa ArgumentError
    @test occursin("at least two times", message(err))
    err = refusal(single_ensemble, method)
    @test err isa ArgumentError
    @test occursin("at least two times", message(err))

    # An ensemble with no member holds no state pair at all, so there is nothing to fit: that is
    # the same refusal and not a `BoundsError` from the regression.
    err = refusal(
        TrainingData{IntrinsicSpace, StateData, CanonicalHamiltonianSystem}(
            time, zeros(2, 0, length(time))),
        method)
    @test err isa ArgumentError
    @test occursin("at least one member", message(err))
end

@testset "Only canonical states on an `IntrinsicSpace` are accepted" begin
    method = HamiltonianSINDy(λ = 0.3, integrator_timestep = Δt, polyorder = 2)
    zs = canonical_trajectory([0.4, 0.9], 11, method.picard_iterations)
    states = reduce(hcat, zs)
    time = (0:(length(zs) - 1)) .* Δt

    # A subtype of `IntrinsicSpace`: a package names its own kind of map that way, so a fit through
    # one is accepted and the bound is not the abstract type itself.
    tagged = TrainingData{CanonicalCoordinates, StateData, CanonicalHamiltonianSystem}(
        time,
        states)
    @test identify(tagged, method) isa HamiltonianSINDyResult

    # The data type and the system type are part of what the fit needs: `StateData` on a canonical
    # Hamiltonian system is what says the columns are `q` then `p`. Anything else is refused, and
    # the refusal names the pair it was given.
    for (DT, SY) in ((StateData, NoncanonicalHamiltonianSystem),
        (VectorFieldData, CanonicalHamiltonianSystem))
        err = refusal(TrainingData{IntrinsicSpace, DT, SY}(time, states), method)
        @test err isa ArgumentError
        msg = message(err)
        @test occursin(string(DT), msg)
        @test occursin(string(SY), msg)
        @test occursin("map the data to canonical", msg)
        @test occursin("tag them `IntrinsicSpace`", msg)
    end

    # The refusal is of the *pair*, so the other method this package has is refused on the same
    # data by the same rule, and the refusal names that method.
    err = refusal(canonical_data(zs), SINDy(CompoundBasis(polyorder = 2); λ = 0.05))
    @test err isa ArgumentError
    @test startswith(err.msg, "SINDy cannot be applied")
end
