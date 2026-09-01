
"""
    HamiltonianSINDy(; λ, polyorder, trigonometric, integrator_timestep, nloops, picard_iterations)

Sparse identification of a *Hamiltonian* system (Khan 2023).

A scalar Hamiltonian is parametrised over a library of candidate functions,
`H(z; a) = Σₖ aₖ φₖ(z)`, and the vector field is obtained as `ż = J ∇H(z)`. The identified dynamics
is therefore Hamiltonian **by construction** rather than by penalty: every candidate the fit
considers is a symplectic gradient field, whatever the coefficients happen to be.

The coefficients are fitted by matching the flow map — minimising `Σⱼ ‖Φ_Δt(zⱼ; a) − zⱼ₊₁‖²` where
`Φ_Δt` is an implicit-midpoint step. This is nonlinear in `a` and needs an optimiser.

Apply it to a [`TrajectoryData`](@ref) problem with [`identify`](@ref).

!!! note "Matching the vector field is cheaper when `ż` is available"
    `J∇H` is *linear* in `a`, so fitting against measured derivatives is an ordinary linear sparse
    regression with no optimiser at all. That formulation is not implemented yet.
"""
struct HamiltonianSINDy{T} <: SparsificationMethod
    λ::T
    integrator_timestep::T

    nloops::Int
    picard_iterations::Int

    polyorder::Int
    trigonometric::Int

    function HamiltonianSINDy(;
            λ::T = DEFAULT_LAMBDA,
            integrator_timestep::T = DEFAULT_INTEGRATOR_TIMESTEP,
            nloops::Int = DEFAULT_NLOOPS,
            picard_iterations::Int = DEFAULT_PICARD_ITERATIONS,
            polyorder::Int = 3,
            trigonometric::Int = 0) where {T}
        new{T}(λ, integrator_timestep, nloops, picard_iterations, polyorder, trigonometric)
    end
end

GeometricBase.name(::HamiltonianSINDy) = "HamiltonianSINDy"
function GeometricBase.description(::HamiltonianSINDy)
    "Sparse identification of a Hamiltonian, preserving the symplectic structure by construction"
end
function GeometricBase.reference(::HamiltonianSINDy)
    "N. B. Khan, Sparse Identification of Symplectic Hamiltonian Dynamics for Predictive " *
    "Modeling and Analysis, MSc thesis, TU München, 2023. mediaTUM 1747893"
end

# The identified model is J∇H for a scalar H, so it is symplectic and conserves H exactly,
# whatever the fitted coefficients turn out to be. This is the whole point of the method.
GeometricBase.issymplectic(::HamiltonianSINDy) = true
GeometricBase.isenergypreserving(::HamiltonianSINDy) = true
# The regression goes through an implicit-midpoint step, so it is not a direct solve.
GeometricBase.isexplicit(::HamiltonianSINDy) = false
GeometricBase.isimplicit(::HamiltonianSINDy) = true

"""
    HamiltonianSINDyResult

The outcome of [`identify`](@ref) with [`HamiltonianSINDy`](@ref).

Reach the coefficients with `parameters` and the compiled Hamiltonian and its symplectic
gradient with `functions`. Convert to a `GeometricEquations.HODEProblem` to integrate the
identified system with a symplectic integrator.
"""
struct HamiltonianSINDyResult{DT, CT <: AbstractVector{DT}, FT, MT <: HamiltonianSINDy}
    method::MT
    coefficients::CT
    hamiltonian::FT

    function HamiltonianSINDyResult(method::MT, coefficients::CT,
            hamiltonian::FT) where {
            DT, MT <: HamiltonianSINDy, CT <: AbstractVector{DT}, FT}
        new{DT, CT, FT, MT}(method, coefficients, hamiltonian)
    end
end

GeometricBase.parameters(result::HamiltonianSINDyResult) = result.coefficients
GeometricBase.functions(result::HamiltonianSINDyResult) = result.hamiltonian
GeometricBase.datatype(::HamiltonianSINDyResult{DT}) where {DT} = DT
method(result::HamiltonianSINDyResult) = result.method

"""
    degreesoffreedom(result)

The number of degrees of freedom `d`; the phase space has `2d` dimensions.
"""
degreesoffreedom(result::HamiltonianSINDyResult) = result.hamiltonian.d

nterms(result::HamiltonianSINDyResult) = count(!iszero, result.coefficients)

function Base.show(io::IO, result::HamiltonianSINDyResult)
    print(io, "Hamiltonian SINDy result: ", nterms(result), " of ",
        length(result.coefficients), " coefficients retained, ",
        degreesoffreedom(result), " degrees of freedom")
end

"""
    sparsify(method::HamiltonianSINDy, fθ, problem::TrajectoryData, solver; verbose = false)

Sequentially thresholded regression of the Hamiltonian coefficients against the flow map.
"""
function sparsify(method::HamiltonianSINDy, fθ, problem::TrajectoryData, solver;
        verbose = false)
    # `fθ` is built for `d` degrees of freedom, i.e. `2d` phase-space variables, and
    # `calculate_nparams` takes that same `d`. Passing the full state dimension here instead is
    # what made the optimiser search 212 coefficients for a basis that reads 58 of them.
    d = statedimension(problem) ÷ 2
    nparam = calculate_nparams(d, method.polyorder, method.trigonometric)

    coeffs = zeros(nparam)

    function loss_kernel(x₀, x₁, a, Δt)
        x̄ = zeros(eltype(a), axes(x₁))
        x̃ = zeros(eltype(a), axes(x₁))
        f = zeros(eltype(a), axes(x₁))

        # gradient at the current state; explicit Euler for the first guess
        fθ(f, x₀, a)
        x̃ .= x₀ .+ Δt .* f

        # fixed-point iteration for the implicit midpoint step
        for _ in 1:(method.picard_iterations)
            x̄ .= (x₀ .+ x̃) ./ 2
            fθ(f, x̄, a)
            x̃ .= x₀ .+ Δt .* f
        end

        sum(abs2, x₁ .- x̃)
    end

    xs, ys = _as_vector_of_states(problem.x), _as_vector_of_states(problem.y)

    function loss(a::AbstractVector)
        mapreduce(z -> loss_kernel(z..., a, method.integrator_timestep), +, zip(xs, ys))
    end

    verbose && println("Initial guess...")
    coeffs .= minimize(loss, coeffs, solver)

    for n in 1:nloops(method)
        verbose && println("Iteration #$n...")

        smallinds = abs.(coeffs) .< method.λ
        biginds = .~smallinds

        # the support has stopped changing
        all(coeffs[smallinds] .== 0) && break

        coeffs[smallinds] .= 0

        # regress onto the surviving terms only
        function sparseloss(b::AbstractVector)
            c = zeros(eltype(b), axes(coeffs))
            c[biginds] .= b
            loss(c)
        end

        # `coeffs[biginds]` is a copy, so the result has to be written back through `biginds`.
        # Assigning into the copy is what silently discarded every refit after the first.
        coeffs[biginds] .= minimize(sparseloss, coeffs[biginds], solver)
    end

    return coeffs
end

function identify(problem::TrajectoryData, method::HamiltonianSINDy;
        solver = OptimizerSolver(), verbose = false)
    nd = statedimension(problem)
    iseven(nd) ||
        throw(ArgumentError("a Hamiltonian system needs an even state dimension, got $nd"))

    hfuns = hamiltonian_functions(nd ÷ 2, method.polyorder, method.trigonometric)
    coeffs = sparsify(method, hfuns.ż, problem, solver; verbose)

    HamiltonianSINDyResult(method, coeffs, hfuns)
end

"""
    HamiltonianSINDyVectorField(result)

The identified Hamiltonian vector field `J∇H`, callable as `f(dz, z)`.
"""
struct HamiltonianSINDyVectorField{DT, CT, GHT} <: VectorField
    coefficients::CT
    fθ::GHT

    function HamiltonianSINDyVectorField(coefficients::CT,
            fθ::GHT) where {DT, CT <: AbstractVector{DT}, GHT <: Base.Callable}
        new{DT, CT, GHT}(coefficients, fθ)
    end
end

function HamiltonianSINDyVectorField(result::HamiltonianSINDyResult)
    HamiltonianSINDyVectorField(result.coefficients, result.hamiltonian.ż)
end

function (vectorfield::HamiltonianSINDyVectorField)(dz, z)
    vectorfield.fθ(dz, z, vectorfield.coefficients)
    return dz
end

(vf::HamiltonianSINDyVectorField)(dz, t::Number, z::AbstractVector, params) = vf(dz, z)
