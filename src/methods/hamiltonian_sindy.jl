
"""
    HamiltonianSINDy(; λ, nloops, polyorder, trigonometric, integrator_timestep, picard_iterations)

Sparse identification of a *Hamiltonian* system.

A scalar Hamiltonian is parametrised over a library of candidate functions,
`H(z; a) = Σₖ aₖ φₖ(z)`, and the vector field is obtained as `ż = J ∇H(z)`. The identified
dynamics is therefore Hamiltonian **by construction** rather than by penalty.

The coefficients are fitted by matching the flow map: minimise `Σⱼ ‖Φ_Δt(zⱼ; a) − zⱼ₊₁‖²` where
`Φ_Δt` is an implicit-midpoint step. This is nonlinear in `a` and needs an optimiser.

!!! note "Matching the vector field is cheaper when `ż` is available"
    `J∇H` is *linear* in `a`, so fitting against measured derivatives is an ordinary linear sparse
    regression — one `\\` plus thresholding — with no optimiser at all. That formulation is not
    yet implemented here; see the package documentation.
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

"""
    sparsify(method::HamiltonianSINDy, fθ, data::TrajectoryData, solver; verbose = false)

Sequentially thresholded regression of the Hamiltonian coefficients against the flow map.
"""
function sparsify(
        method::HamiltonianSINDy, fθ, data::TrajectoryData, solver; verbose = false)
    # `fθ` is built for `d` degrees of freedom, i.e. `2d` phase-space variables, and
    # `calculate_nparams` takes that same `d`. Passing the full state dimension here instead is
    # what made the optimiser search 212 coefficients for a basis that reads 58 of them.
    d = statedimension(data) ÷ 2
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

    function loss(a::AbstractVector)
        mapreduce(z -> loss_kernel(z..., a, method.integrator_timestep), +, zip(data.x, data.y))
    end

    verbose && println("Initial guess...")
    coeffs .= minimize(loss, coeffs, solver)

    for n in 1:(method.nloops)
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

struct HamiltonianSINDyVectorField{DT, CT, GHT} <: VectorField
    coefficients::CT
    fθ::GHT

    function HamiltonianSINDyVectorField(coefficients::CT,
            fθ::GHT) where {DT, CT <: AbstractVector{DT}, GHT <: Base.Callable}
        new{DT, CT, GHT}(coefficients, fθ)
    end
end

function VectorField(method::HamiltonianSINDy, data::TrajectoryData;
        solver = OptimizerSolver(), verbose = false)
    nd = statedimension(data)
    iseven(nd) ||
        throw(ArgumentError("a Hamiltonian system needs an even state dimension, got $nd"))

    # returns a function that evaluates the Hamiltonian vector field J∇H
    fθ = hamilGrad_func_builder(nd ÷ 2, method.polyorder, method.trigonometric)

    coeffs = sparsify(method, fθ, data, solver; verbose)

    HamiltonianSINDyVectorField(coeffs, fθ)
end

function (vectorfield::HamiltonianSINDyVectorField)(dz, z)
    vectorfield.fθ(dz, z, vectorfield.coefficients)
    return dz
end

(vectorfield::HamiltonianSINDyVectorField)(dz, z, p, t) = vectorfield(dz, z)
