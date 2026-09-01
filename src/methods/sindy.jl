
"""
    SINDy(; λ = DEFAULT_LAMBDA, nloops = DEFAULT_NLOOPS)

Sparse Identification of Nonlinear Dynamics (Brunton, Proctor & Kutz, PNAS 2016).

Fits `ẋ = Θ(x) Ξ` over a fixed library of candidate functions `Θ` and sparsifies `Ξ` by
sequentially thresholded least squares: threshold every coefficient below `λ` to zero, refit on
the surviving support, and repeat until the support stops changing or `nloops` is reached.

`λ` is a hard threshold on coefficient magnitude applied *after* each least-squares fit — it is
not an ℓ¹ penalty and does not appear in the objective being minimised.
"""
struct SINDy{T} <: SparsificationMethod
    λ::T
    nloops::Int

    function SINDy(; λ::T = DEFAULT_LAMBDA, nloops::Int = DEFAULT_NLOOPS) where {T}
        new{T}(λ, nloops)
    end
end

"""
    sparsify(method::SINDy, Θ, ẋ, solver)

Sequentially thresholded least squares.

Returns the coefficient matrix `Ξ` with `size(Ξ) == (size(Θ, 2), size(ẋ, 1))`.
"""
function sparsify(method::SINDy, Θ, ẋ, solver)
    # initial guess: least squares over the full library
    Ξ = solve(Θ, ẋ', solver)

    for _ in 1:(method.nloops)
        # find coefficients below the λ threshold
        smallinds = abs.(Ξ) .< method.λ

        # the support has stopped changing
        all(Ξ[smallinds] .== 0) && break

        # set all small coefficients to zero
        Ξ[smallinds] .= 0

        # regress the dynamics onto the remaining terms
        for ind in axes(ẋ, 1)
            biginds = .~(smallinds[:, ind])
            Ξ[biginds, ind] .= solve(Θ[:, biginds], ẋ[ind, :], solver)
        end
    end

    return Ξ
end

struct SINDyVectorField{DT, BT, CT} <: VectorField
    basis::BT
    coefficients::CT

    function SINDyVectorField(basis::BT,
            coefficients::CT) where {DT, BT <: AbstractBasis, CT <: AbstractArray{DT}}
        new{DT, BT, CT}(basis, coefficients)
    end
end

function (vf::SINDyVectorField)(dy, y, p, t)
    yPool = evaluate(y, vf.basis)
    ẏ = yPool * vf.coefficients
    @assert axes(dy, 1) == axes(ẏ, 2)
    for index in eachindex(dy)
        dy[index] = ẏ[1, index]
    end
    return dy
end

# TODO: Add basis as field of SINDy method

function VectorField(method::SINDy, basis::AbstractBasis, data::TrainingData;
        solver::AbstractSolver = JuliaLeastSquare())
    # Pool Data (evaluate library of candidate basis functions on training data)
    Θ = evaluate(data.x, basis)

    # Compute Sparse Regression
    Ξ = sparsify(method, Θ, data.ẋ, solver)

    SINDyVectorField(basis, Ξ)
end
