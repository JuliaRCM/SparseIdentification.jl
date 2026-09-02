
"""
    SINDy(basis; λ = 0.05, nloops = 10)

Sparse Identification of Nonlinear Dynamics (Brunton, Proctor & Kutz, PNAS 2016).

Fits `ẋ = Θ(x) Ξ` over the fixed library `basis` and sparsifies `Ξ` by sequentially thresholded
least squares: threshold every coefficient below `λ` to zero, refit on the surviving support, and
repeat until the support stops changing or `nloops` is reached.

`λ` is a hard threshold on coefficient magnitude applied *after* each least-squares fit — it is not
an ℓ¹ penalty and does not appear in the objective being minimised.

Apply it to a [`TrainingData`](@ref) problem with [`identify`](@ref).

# Examples

```jldoctest
julia> A = [-0.1 2.0; -2.0 -0.1];

julia> x = randn(2, 500);

julia> result = identify(TrainingData(x, A * x), SINDy(CompoundBasis(polyorder = 3)));

julia> isapprox(parameters(result)[2:3, :], A'; atol = 1e-10)
true
```
"""
struct SINDy{T, BT <: AbstractBasis} <: SparsificationMethod
    basis::BT
    λ::T
    nloops::Int

    function SINDy(basis::BT; λ::T = DEFAULT_LAMBDA,
            nloops::Int = DEFAULT_NLOOPS) where {T, BT <: AbstractBasis}
        new{T, BT}(basis, λ, nloops)
    end
end

GeometricBase.name(::SINDy) = "SINDy"
GeometricBase.description(::SINDy) = "Sparse Identification of Nonlinear Dynamics"
function GeometricBase.reference(::SINDy)
    "Brunton, Proctor & Kutz, PNAS 113(15), 3932-3937, 2016. doi:10.1073/pnas.1517384113"
end

# The identified model is a general vector field: nothing constrains it to conserve anything.
GeometricBase.issymplectic(::SINDy) = false
GeometricBase.isenergypreserving(::SINDy) = false
# The regression is a direct linear solve.
GeometricBase.isexplicit(::SINDy) = true
GeometricBase.isimplicit(::SINDy) = false

"""
    SINDyResult

The outcome of [`identify`](@ref) with [`SINDy`](@ref).

Carries the coefficient matrix and the method that produced it. Reach the coefficients with
`parameters` and the basis they refer to with `basis`; convert to a
`GeometricEquations.ODEProblem` to integrate the identified system.
"""
struct SINDyResult{DT, CT <: AbstractArray{DT}, MT <: SINDy}
    method::MT
    coefficients::CT

    function SINDyResult(method::MT,
            coefficients::CT) where {DT, MT <: SINDy, CT <: AbstractArray{DT}}
        new{DT, CT, MT}(method, coefficients)
    end
end

GeometricBase.parameters(result::SINDyResult) = result.coefficients
# The basis is the method's; a result cannot refer to a different one than it was fitted with.
GeometricBase.basis(result::SINDyResult) = result.method.basis
GeometricBase.datatype(::SINDyResult{DT}) where {DT} = DT
method(result::SINDyResult) = result.method

"""
    nterms(result)

The number of library terms retained after sparsification.
"""
nterms(result::SINDyResult) = count(!iszero, result.coefficients)

function Base.show(io::IO, result::SINDyResult)
    print(io, "SINDy result: ", nterms(result), " of ",
        length(result.coefficients), " coefficients retained")
end

"""
    sparsify(method::SINDy, Θ, ẋ, solver)

Sequentially thresholded least squares.

Returns the coefficient matrix `Ξ` with `size(Ξ) == (size(Θ, 2), size(ẋ, 1))`.
"""
function sparsify(method::SINDy, Θ, ẋ, solver)
    # initial guess: least squares over the full library
    Ξ = solve(Θ, ẋ', solver)

    for _ in 1:nloops(method)
        # find coefficients below the λ threshold
        smallinds = abs.(Ξ) .< sparsity_threshold(method)

        # the support has stopped changing
        all(Ξ[smallinds] .== 0) && break

        # set all small coefficients to zero
        Ξ[smallinds] .= 0

        # regress the dynamics onto the remaining terms, one state component at a time
        for ind in axes(ẋ, 1)
            biginds = .~(smallinds[:, ind])
            Ξ[biginds, ind] .= solve(Θ[:, biginds], ẋ[ind, :], solver)
        end
    end

    return Ξ
end

function identify(problem::TrainingData, method::SINDy;
        solver::AbstractSolver = JuliaLeastSquare())
    # evaluate the library of candidate functions on the training data
    Θ = evaluate(problem.x, method.basis)

    # `sparsify` indexes the derivatives as a matrix, one column per snapshot. `TrainingData`
    # also accepts a vector of state vectors, which is the shape `TrainingData(solution)` builds.
    Ξ = sparsify(method, Θ, _as_matrix_of_states(problem.ẋ), solver)

    SINDyResult(method, Ξ)
end

"""
    SINDyVectorField(result)

The identified vector field, callable as `f(dy, y, params, t)`.
"""
struct SINDyVectorField{DT, BT, CT} <: VectorField
    basis::BT
    coefficients::CT

    function SINDyVectorField(basis::BT,
            coefficients::CT) where {DT, BT <: AbstractBasis, CT <: AbstractArray{DT}}
        new{DT, BT, CT}(basis, coefficients)
    end
end

SINDyVectorField(result::SINDyResult) = SINDyVectorField(basis(result), result.coefficients)

function (vf::SINDyVectorField)(dy, y)
    yPool = evaluate(y, vf.basis)
    ẏ = yPool * vf.coefficients
    @assert axes(dy, 1) == axes(ẏ, 2)
    for index in eachindex(dy)
        dy[index] = ẏ[1, index]
    end
    return dy
end

# GeometricEquations calls a vector field as `v(v, t, q, params)`. That is the only calling
# convention supported: the SciML `f(du, u, p, t)` order is deliberately not provided, since
# accepting both makes a transposed call silently do the wrong thing rather than error.
(vf::SINDyVectorField)(dy, t::Number, y::AbstractVector, params) = vf(dy, y)
