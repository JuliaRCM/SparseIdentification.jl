
struct SINDy{T} <: SparsificationMethod
    λ::T
    ϵ::T
    nloops::Int

    function SINDy(; lambda::T = DEFAULT_LAMBDA, noise_level::T = DEFAULT_NOISE_LEVEL, nloops = DEFAULT_NLOOPS) where {T}
        new{T}(lambda, noise_level, nloops)
    end
end

"sequential least squares"
function sparsify(method::SINDy, Θ, ẋ, solver)
    # add noise
    ẋnoisy = ẋ .+ method.ϵ .* randn(size(ẋ))

    # initial guess: least-squares
    Ξ = solve(Θ, ẋnoisy', solver)

    for _ in 1:method.nloops
        # find coefficients below λ threshold
        smallinds = abs.(Ξ) .< method.λ

        # check if there are any small coefficients != 0 left
        all(Ξ[smallinds] .== 0) && break

        # set all small coefficients to zero
        Ξ[smallinds] .= 0

        # Regress dynamics onto remaining terms to find sparse Ξ
        for ind in axes(ẋnoisy,1)
            biginds = .~(smallinds[:,ind])
            Ξ[biginds,ind] .= solve(Θ[:,biginds], ẋnoisy[ind,:], solver)
        end
    end
    
    return Ξ
end


struct SINDyVectorField{DT,BT,CT} <: VectorField
    basis::BT
    coefficients::CT

    function SINDyVectorField(basis::BT, coefficients::CT) where {DT, BT <: AbstractBasis, CT <: AbstractArray{DT}}
        new{DT,BT,CT}(basis, coefficients)
    end
end


function (vf::SINDyVectorField)(dy, y, p, t)
    yPool = vf.basis(y)
    ẏ = yPool * vf.coefficients
    @assert axes(dy,1) == axes(ẏ,2)
    for index in eachindex(dy)
        dy[index] = ẏ[1, index]
    end
    return dy
end

 
# TODO: Add basis as field of SINDy method

function VectorField(method::SINDy, basis::AbstractBasis, data::TrainingData; solver::AbstractSolver = JuliaLeastSquare())
    # Pool Data (evaluate library of candidate basis functions on training data)
    Θ = basis(data.x)

    # Compute Sparse Regression
    Ξ = sparsify(method, Θ, data.ẋ, solver)

    SINDyVectorField(basis, Ξ)
end
