using Flux

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

 

function sparsify_NN(method::SINDy, basis, tdata, solver)
    # add noise to transpose of ẋ b/c we will need the transpose later
    ẋnoisy = (tdata.ẋ)' .+ method.ϵ .* randn(size((tdata.ẋ)'))

    # Make a new training data structure just to be able to pass ẋnoisy to the solve function
    data = TrainingData(tdata.x, ẋnoisy)

    # Pool Data (evaluate library of candidate basis functions on training data)
    Θ = basis(data.x)

    # Ξ is the coefficients of the bases(Θ)
    Ξ = zeros(size(Θ,2), size(data.ẋ, 2))

    function set_model(data, Ξ)
        ld = size(data.x)[1]
        ndim = size(data.x)[1]
        model = ( 
            (W = rand(ld, ndim), b = zeros(ndim)),
            (W = rand(ndim, ld), b = zeros(ld)),
            (W = Ξ, ),
        )
        return model
    end

    # initialize parameters
    initial_model = set_model(data, Ξ)

    # initial optimization for parameters
    model = solve(basis, data, initial_model, solver)
    Ξ = model[3].W

    for n in 1:method.nloops
        println("Iteration #$n...")
        println()
        # find coefficients below λ threshold
        smallinds = abs.(Ξ) .< method.λ
        biginds = .~smallinds

        # check if there are any small coefficients != 0 left
        all(Ξ[smallinds] .== 0) && break

        println("Ξ in sparsification before zeroing: $Ξ")
        println()

        # set all small coefficients to zero
        Ξ[smallinds] .= 0
        println("Ξ in sparsification after zeroing: $Ξ")
        println()
        println("model[3].W in sparsification after Ξ zeroing: $(model[3].W)")
        println()
        
        # Solver for sparsified coefficients
        model = sparse_solve(basis, tdata, model, smallinds)
        Ξ = model[3].W
        println("Sparse Coefficients: $Ξ")
        println()
    end

    return Ξ, model
end

# TODO: Add basis as field of SINDy method

function VectorField(method::SINDy, basis::AbstractBasis, data::TrainingData; solver::AbstractSolver = JuliaLeastSquare())
    # Compute Sparse Regression
    if isa(solver, NNSolver)
        Ξ, model = sparsify_NN(method, basis, data, solver)
        return SINDyVectorField(basis, Ξ), model
    else
        # Pool Data (evaluate library of candidate basis functions on training data)
        Θ = basis(data.x)
        Ξ = sparsify(method, Θ, data.ẋ, solver)
        return SINDyVectorField(basis, Ξ)
    end
end
