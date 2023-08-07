using Flux

struct SINDy{T} <: SparsificationMethod
    lambda::T
    noise_level::T
    nloops::Int

    function SINDy(; lambda::T = DEFAULT_LAMBDA, noise_level::T = DEFAULT_NOISE_LEVEL, nloops = DEFAULT_NLOOPS) where {T}
        new{T}(lambda, noise_level, nloops)
    end
end

"sequential least squares"
function sparsify(method::SINDy, Θ, ẋ, solver)
    # add noise
    ẋnoisy = ẋ .+ method.noise_level .* randn(size(ẋ))

    # initial guess: least-squares
    Ξ = solve(Θ, ẋnoisy', solver)

    for _ in 1:method.nloops
        # find coefficients below lambda threshold
        smallinds = abs.(Ξ) .< method.lambda

        # check if there are any small coefficients != 0 left
        all(Ξ[smallinds] .== 0) && break

        # set all small coefficients to zero
        Ξ[smallinds] .= 0

        # Regress dynamics onto remaining terms to find sparse Ξ
        for ind in axes(ẋnoisy,1)
            biginds = .~(smallinds[:,ind])

            #TODO: maybe this needs to be ẋnoisy' i.e. transpose
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

 
# Initialize a model with random parameters and Ξ = 0
function set_model(data, Ξ)
    encoder = Chain(
    Dense(size(data.x)[1] => 16, sigmoid), 
    Dense(16 => 8, sigmoid), 
    Dense(8 => 4, sigmoid),
    Dense(4 => size(data.x)[1])
    )

    decoder = Chain(
    Dense(size(data.x)[1] => 16, sigmoid),  
    Dense(16 => 8, sigmoid),
    Dense(8 => 4, sigmoid),
    Dense(4 => size(data.x)[1])
    )

    model = ( 
        (W = encoder,),
        (W = decoder,),
        (W = Ξ, ),
    )
    return model
end

function separate_coeffs(model_W, smallinds)
    # Ξ = Tuple{Vector{Float64}, Vector{Float64}}[]
    Ξ = Vector{Vector{Float64}}()

    for ind in 1:size(model_W, 2)
        column = model_W[:, ind]
        non_zero_indices = findall(.~smallinds[:, ind])
        non_zero_values = column[non_zero_indices]
        push!(Ξ, non_zero_values)
    end
    
    return Ξ
end

function sparsify_NN(method::SINDy, basis, data, solver)
    # Pool Data (evaluate library of candidate basis functions on training data)
    # Values of basis functions on all samples of the training data states
    Θ = basis(data.x)

    # Ξ is the coefficients of the bases(Θ), it depends on the number of 
    # features (bases), and the number of states for those features to act on
    Ξ = zeros(size(Θ,2), size(data.ẋ, 1))

    # initialize parameters
    model = set_model(data, Ξ)

    # initial optimization for parameters
    model = solve(data, model, basis, solver)

    for n in 1:method.nloops
        println("Iteration #$n...")
        println()
        # find coefficients below λ threshold
        smallinds = abs.(model[3].W) .< method.lambda

        # check if there are any small coefficients != 0 left
        all(model[3].W[smallinds] .== 0) && break

        # set all small coefficients to zero
        model[3].W[smallinds] .= 0
        
        Ξ = separate_coeffs(model[3].W, smallinds)

        # Solver for sparsified coefficients
        model = sparse_solve(basis, data, model, Ξ, smallinds)
        
        println("Sparse Coefficients: $(model[3].W)")
        println()
    end

    # Iterate once more for optimization without sparsification
    model = sparse_solve(basis, data, model, Ξ, smallinds)
    
    Ξ = model[3].W
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
