using Flux
using Plots

struct SINDy{T} <: SparsificationMethod
    lambda::T
    noise_level::T
    nloops::Int
    coeff::Float64
    batch_size::Int

    function SINDy(; lambda::T = DEFAULT_LAMBDA, noise_level::T = DEFAULT_NOISE_LEVEL, nloops = DEFAULT_NLOOPS, coeff = 0.6, batch_size::Int = 1) where {T}
        new{T}(lambda, noise_level, nloops, coeff, batch_size)
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
function set_model(data, basis)
    encoder = Chain(
    Dense(size(data.x)[1] => 2), 
    # Dense(2 => 2), 
    )

    decoder = Chain(
    Dense(2 => size(data.x)[1]),  
    # Dense(2 => size(data.x)[1])
    )

    # Encode all states at first sample to initialize basis
    Θ = basis(encoder(data.x[:,1]))

    # Ξ is the coefficients of the bases(Θ), it depends on the number of 
    # features (encoded bases), and the number of states for those features to act on
    Ξ = zeros(size(Θ,2), size(encoder(data.ẋ), 1))

    model = ( 
        (W = encoder,),
        (W = decoder,),
        (W = Ξ, ),
    )
    return model
end

function separate_coeffs(model_W, smallinds)
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
    # initialize parameters
    model = set_model(data, basis)

    # initial optimization for parameters
    model, loss_vec = solve(data, method, model, basis, solver)

    # Plot the initial loss array
    display(plot(log.(loss_vec), label = "Initial Optimization Loss"))

    # Initialize smallinds before the loop
    smallinds = falses(size(model[3].W))

    # Array to store the losses of each SINDy loop
    SINDy_loss_array = Vector{Vector{Float64}}()  # Store vectors of losses

    for n in 1:method.nloops
        println("Iteration #$n...")
        println()
        # find coefficients below λ threshold
        smallinds .= abs.(model[3].W) .< method.lambda

        # check if there are any small coefficients != 0 left
        all(model[3].W[smallinds] .== 0) && break

        # set all small coefficients to zero
        model[3].W[smallinds] .= 0
        
        Ξ = separate_coeffs(model[3].W, smallinds)

        # Solver for sparsified coefficients
        model, sparse_loss = sparse_solve(data, method, model, basis, Ξ, smallinds, solver::NNSolver)

        # Store the SINDy loop loss
        push!(SINDy_loss_array, sparse_loss)
        
        println("Sparse Coefficients: $(model[3].W)")
        println()
    end

    # Convert vector of vectors to a single vector
    SINDy_loss_array = vcat(SINDy_loss_array...)

    # Plot the SINDy loss array
    display(plot(log.(SINDy_loss_array), label = "SINDy Optimization Loss"))

    # Iterate once more for optimization without sparsification
    println("Final Iteration...")
    println()

    Ξ = separate_coeffs(model[3].W, smallinds)
    model, final_loss = sparse_solve(data, method, model, basis, Ξ, smallinds, solver::NNSolver)

    display(plot(log.(final_loss), label = "Final Optimization Loss"))
    
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