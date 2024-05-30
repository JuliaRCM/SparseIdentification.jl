using Flux
using Plots
using DelimitedFiles

"""
    SINDy{T}(; lambda::T = DEFAULT_LAMBDA, noise_level::T = DEFAULT_NOISE_LEVEL, nloops = DEFAULT_NLOOPS, l_dim = 1, coeff = 0.6, batch_size::Int = 1)

Defines the SINDy sparsification method.

# Fields
- `lambda::T`: Sparsification threshold.
- `noise_level::T`: Level of noise to be added.
- `nloops::Int`: Number of sparsification loops.
- `l_dim::Int`: Dimension of the latent space.
- `coeff::Float64`: Coefficient for SINDy regularization.
- `batch_size::Int`: Batch size for training.

# Constructor
- `SINDy(lambda, noise_level, nloops, l_dim, coeff, batch_size)`: Creates a `SINDy` object with the specified parameters.
"""
struct SINDy{T} <: SparsificationMethod
    lambda::T
    noise_level::T
    nloops::Int
    l_dim::Int
    coeff::Float64
    batch_size::Int

    function SINDy(; lambda::T = DEFAULT_LAMBDA, noise_level::T = DEFAULT_NOISE_LEVEL, nloops = DEFAULT_NLOOPS, l_dim = 1, coeff = 0.6, batch_size::Int = 1) where {T}
        new{T}(lambda, noise_level, nloops, l_dim, coeff, batch_size)
    end
end

"""
    sparsify(method::SINDy, Θ, ẋ, solver)

Performs sparsification using the SINDy method.

# Arguments
- `method::SINDy`: The SINDy structure parameters.
- `Θ`: The library of candidate basis functions.
- `ẋ`: The time derivatives of the state variables.
- `solver`: The solver to be used for finding the coefficients.

# Returns
- `Ξ`: The sparse coefficient matrix.
"""
function sparsify(method::SINDy, Θ, ẋ, solver)
    # add noise
    ẋnoisy = ẋ .+ method.noise_level .* randn(size(ẋ))

    # initial guess: least-squares
    Ξ = solve(Θ, ẋnoisy', solver)

    for n in 1:method.nloops
        println("SINDy cycle #$n...")

        # find coefficients below lambda threshold
        smallinds = abs.(Ξ) .< method.lambda

        # check if there are any small coefficients != 0 left
        all(Ξ[smallinds] .== 0) && break

        # set all small coefficients to zero
        Ξ[smallinds] .= 0

        # Regress dynamics onto remaining terms to find sparse Ξ
        for ind in axes(ẋnoisy, 1)
            biginds = .~(smallinds[:, ind])
            Ξ[biginds, ind] .= solve(Θ[:, biginds], ẋnoisy[ind, :], solver)
        end
    end
    
    return Ξ
end

"""
    SINDyVectorField{DT,BT,CT}(basis::BT, coefficients::CT)

Defines a vector field represented by the SINDy method.

# Fields
- `basis::BT`: The basis functions.
- `coefficients::CT`: The coefficients of the basis functions.

# Constructor
- `SINDyVectorField(basis, coefficients)`: Creates a `SINDyVectorField` object with the specified basis and coefficients.
"""
struct SINDyVectorField{DT,BT,CT} <: VectorField
    basis::BT
    coefficients::CT

    function SINDyVectorField(basis::BT, coefficients::CT) where {DT, BT <: AbstractBasis, CT <: AbstractArray{DT}}
        new{DT,BT,CT}(basis, coefficients)
    end
end

"""
    (vf::SINDyVectorField)(dy, y, p, t)

Evaluates the vector field at a given state.

# Arguments
- `vf::SINDyVectorField`: The SINDy vector field.
- `dy`: The output array for the derivatives.
- `y`: The state variables.
- `p`: Additional parameters (not used) but required by integrator library.
- `t`: Time (not used) but required by integrator library.

# Returns
- `dy`: The updated derivatives.
"""
function (vf::SINDyVectorField)(dy, y, p, t)
    yPool = vf.basis(y)
    ẏ = yPool * vf.coefficients
    @assert axes(dy, 1) == axes(ẏ, 2)
    for index in eachindex(dy)
        dy[index] = ẏ[1, index]
    end
    return dy
end

"""
    newtSINDy_Ham_grad(dy, y, p, t)

Evaluates the gradient of a Hamiltonian system using the Classical SINDy optimized coefficients.

# Arguments
- `dy`: The output array for the derivatives.
- `y`: The state variables.
- `p`: Parameters containing the dictionary results from VectorField_Newt_Ham()
- `t`: Time (not used).

# Returns
- `dy`: The updated derivatives.
"""
function newtSINDy_Ham_grad(dy, y, p, t)
    basis_pool = zeros(eltype(y), length(p.basis))
    # evaluate the basis functions at the current state y
    p.bases_gen_func(basis_pool, y)
    # calculate the time derivative of the state variables using the optimized coefficients and basis values
    ẏ = transpose(basis_pool) * p.Ξ
    # update the derivatives
    @assert axes(dy, 1) == axes(ẏ, 2)
    for index in eachindex(dy)
        dy[index] = ẏ[1, index]
    end
    return dy
end

"""
    set_model(data, basis, l_dim)

Initializes a model with random parameters and Ξ = 0.

# Arguments
- `data`: The training data.
- `basis`: The basis functions.
- `l_dim`: The latent dimension.

# Returns
- `model`: The initialized model.
"""
function set_model(data, basis, l_dim)
    encoder = Chain(
        Dense(size(data.x)[1] => l_dim)
    )

    decoder = Chain(
        Dense(l_dim => size(data.x)[1])
    )

    # Encode all states at first sample to initialize basis
    Θ = basis(encoder(data.x[:, 1]))

    # Ξ is the coefficients of the bases(Θ), it depends on the number of 
    # features (encoded bases), and the number of states for those features to act on
    Ξ = zeros(size(Θ, 2), size(encoder(data.ẋ), 1))

    model = ( 
        (W = encoder,),
        (W = decoder,),
        (W = Ξ,)
    )
    return model
end

"""
    separate_coeffs(model_W, smallinds)

Separates non-zero coefficients from the model weights.

# Arguments
- `model_W`: The model weights.
- `smallinds`: Indices of small coefficients to be set to zero.

# Returns
- `Ξ`: A vector of vectors containing non-zero coefficients.
"""
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

"""
    sparsify_NN(method::SINDy, basis, data, solver, script_name)

Performs sparsification using a neural network with the SINDy method.

# Arguments
- `method::SINDy`: SINDy parameters.
- `basis`: The basis functions.
- `data`: The training data.
- `solver`: The solver to be used for optimization.
- `script_name`: The name of the script for saving files.

# Returns
- `Ξ`: The sparse coefficient matrix.
- `model`: The trained model.
"""
function sparsify_NN(method::SINDy, basis, data, solver, script_name)
    # if no directory create one
    nn_dir = "nn_$script_name"
    if !isdir(nn_dir)
        mkdir(nn_dir)
    end

    # initialize parameters
    model = set_model(data, basis, method.l_dim)

    # initial optimization for parameters
    model, loss_vec = solve(data, method, model, basis, solver)

    # Plot the initial loss array
    display(plot(log.(loss_vec), label = "Initial Optimization Loss", xlabel = "Iterations", ylabel = "Log-Loss"))

    # save file name according to parameters
    initialLoss_file = joinpath(nn_dir, "initial_Loss_thr_$(method.lambda)_noise_$(method.noise_level)_coeff_$(method.coeff)_batch_$(method.batch_size).csv")
    writedlm(initialLoss_file, loss_vec, ',')

    # Initialize smallinds before the loop
    smallinds = falses(size(model[3].W))

    # Array to store the losses of each SINDy loop
    SINDy_loss_array = Vector{Vector{Float64}}()  # Store vectors of losses
    
    for n in 1:method.nloops
        println("SINDy cycle #$n...")
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
    display(plot(log.(SINDy_loss_array), label = "SINDy Optimization Loss", xlabel="Iterations", ylabel="Log-Loss"))

    # save file name according to parameters
    sindy_Loss_file = joinpath(nn_dir, "sindy_Loss_thr_$(method.lambda)_noise_$(method.noise_level)_coeff_$(method.coeff)_batch_$(method.batch_size).csv")
    writedlm(sindy_Loss_file, SINDy_loss_array, ',')

    # Iterate once more for optimization without sparsification
    println("Final Iteration...")
    println()

    Ξ = separate_coeffs(model[3].W, smallinds)
    model, final_loss = sparse_solve(data, method, model, basis, Ξ, smallinds, solver::NNSolver)

    display(plot(log.(final_loss), label = "Final Optimization Loss", xlabel="Iterations", ylabel="Log-Loss"))

    # save file name according to parameters
    final_Loss_file = joinpath(nn_dir, "final_Loss_thr_$(method.lambda)_noise_$(method.noise_level)_coeff_$(method.coeff)_batch_$(method.batch_size).csv")
    writedlm(final_Loss_file, final_loss, ',')
    
    Ξ = model[3].W
    return Ξ, model
end

"""
    VectorField(method::SINDy, basis::AbstractBasis, data::TrainingData; solver::AbstractSolver = JuliaLeastSquare(), script_name = "")

Creates a SINDy vector field and trains a model to find the sparse coefficients.

# Arguments
- `method::SINDy`: SINDy parameters object.
- `basis::AbstractBasis`: The basis functions to use.
- `data::TrainingData`: The training data.
- `solver::AbstractSolver`: The solver for optimization (default is `JuliaLeastSquare()`).
- `script_name::String`: The name of the script for saving files (optional).

# Returns
- `SINDyVectorField`: The trained SINDy vector field.
- `model`: The trained model (only returned if using `NNSolver`).
"""
function VectorField(method::SINDy, basis::AbstractBasis, data::TrainingData; solver::AbstractSolver = JuliaLeastSquare(), script_name = "")
    if isa(solver, NNSolver)
        Ξ, model = sparsify_NN(method, basis, data, solver, script_name)
        return SINDyVectorField(basis, Ξ), model
    else
        Θ = basis(data.x)
        Ξ = sparsify(method, Θ, data.ẋ, solver)
        return SINDyVectorField(basis, Ξ)
    end
end

"""
    VectorField_Newt_Ham(method::SINDy, z::Vector{Symbolics.Num}, basis::Vector{Symbolics.Num}, data::TrainingData; solver::AbstractSolver = JuliaLeastSquare())

Creates a Hamiltonian vector field and optimizes it with the Classical SINDy method.

# Arguments
- `method::SINDy`: SINDy parameters object.
- `z::Vector{Symbolics.Num}`: The state variables in symbolic form.
- `basis::Vector{Symbolics.Num}`: The basis functions in symbolic form.
- `data::TrainingData`: The training data.
- `solver::AbstractSolver`: The solver for optimization (default is `JuliaLeastSquare()`).

# Returns
- `sol_fields`: A dictionary containing:
  - `bases_gen_func`: The function to generate basis values.
  - `Ξ`: The sparse coefficient matrix.
  - `basis`: The basis functions.
"""
function VectorField_Newt_Ham(method::SINDy, z::Vector{Symbolics.Num}, basis::Vector{Symbolics.Num}, data::TrainingData; solver::AbstractSolver = JuliaLeastSquare())
    bases_gen_func = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(build_function(basis, z)[2]))
    Θ = zeros(size(data.x, 2), length(basis))
    res = zeros(length(basis))
    for i in axes(data.x, 2)
        bases_gen_func(res, data.x[:,i])
        Θ[i,:] .= res
    end    
    Ξ = sparsify(method, Θ, data.ẋ, solver)

    sol_fields = (
        bases_gen_func = bases_gen_func,
        Ξ = Ξ,
        basis = basis,
    )
    
    return sol_fields
end
