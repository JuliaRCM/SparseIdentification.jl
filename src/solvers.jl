
using Optim
using Flux
using Random
using Zygote


"""
Abstract type for all solvers.
"""
abstract type AbstractSolver end

"""
Abstract type for nonlinear solvers.
"""
abstract type NonlinearSolver <: AbstractSolver end

"""
Solver using Julia's least squares method.
"""
struct JuliaLeastSquare <: AbstractSolver end

"""
Solves a linear system using Julia's least squares method.

# Arguments
- `Θ`: The matrix of basis functions.
- `ẋ`: The vector of time derivatives.

# Returns
- The solution vector.
"""
function solve(Θ, ẋ, ::JuliaLeastSquare)
    Θ \ ẋ
end

"""
Solver using an optimization method from the `Optim` package.

# Fields
- `method`: The optimization method to be used. The default method is `BFGS()`.
"""
struct OptimSolver <: NonlinearSolver 
    method

    function OptimSolver(method = BFGS())
        new(method)
    end
end

"""
Optimizes a loss function using the specified solver.

# Arguments
- `loss`: The loss function to be minimized.
- `x₀`: Initial guess for the solution.
- `solver::OptimSolver`: The solver to be used.

# Returns
- The minimizer vector of the loss function.
"""
function optimize(loss, x₀, solver::OptimSolver)
    result = Optim.optimize(loss, x₀, solver.method, Optim.Options(iterations=500))
    println(result)
    return result.minimizer
end

"""
Solves a nonlinear system using a nonlinear solver.

# Arguments
- `Θ`: The matrix of basis functions.
- `ẋ`: The vector of time derivatives.
- `solver::NonlinearSolver`: The solver to be used.

# Returns
- The minimizing coefficients matrix.
"""
function solve(Θ, ẋ, solver::NonlinearSolver)
    x₀ = zeros(size(Θ,2), size(ẋ,2))
    loss(x) = mapreduce(y -> y^2, +, ẋ .- Θ * x)
    result = optimize(x -> loss(reshape(x, size(x₀))), x₀[:], solver)
    return reshape(result, size(x₀))
end

"""
Solver using a neural network-based method.
# Fields
- `optimizer`: The optimizer to be used for training the neural network. The default optimizer is `Adam()`.
"""
struct NNSolver <: NonlinearSolver 
    optimizer

    function NNSolver(optimizer = Adam())
        new(optimizer)
    end
end

"""
Calculates batched Jacobian for a model layer.

# Arguments
- `model_layer`: The layer of the model.
- `x_batch`: The batch of input data.

# Returns
- The batched Jacobian.
"""
function batched_jacobian(model_layer, x_batch)
    output_dim = size(model_layer(x_batch[:, 1]))[1]
    batch_size = size(x_batch, 2)
    batch_jac = zeros(output_dim, batch_size, size(x_batch, 1))
    
    for i in 1:batch_size
        x_input = x_batch[:, i]
        jac = Flux.jacobian(model_layer, x_input)[1]
        batch_jac[:, i, :] = jac
    end
    
    return batch_jac
end

"""
Updates the model coefficients based on the given indices and values.

# Arguments
- `model_W`: The model weights to be updated.
- `smallinds`: The indices of coefficients below sparsification threshold.
- `Ξ`: The values to update the weights with.
"""
function update_model_coeffs!(model_W, smallinds, Ξ)
    for ind in 1:size(model_W, 2)
        non_zero_indices = findall(.~smallinds[:, ind])
        @views model_W[:, ind][non_zero_indices] .= Ξ[ind]
    end
end

"""
Calculates `ż` from `dz/dx` and `ẋ`.

# Arguments
- `enc_jac_batch`: Encoder Jacobian (dz/dx) of a batch.
- `ẋ_batch`: Batch of time derivatives.

# Returns
- The calculated `ż`.
"""
function enc_ż(enc_jac_batch, ẋ_batch)
    # Size is equal to encoded features and number of batches
    ż_ref = zero(enc_jac_batch[:,:, 1])
    for b in 1:size(enc_jac_batch, 2)
        ż_ref[:, b] = enc_jac_batch[:, b, :] * ẋ_batch[:, b]
    end
    return ż_ref
end

"""
This function calculates the time derivatives of the decoded state variables (`ẋ`) using the Jacobian of the decoder `dx/dz` and the time derivatives of the latent variables (`ż`).

# Arguments
- `dec_jac_batch`: A 3D array containing the Jacobian matrices of the decoder for each batch.
- `ż`: A 2D array containing the time derivatives of the latent variables for each batch.

# Returns
- A 2D array containing the time derivatives of the decoded state variables.
"""
function dec_ẋ(dec_jac_batch, ż)
    # Size is equal to decoded features and number of batches
    dec_mult_ẋ = Zygote.Buffer(dec_jac_batch[:,:, 1])
    for i in 1:size(dec_jac_batch, 2)
        dec_mult_ẋ[:, i] = dec_jac_batch[:,i,:] * ż[:,i]
    end
    return copy(dec_mult_ẋ)
end

"""
This function calculates the time derivatives of the latent variables (`ż`) using the SINDy coefficients and the basis functions.

# Arguments
- `enc_x_batch`: A 2D array containing the encoded state variables for each batch.
- `Θ`: A 2D array containing the encoded values of the basis functions for each batch.
- `Ξ`: A vector containing the SINDy coefficients.
- `smallinds`: A 2D array indicating the coefficients below sparsification threshold.

# Returns
- A 2D array containing the time derivatives of the latent variables.
"""
function set_ż_SINDY(enc_x_batch, Θ, Ξ, smallinds)
    ż_SINDy = Zygote.Buffer(zeros(size(enc_x_batch, 1), size(Θ,1)))
    for ind in axes(enc_x_batch, 1)
        # non-zero coefficients of the ind state
        biginds = .~(smallinds[:, ind])
        ż_SINDy[ind,:] = Θ[:, biginds] * Ξ[ind]
    end
    return copy(ż_SINDy)
end

"""
# Solve the optimization problem for the given data and model using a neural network solver

This function trains a neural network model to find the optimal parameters for the given data using a specified optimization method.

# Arguments
- `data`: The training data.
- `method`: The method struct containing relevant parameters.
- `model`: The neural network model.
- `basis`: The basis functions used for SINDy.
- `solver::NNSolver`: The chosen neural network solver.

# Returns
- The trained model and an array containing the loss for each epoch.
"""
function solve(data, method, model, basis, solver::NNSolver)
    total_samples = size(data.x)[2]
    num_batches = ceil(Int, total_samples / method.batch_size)

    # Flux gradient has problem working with the structure data directly
    x = Float32.(data.x)
    ẋ = Float32.(data.ẋ)

    # Coefficients for the loss_kernel terms
    alphas = round(sum(abs2, x) / sum(abs2, ẋ), sigdigits = 3)

    function loss(model, x_batch, ẋ_batch, ż_ref, dec_jac_batch, basis, method, alphas)
        # Compute the reconstruction loss for the entire batch
        L_r = sum(abs2, model[2].W(model[1].W(x_batch)) .- x_batch)

        # Values of basis functions on the current batch of the encoded training data states
        Θ_batch = basis(model[1].W(x_batch))

        # Encoded SINDy gradient
        ż_SINDy = (Θ_batch * model[3].W)'

        # Loss from difference between encoded variables through SINDy and reference
        L_ż = alphas / 10 * sum(abs2, ż_ref .- ż_SINDy)
    
        # Compute the loss terms involving dec_jac_batch and Θ_batch
        dec_mult_ẋ = dec_ẋ(dec_jac_batch, ż_SINDy)

        # Loss from difference between decoded variables through SINDy and reference
        L_ẋ = alphas * sum(abs2, dec_mult_ẋ  .- ẋ_batch)

        # Compute the total loss for the entire batch
        batchLoss = L_r + L_ż + L_ẋ

        # Mean of the coefficients averaged
        L_c = sum(abs.(model[3].W)) / length(model[3].W)

        batch_loss_average = batchLoss / size(x_batch, 2) + method.coeff * L_c
    
        return batch_loss_average
    end

    # Array to store the losses
    epoch_loss_array = Vector{Float64}()

    # Set up the optimizer's state
    opt_state = Flux.setup(solver.optimizer, model)

    for epoch in 1:2000
        epoch_loss = 0.0
        # Shuffle the data indices for each epoch
        shuffled_indices = shuffle(1:total_samples)

        for batch in 1:num_batches
            # Get the indices for the current batch
            batch_start = (batch - 1) * method.batch_size + 1
            batch_end = min(batch * method.batch_size, total_samples)
            batch_indices = shuffled_indices[batch_start:batch_end]

            # Extract the data for the current batch
            x_batch = x[:, batch_indices]
            ẋ_batch = ẋ[:, batch_indices]

            # Derivatives of the encoder and decoder
            enc_jac_batch = batched_jacobian(model[1].W, x_batch)
            dec_jac_batch = batched_jacobian(model[2].W, model[1].W(x_batch))

            # Compute the loss terms involving enc_jac_batch and Θ_batch
            ż_ref = enc_ż(enc_jac_batch, ẋ_batch)

            # Compute gradients using Flux
            gradients = Flux.gradient(model -> loss(model, x_batch, ẋ_batch, ż_ref, dec_jac_batch, basis, method, alphas), model)[1]

            # Update the parameters
            Flux.Optimise.update!(opt_state, model, gradients)

            # Accumulate the loss for the current batch
            epoch_loss += loss(model, x_batch, ẋ_batch, ż_ref, dec_jac_batch, basis, method, alphas)
        end
        # Compute the average loss for the epoch
        epoch_loss /= num_batches

        # Store the epoch loss
        push!(epoch_loss_array, epoch_loss)

        # Print loss after some iterations
        if epoch % 100 == 0
            println("Epoch $epoch: Average Loss: $epoch_loss")
            println("Epoch $epoch: Coefficents: $(model[3].W)")
            println()
        end
    end
    return model, epoch_loss_array
end

"""
# Sparsify and solve the optimization problem for the given data and model

This function trains a neural network model to find sparse optimal parameters for the given data using specified method parameters, basis functions, and SINDy coefficients.

# Arguments
- `data`: The training data.
- `method`: The method struct containing relevant parameters.
- `model`: The neural network model.
- `basis`: The basis functions used for SINDy.
- `Ξ`: A vector containing the SINDy coefficients.
- `smallinds`: A 2D array indicating the coefficients below sparsification threshold
- `solver::NNSolver`: The chosen neural network solver.

# Returns
- The trained model and an array containing the loss for each epoch.
"""
function sparse_solve(data, method, model, basis, Ξ, smallinds, solver::NNSolver)
    total_samples = size(data.x)[2]
    num_batches = ceil(Int, total_samples / method.batch_size)

    # Flux gradient has problem working with the structure data
    x = Float32.(data.x)
    ẋ = Float32.(data.ẋ)

    # Coefficients for the loss_kernel terms
    alphas = round(sum(abs2, x) / sum(abs2, ẋ), sigdigits = 3)

    # Total loss
    function sparse_loss(enc_paras, dec_paras, Ξ, model_coeffs, basis, smallinds, x_batch, ẋ_batch, ż_ref, dec_jac, method, alphas)
        # Values of basis functions on the current sample of the encoded training data states
        Θ = basis(enc_paras(x_batch))

        # Encoded SINDy gradient
        enc_x_batch = enc_paras(x_batch)
        ż_SINDy = set_ż_SINDY(enc_x_batch, Θ, Ξ, smallinds)

        # Decoded SINDy gradient
        ẋ_SINDy = dec_ẋ(dec_jac, ż_SINDy)

        # Reconstruction loss from encoded-decoded xᵢₙ
        L_r = sum(abs2, dec_paras(enc_paras(x_batch)) .- x_batch)

        # Ξ[ind]: gives the values of the coefficients of a state to be 
        # multiplied with the Θ biginds basis functions at that state, 
        # to give the encoded gradients from the SINDy method 
        L_ż = alphas / 10 * sum(abs2, ż_ref .- ż_SINDy)
        
        # Decoded gradients from SINDy and reference
        L_ẋ = alphas * sum(abs2, ẋ_batch .- ẋ_SINDy)

        batchLoss = L_r + L_ż + L_ẋ
    
        # Mean of the coefficients averaged
        L_c = sum(abs.(model_coeffs)) / length(model_coeffs)

        return batchLoss / size(x_batch, 2) + method.coeff * L_c
    end
    
    # Array to store the losses
    epoch_loss_array = Vector{Float64}()

    # Set up the optimizer's state
    opt_state = Flux.setup(solver.optimizer, (model[1].W, model[2].W, Ξ))

    for epoch in 1:500
        epoch_loss = 0.0
        # Shuffle the data indices for each epoch
        shuffled_indices = shuffle(1:total_samples)

        for batch in 1:num_batches
            # Get the indices for the current batch
            batch_start = (batch - 1) * method.batch_size + 1
            batch_end = min(batch * method.batch_size, total_samples)
            batch_indices = shuffled_indices[batch_start:batch_end]

            # Extract the data for the current batch
            x_batch = x[:, batch_indices]
            ẋ_batch = ẋ[:, batch_indices]

            # Derivatives of the encoder and decoder
            enc_jac_batch = batched_jacobian(model[1].W, x_batch)
            dec_jac_batch = batched_jacobian(model[2].W, model[1].W(x_batch))
            
            # Get the encoded derivative: ż
            ż_ref = enc_ż(enc_jac_batch, ẋ_batch)

            # Compute gradients using Flux
            gradients = Flux.gradient((enc_paras, dec_paras, Ξ) -> sparse_loss(enc_paras, dec_paras, Ξ, model[3].W, basis, smallinds, x_batch, ẋ_batch, ż_ref, dec_jac_batch, method, alphas), model[1].W, model[2].W, Ξ)

            # Update the parameters
            Flux.Optimise.update!(opt_state, (model[1].W, model[2].W, Ξ), gradients)

            update_model_coeffs!(model[3].W, smallinds, Ξ)

            # Accumulate the loss for the current batch
            epoch_loss += sparse_loss(model[1].W, model[2].W, Ξ, model[3].W, basis, smallinds, x_batch, ẋ_batch, ż_ref, dec_jac_batch, method, alphas)
        end
        # Compute the average loss for the epoch
        epoch_loss /= num_batches
        
        # Store the epoch loss
        push!(epoch_loss_array, epoch_loss)

        # Print loss after some iterations
        if epoch % 100 == 0
            println("Epoch $epoch: Loss: $epoch_loss")
            println("Coefficients: $(model[3].W))")
            println()
        end
    end
    return model, epoch_loss_array
end
