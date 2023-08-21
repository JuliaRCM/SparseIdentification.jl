
using Optim
using Flux
using Distances
using Random
using Zygote


abstract type AbstractSolver end

abstract type NonlinearSolver <: AbstractSolver end


struct JuliaLeastSquare <: AbstractSolver end

function solve(Θ, ẋ, ::JuliaLeastSquare)
    Θ \ ẋ
end


struct OptimSolver <: NonlinearSolver 
    method

    function OptimSolver(method = BFGS())
        new(method)
    end
end

function optimize(loss, x₀, solver::OptimSolver)
    result = Optim.optimize(loss, x₀, solver.method, Optim.Options(iterations=483)) #TODO: do i still need to specify iterations
    println(result)
    return result.minimizer
end

   
function solve(Θ, ẋ, solver::NonlinearSolver)
    x₀ = zeros(size(Θ,2), size(ẋ,2))
    loss(x) = mapreduce( y -> y^2, +, ẋ .- Θ * x )
    optimize(loss, x₀, solver)
end



struct NNSolver <: NonlinearSolver 
    optimizer

    function NNSolver(optimizer = Adam())
        new(optimizer)
    end
end


# Needed because Flux.gradient can't handle Flux.jacobian
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

# Evaluate basis functions for all xᵢₙ in batch
function evaluate_basis_batch(basis, data_batch)
    bases(b::CompoundBasis) = b.bases
    basis_output = [b(data_batch) for b in bases(basis)]
    return hcat(basis_output...)
end

function update_model_coeffs!(model_W, smallinds, Ξ)
    for ind in 1:size(model_W, 2)
        non_zero_indices = findall(.~smallinds[:, ind])
        @views model_W[:, ind][non_zero_indices] .= Ξ[ind]
    end
end

function dzdt(smallinds, Ξ, model_enc, x, basis)
    num_states = size(x, 1)
    num_samples = size(x, 2)
    dz_dt = Zygote.Buffer(x)
    
    for i in 1:num_samples
        for ind in 1:num_states
            biginds = .~(smallinds[:, ind])
            Θ = (basis(model_enc(x[:, i])))[:, biginds]
            dz_dt[ind, i] = (Θ * Ξ[ind])[1]
        end
    end
    
    return copy(dz_dt)
end

# Get ż from encoder derivative and ẋ
function enc_ż(enc_jac_batch, ẋ_batch)
    enc_mult_ż = Zygote.Buffer(ẋ_batch)
    for i in 1:size(enc_jac_batch, 2)
        enc_mult_ż[:, i] = (enc_jac_batch[:,i,:] * (ẋ_batch[:,i]))
    end
    return copy(enc_mult_ż)
end

# Get ẋ from decoder derivative and ż
function dec_ẋ(dec_jac_batch, ż)
    dec_mult_ẋ = Zygote.Buffer(ż')
    for i in 1:size(dec_jac_batch, 2)
        dec_mult_ẋ[:, i] = dec_jac_batch[:,i,:] * ż[i,:]
    end
    return copy(dec_mult_ẋ)
end


function solve(data, method, model, basis, solver::NNSolver)
    
    total_samples = size(data.x)[2]
    num_batches = ceil(Int, total_samples / method.batch_size)

    # Flux gradient has problem working with the structure data directly
    x = data.x
    ẋ = data.ẋ

    # Coefficients for the loss_kernel terms
    alphas = round(sum(abs2, x) / sum(abs2, ẋ), sigdigits = 3)

    function loss(model, x_batch, ẋ_batch, enc_jac_batch, dec_jac_batch, Θ_batch, method, alphas)
        # Compute the reconstruction loss for the entire batch
        L_r = sum((model[2].W(model[1].W(x_batch)) .- x_batch) .^ 2)

        # Compute the loss terms involving enc_jac_batch and Θ_batch
        enc_mult_ż = enc_ż(enc_jac_batch, ẋ_batch)

        # Loss from difference between encoded variables through SINDy and reference
        L_ż = alphas / 10 * sum((enc_mult_ż .- (Θ_batch * model[3].W)') .^ 2)

        # Compute the loss terms involving dec_jac_batch and Θ_batch
        ż = Θ_batch * model[3].W
        dec_mult_ẋ = dec_ẋ(dec_jac_batch, ż)

        # Loss from difference between decoded variables through SINDy and reference
        L_ẋ = alphas * sum((dec_mult_ẋ  .- ẋ_batch).^2)

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

    # Initialize inplace storage
    x_batch = zeros(size(x,1), method.batch_size)
    ẋ_batch = zeros(size(x,1), method.batch_size)
    enc_jac_batch  = zeros(size(x,1), method.batch_size, size(x,1))
    dec_jac_batch  = zeros(size(x,1), method.batch_size, size(x,1))
    Θ_batch = zeros(method.batch_size, size(basis(x[:,1]),2))

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
            x_batch .= x[:, batch_indices]
            ẋ_batch .= ẋ[:, batch_indices]

            # Derivatives of the encoder and decoder
            enc_jac_batch .= batched_jacobian(model[1].W, x_batch)
            dec_jac_batch .= batched_jacobian(model[2].W, model[1].W(x_batch))
        
            # Values of basis functions on all samples of the encoded training data states
            Θ_batch .= evaluate_basis_batch(basis, model[1].W(x_batch))

            # Compute gradients using Flux
            gradients = Flux.gradient(model -> loss(model, x_batch, ẋ_batch, enc_jac_batch, dec_jac_batch, Θ_batch, method, alphas), model)[1]

            # Update the parameters
            Flux.Optimise.update!(opt_state, model, gradients)

            # Accumulate the loss for the current batch
            epoch_loss += loss(model, x_batch, ẋ_batch, enc_jac_batch, dec_jac_batch, Θ_batch, method, alphas)
        end
        # Compute the average loss for the epoch
        epoch_loss /= num_batches

        # Store the epoch loss
        push!(epoch_loss_array, epoch_loss)

        # Print loss after some iterations
        if epoch % 50 == 0
            println("Epoch $epoch: Average Loss: $epoch_loss")
            println("Epoch $epoch: Coefficents: $(model[3].W)")
            println()
        end
    end
    return model, epoch_loss_array
end



function sparse_solve(data, method, model, basis, Ξ, smallinds, solver::NNSolver)
    
    total_samples = size(data.x)[2]
    num_batches = ceil(Int, total_samples / method.batch_size)

    # Flux gradient has problem working with the structure data
    x = data.x
    ẋ = data.ẋ

    # Coefficients for the loss_kernel terms
    alphas = round(sum(abs2, x) / sum(abs2, ẋ), sigdigits = 3)

    # Total loss
    function sparse_loss(enc_paras, dec_paras, Ξ, model, basis, smallinds, x, ẋ, enc_jac, dec_jac, method, alphas)
        batchLoss = 0

        # Get the encoded derivative: ż
        enc_mult_ż = enc_ż(enc_jac, ẋ)
        # Get the decoded derivative ẋ from ż and dx\dz
        SINDy_ẋ = (hcat([dec_jac[:, i, :] * enc_mult_ż[:, i] for i in axes(enc_mult_ż, 2)]...))

        # Values of basis functions on the current sample of the encoded training data states
        Θ = (basis(enc_paras(x)))

        # Loop over each state
        for ind in axes(x, 1)
            # non-zero coefficients of the ind state
            biginds = .~(smallinds[:, ind])

            # Reconstruction loss from encoded-decoded xᵢₙ
            L_r = sum((dec_paras(enc_paras(x)) .- x)[ind,:].^ 2)

            # Ξ[ind]: gives the values of the coefficients of a state to be 
            # multiplied with the Θ biginds basis functions at that state, 
            # to give the encoded gradients from the SINDy method 
            L_ż = alphas / 10 * sum(((enc_mult_ż)[ind, :] .- (Θ[:, biginds] * Ξ[ind])) .^ 2)
            
            # Decoded gradients from SINDy and reference
            L_ẋ = alphas * sum((SINDy_ẋ .- ẋ)[ind, :].^2)

            batchLoss += L_r + L_ż + L_ẋ
        end
    
        # Mean of the coefficients averaged
        L_c = sum(abs.(model[3].W)) / length(model[3].W)
        return batchLoss / size(x, 2) + method.coeff * L_c
    end
    
    # Array to store the losses
    epoch_loss_array = Vector{Float64}()

    # Set up the optimizer's state
    opt_state = Flux.setup(solver.optimizer, (model[1].W, model[2].W, Ξ))

    # Initialize inplace storage
    x_batch = zeros(size(x,1), method.batch_size)
    ẋ_batch = zeros(size(x,1), method.batch_size)
    # Derivatives of the encoder and decoder
    enc_jac_batch = zeros(size(x,1), method.batch_size, size(x,1))
    dec_jac_batch = zeros(size(x,1), method.batch_size, size(x,1))

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
            x_batch .= x[:, batch_indices]
            ẋ_batch .= ẋ[:, batch_indices]

            # Derivatives of the encoder and decoder
            enc_jac_batch .= batched_jacobian(model[1].W, x_batch)
            dec_jac_batch .= batched_jacobian(model[2].W, model[1].W(x_batch))
        
            # Compute gradients using Flux
            gradients = Flux.gradient((enc_paras, dec_paras, Ξ) -> sparse_loss(enc_paras, dec_paras, Ξ, model, basis, smallinds, x_batch, ẋ_batch, enc_jac_batch, dec_jac_batch, method, alphas), model[1].W, model[2].W, Ξ)

            # Update the parameters
            Flux.Optimise.update!(opt_state, (model[1].W, model[2].W, Ξ), gradients)

            update_model_coeffs!(model[3].W, smallinds, Ξ)

            # Accumulate the loss for the current batch
            epoch_loss += sparse_loss(model[1].W, model[2].W, Ξ, model, basis, smallinds, x_batch, ẋ_batch, enc_jac_batch, dec_jac_batch, method, alphas)
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