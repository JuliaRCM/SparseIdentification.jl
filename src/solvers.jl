
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
function layer_derivative(model_layer, x)
    # Compute the Jacobian matrix
    return Flux.jacobian(model_layer, x)
end

function loss_kernel(xᵢₙ, ẋᵢₙ, enc_jacob, dec_jacob, Θ, model)
    #Reconstruction loss from encoded-decoded xᵢₙ
    L_r = sqeuclidean(model[2].W(model[1].W(xᵢₙ)), xᵢₙ)

    L_ż = 0.0245 * sum(((enc_jacob * ẋᵢₙ) .- (Θ * model[3].W)).^2)

    L_ẋ = 0.245 * sum(((Θ * (model[3].W * dec_jacob)) .- ẋᵢₙ).^2)
    
    return L_r + L_ż + L_ẋ
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


function solve(data, method, model, basis, solver::NNSolver, batch_size = floor(Int, 0.2*size(data.x, 2)))
    
    total_samples = size(data.x)[2]
    num_batches = ceil(Int, total_samples / batch_size)

    # Flux gradient has problem working with the structure data
    x = Float32.(data.x)
    ẋ = Float32.(data.ẋ)

    function loss(model, x_batch, ẋ_batch, enc_jac_batch, dec_jac_batch, Θ_batch, method)
        # Initialize the loss
        batchLoss = 0
        
        for i in 1:size(x_batch, 2)
            batchLoss += loss_kernel(x_batch[:, i], ẋ_batch[:, i], enc_jac_batch[i], dec_jac_batch[i], Θ_batch[i], model)
        end
        # Mean of the coefficients averaged
        L_c = sum(abs.(model[3].W))/length(model[3].W)

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
            batch_start = (batch - 1) * batch_size + 1
            batch_end = min(batch * batch_size, total_samples)
            batch_indices = shuffled_indices[batch_start:batch_end]

            # Extract the data for the current batch
            x_batch = x[:, batch_indices]
            ẋ_batch = ẋ[:, batch_indices]

            # Derivatives of the encoder and decoder
            enc_jac_batch = [(layer_derivative(model[1].W, Float32.(in)))[1] for in in eachcol(x_batch)]
            dec_jac_batch = [(layer_derivative(model[2].W, Float32.(in)))[1] for in in eachcol(x_batch)]
            
            # Values of basis functions on all samples of the encoded training data states
            Θ_batch = [basis(model[1].W(xᵢₙ)) for xᵢₙ in eachcol(x_batch)]

            # Compute gradients using Flux
            gradients = Flux.gradient(model -> loss(model, x_batch, ẋ_batch, enc_jac_batch, dec_jac_batch, Θ_batch, method), model)[1]

            # Update the parameters
            Flux.Optimise.update!(opt_state, model, gradients)

            # Accumulate the loss for the current batch
            epoch_loss += loss(model, x_batch, ẋ_batch, enc_jac_batch, dec_jac_batch, Θ_batch, method)
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



function sparse_solve(data, method, model, basis, Ξ, smallinds, solver::NNSolver, batch_size = floor(Int, 0.2*size(data.x, 2)))
    
    total_samples = size(data.x)[2]
    num_batches = ceil(Int, total_samples / batch_size)

    # Flux gradient has problem working with the structure data
    x = Float32.(data.x)
    ẋ = Float32.(data.ẋ)

    # Total loss
    function sparse_loss(enc_paras, dec_paras, Ξ, model, basis, smallinds, x, ẋ, enc_jac, dec_jac, method)
        batchLoss = 0

        dz_dt = dzdt(smallinds, Ξ, model[1].W, x, basis)

        # Loop over samples
        for i in 1:size(x, 2)
            # Loop over each state
            for ind in axes(x, 1)
                # non-zero coefficients of the ind state
                biginds = .~(smallinds[:, ind])
    
                # Reconstruction loss from encoded-decoded xᵢₙ
                L_r = sqeuclidean(dec_paras(enc_paras(x[:, i]))[ind], x[ind, i])
    
                # Values of basis functions on the current sample of the encoded training data states at biginds coefficients
                Θ = (basis(enc_paras(x[:, i])))[:, biginds]

                # Ξ[ind]: gives the values of the coefficients of a state to be 
                # multiplied with the Θ biginds basis functions at that state, 
                # to give the gradients from the SINDy method 
                L_ż = 0.0245 * sum(((enc_jac[i] * ẋ[:, i])[ind, :] .- (Θ * Ξ[ind])).^2)
                
                L_ẋ = 0.245 * sum(((dec_jac[i] * dz_dt[:,i])[ind, :] .- ẋ[ind, i]).^2)   

                batchLoss += L_r + L_ż + L_ẋ
            end
        end
    
        # Mean of the coefficients averaged
        L_c = sum(abs.(model[3].W)) / length(model[3].W)
        return batchLoss / size(x, 2) + method.coeff * L_c
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
            batch_start = (batch - 1) * batch_size + 1
            batch_end = min(batch * batch_size, total_samples)
            batch_indices = shuffled_indices[batch_start:batch_end]

            # Extract the data for the current batch
            x_batch = x[:, batch_indices]
            ẋ_batch = ẋ[:, batch_indices]

            # Derivatives of the encoder and decoder
            enc_jac_batch = [(layer_derivative(model[1].W, Float32.(in)))[1] for in in eachcol(x_batch)]
            dec_jac_batch = [(layer_derivative(model[2].W, Float32.(in)))[1] for in in eachcol(x_batch)]
            
            # Compute gradients using Flux
            gradients = Flux.gradient((enc_paras, dec_paras, Ξ) -> sparse_loss(enc_paras, dec_paras, Ξ, model, basis, smallinds, x_batch, ẋ_batch, enc_jac_batch, dec_jac_batch, method), model[1].W, model[2].W, Ξ)

            # Update the parameters
            Flux.Optimise.update!(opt_state, (model[1].W, model[2].W, Ξ), gradients)

            update_model_coeffs!(model[3].W, smallinds, Ξ)

            # Accumulate the loss for the current batch
            epoch_loss += sparse_loss(model[1].W, model[2].W, Ξ, model, basis, smallinds, x_batch, ẋ_batch, enc_jac_batch, dec_jac_batch, method)
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