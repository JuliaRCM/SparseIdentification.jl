
using Optim
using Flux
using Distances


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
    method

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

    L_ż = 0.095*sum(((enc_jacob * ẋᵢₙ) .- (Θ * model[3].W)).^2)

    L_ẋ = 0.95* sum(((Θ * (model[3].W * dec_jacob)) .- ẋᵢₙ).^2)
    
    return L_r + L_ż + L_ẋ
end


function solve(data, model, basis, solver::NNSolver)
    # Flux gradient has problem working with the structure data
    x = data.x
    ẋ = data.ẋ 
    function loss(model)
        # Initialize the loss
        batchLoss = 0
        
        for i in 1:size(data.x)[2]
            batchLoss += loss_kernel(x[:,i], ẋ[i,:], enc_jac[i], dec_jac[i], Θ[i], model)
        end
        # Mean of the coefficients averaged
        L_c = sum(abs.(model[3].W))/length(model[3].W)

        batch_loss_average = batchLoss / size(data.x)[2] + L_c
    
        return batch_loss_average
    end

    enc_jac = [(layer_derivative(model[1].W, Float32.(in)))[1] for in in eachcol(data.x)]
    dec_jac = [(layer_derivative(model[2].W, Float32.(in)))[1] for in in eachcol(data.x)]
    Θ = [basis(model[1].W(xᵢₙ)) for xᵢₙ in eachcol(data.x)]

    # Set up the optimizer's state
    opt_state = Flux.setup(Adam(), model)

    # Array to store the losses
    epoch_loss_array = Vector{Float64}()

    for epoch in 1:500
        # Derivatives of the encoder and decoder
        enc_jac = [(layer_derivative(model[1].W, Float32.(in)))[1] for in in eachcol(data.x)]
        dec_jac = [(layer_derivative(model[2].W, Float32.(in)))[1] for in in eachcol(data.x)]

        # Values of basis functions on all samples of the encoded training data states
        Θ = [basis(model[1].W(xᵢₙ)) for xᵢₙ in eachcol(data.x)]

        # Compute gradients using Flux
        gradients = Flux.gradient(loss, model)[1]

        # Update the parameters
        Flux.Optimise.update!(opt_state, model, gradients)

        # Store the epoch loss
        epoch_loss = loss(model)
        push!(epoch_loss_array, epoch_loss)

        # Print loss after some iterations
        if epoch % 50 == 0
            println("Epoch $epoch: Loss: $epoch_loss")
            println("Epoch $epoch: Coefficients: $(model[3].W)")
            println()
        end
    end
    return model
end


function sparse_solve(basis, data, model, smallinds)
    # Flux gradient has problem working with the structure data
    x = data.x
    ẋ = data.ẋ 

    # Derivatives dz/dx and dx/dz for all samples (must be outside of loss function because of the way Flux.gradient works)
    enc_jac = [(layer_derivative(model[1].W, Float32.(xᵢₙ)))[1] for xᵢₙ in eachcol(x)]
    dec_jac = [(layer_derivative(model[2].W, Float32.(xᵢₙ)))[1] for xᵢₙ in eachcol(x)]
        
    # Total loss
    function sparse_loss(model)
        loss_sum = 0

        # Loop over samples
        for i in 1:size(x,2)
            # Loop over each state
            for ind in axes(data.x,1)
                # non-zero coefficients of the ind state
                biginds = .~(smallinds[:,ind])

                #Reconstruction loss from encoded-decoded xᵢₙ
                L_r = sqeuclidean(model[2].W(model[1].W(x[:, i]))[ind], x[ind, i])
                
                # Values of basis functions on the current sample of the encoded training data states at biginds coefficients
                Θ = (basis(model[1].W(x[:,i])))[:,biginds]
                
                # model[3].W[biginds,1]: gives the values of the coefficients to be 
                # multiplied with the Θ basis functions to give the gradients from the SINDy method 
                L_ż = 0.095*sum(((enc_jac[i] * ẋ[:,i])[ind,:] .- (Θ * model[3].W[biginds,ind])).^2)
                
                # Note: dz/dt = Θ * model[3].W[biginds,:] This gives the encoded biginds gradients with their biginds coefficients
                # Note: dx/dz = dec_jac[i] This gives the jacobian of the decoded x with respect to the encoded z
                L_ẋ = 0.95 * sum((((Θ * model[3].W[biginds,:]) * dec_jac[i][ind,:]) .- ẋ[ind,i]).^2)

                loss_sum += L_r + L_ż + L_ẋ
            end
        end

        # Mean of the coefficients averaged
        L_c = sum(abs.(model[3].W[.~smallinds,:]))/length(model[3].W[.~smallinds,:])
        return loss_sum/size(x,2) + L_c
    end

    # Array to store the losses
    epoch_loss_array = Vector{Float64}()

    # Set up the optimizer's state
    opt_state = Flux.setup(Adam(), model)
    for epoch in 1:500
        # Compute gradients
        gradients = Flux.gradient(sparse_loss, model)[1]

        # Update the parameters
        Flux.Optimise.update!(opt_state, model, gradients)

        # Derivatives dz/dx and dx/dz for all samples (must be outside of loss function because of the way Flux.gradient works)
        enc_jac = [(layer_derivative(model[1].W, Float32.(xᵢₙ)))[1] for xᵢₙ in eachcol(x)]
        dec_jac = [(layer_derivative(model[2].W, Float32.(xᵢₙ)))[1] for xᵢₙ in eachcol(x)]

        # Store the epoch loss
        epoch_loss = sparse_loss(model)
        push!(epoch_loss_array, epoch_loss)

        # Print loss after some iterations
        if epoch % 100 == 0
            println("Epoch $epoch: Loss: $epoch_loss")
            println("Coefficients: $(model[3].W))")
            println()
        end
    end
    return model
end