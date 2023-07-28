
using Optim
using Flux
using ForwardDiff
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

encoder(x, model) =  (model[1].W  * x .+ model[1].b) 
decoder(x, model) = (model[2].W * x .+ model[2].b)

function loss_kernel(xᵢₙ, ẋᵢₙ, basis, model; biginds = nothing)
    #Reconstruction loss from encoded-decoded xᵢₙ
    L_r = sqeuclidean(decoder(encoder(xᵢₙ, model), model), xᵢₙ)

    Θ = basis(encoder(xᵢₙ, model))
    if (biginds != nothing)
        Θ = Θ[:,biginds]
    end
    L_ż = 0.095 * sqeuclidean(model[1].W * ẋᵢₙ, Θ * model[3].W )
    
    L_ẋ = 0.95* sqeuclidean(Θ * (model[3].W * model[2].W), ẋᵢₙ)

    return L_r + L_ż + L_ẋ
end

function flatten_model(model)
    θ = [model[1].W, model[1].b, model[2].W, model[2].b, model[3].W]
    # Flatten the model into a single vector
    return flattened_model = cat([vec(θ[i]) for i in 1:length(θ)]..., dims=1)
end

function reconstruct_model(flattened_model,ld,ndim,coeff_length)
    reconstructed_model = (
        (W = reshape(flattened_model[1:ld*ndim], ld, ndim), b = reshape(flattened_model[(ld*ndim)+1:(ld*ndim)+ndim], ndim)),
        (W = reshape(flattened_model[(ld*ndim)+ndim+1:(ld*ndim)+ndim+ld*ndim], ndim, ld), b = reshape(flattened_model[(ld*ndim)+ndim+ld*ndim+1:(ld*ndim)+ndim+ld*ndim+ld], ld)),
        (W = reshape(flattened_model[(ld*ndim)+ndim+ld*ndim+ld+1:end], coeff_length, ld), ),
    )
    return reconstructed_model
end

function optimize(loss, model, solver::NNSolver)
    # Reshape model for the gradient and optimizer calculations
    flattened_model = flatten_model(model)

    # Array to store the losses
    epoch_loss_array = Vector{Float64}()

    # Number of coefficients in each state
    coeff_length = size(model[3].W)[1]

    for epoch in 1:500
        # Compute gradients using ForwardDiff.jl
        gradients = ForwardDiff.gradient(loss, flattened_model)

        # Update the parameters using the optimizer
        Flux.Optimise.update!(Adam(), flattened_model, gradients)

        # Store the epoch loss
        epoch_loss = loss(flattened_model)
        push!(epoch_loss_array, epoch_loss)

        # Print loss after some iterations
        if epoch % 20 == 0
            println("Epoch $epoch: Loss: $epoch_loss")
            temp = reconstruct_model(flattened_model, length(model[1].b), length(model[2].b), coeff_length)
            println("Epoch $epoch: Coefficents: $(temp[3].W)")
            println()
        end
    end

    # Reconstruct the model
    return reconstruct_model(flattened_model, length(model[1].b), length(model[2].b), coeff_length)
end

function solve(basis, data, model, solver::NNSolver)
    # Mean of the coefficients summed
    L_c = sum(abs.(model[3].W))/length(model[3].W)

    # Get hidden layer sizes
    ndim = length(model[1].b)
    ld = length(model[2].b)

    # Number of coefficients in each state
    coeff_length = size(model[3].W)[1]

    # Total loss
    function loss(flattened_model) 
        local model = reconstruct_model(flattened_model, ndim, ld, coeff_length)
        
        # loss_kernel takes two arguments (column of data.x and row of data.ẋ)
        # First, define a function to extract columns from a matrix
        colwise(matrix) = (matrix[:, i] for i in 1:size(matrix, 2))
        
        return loss = 1 / size(data.x)[2] * mapreduce(z -> loss_kernel(z..., basis, model), +, Iterators.product(colwise(data.x), eachrow(data.ẋ))) + 0.0095 * L_c
    end
    return optimize(loss, model, solver)
end



function sparse_solve(basis, data, model, smallinds)

    # Get hidden layer sizes
    ndim = length(model[1].b)
    ld = length(model[2].b)

    # Initialize coeffs collector for end of loop
    local Ξ = zeros(size(model[3].W))

    # Initialize flattened_model outside the loop
    flattened_model = []
    for ind in axes(data.ẋ,1)
        biginds = .~(smallinds[:,ind])

        coeffs = model[3].W[.~(smallinds[:,ind]),ind]

        # Reshape model for the gradient and optimizer calculations
        θ = [model[1].W, model[1].b, model[2].W, model[2].b, coeffs,]
        # Flatten the model into a single vector
        flattened_model = cat([vec(θ[i]) for i in 1:length(θ)]..., dims=1)
    
        # Mean of the coefficients summed
        L_c = sum(abs.(coeffs))/length(coeffs)

        # Total loss
        function sparse_loss(flattened_model::AbstractVector)
            local model = (
                (W = reshape(flattened_model[1:ld*ndim], ld, ndim), b = reshape(flattened_model[(ld*ndim)+1:(ld*ndim)+ndim], ndim)),
                (W = reshape(flattened_model[(ld*ndim)+ndim+1:(ld*ndim)+ndim+ld*ndim], ndim, ld), b = reshape(flattened_model[(ld*ndim)+ndim+ld*ndim+1:(ld*ndim)+ndim+ld*ndim+ld], ld)),
                (W = coeffs, ),
            )

            loss_sum = 0
            for i in 1:size(data.x,2)
                #Reconstruction loss from encoded-decoded xᵢₙ
                L_r = sqeuclidean(decoder(encoder(data.x[:,i], model), model), data.x[:,i])

                Θ = basis(encoder(data.x[:,i], model))
                if (biginds != nothing)
                    Θ = Θ[:,biginds]
                end
                
                test1 = (Θ * model[3].W)
                L_ż = 0.095 * sqeuclidean((model[1].W * data.ẋ[:,i])[ind,:], test1)
                
                use = reshape(model[2].W[ind, :], 1, 2)
                L_ẋ = 0.95* sqeuclidean((Θ * (model[3].W * use))[:,ind], (data.ẋ[:,i])[ind,:])

                loss_sum += L_r + L_ż + L_ẋ
            end

            return loss_sum
        end

        # Array to store the losses
        epoch_loss_array = Vector{Float64}()

        for epoch in 1:1000
            # Compute gradients using ForwardDiff.jl
            gradients = ForwardDiff.gradient(sparse_loss, flattened_model)

            # Update the parameters using the optimizer
            Flux.Optimise.update!(Adam(), flattened_model, gradients)
            Ξ[biginds,ind] = flattened_model[(ld*ndim)+ndim+ld*ndim+ld+1:end]
            
            # Store the epoch loss
            epoch_loss = sparse_loss(flattened_model)
            push!(epoch_loss_array, epoch_loss)

            # Print loss after some iterations
            if epoch % 100 == 0
                println("Epoch $epoch: Loss: $epoch_loss")
                println("ind: $ind, Coefficients: $(flattened_model[(ld*ndim)+ndim+ld*ndim+ld+1:end]))")
                println()
            end
        end
    end

    reconstructed_model = (
            (W = reshape(flattened_model[1:ld*ndim], ld, ndim), b = reshape(flattened_model[(ld*ndim)+1:(ld*ndim)+ndim], ndim)),
            (W = reshape(flattened_model[(ld*ndim)+ndim+1:(ld*ndim)+ndim+ld*ndim], ndim, ld), b = reshape(flattened_model[(ld*ndim)+ndim+ld*ndim+1:(ld*ndim)+ndim+ld*ndim+ld], ld)),
            (W = Ξ, ),
        )
    return reconstructed_model
end