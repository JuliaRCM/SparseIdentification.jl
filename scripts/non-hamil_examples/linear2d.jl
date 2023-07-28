
using DifferentialEquations
using Distributions
using ODE
using Plots
using Random
using SparseIdentification

gr()


# --------------------
# Setup
# --------------------

# generate basis
#  - search space up to fifth order polynomials
#  - no trigonometric functions
basis = CompoundBasis(polyorder = 5, trigonometric = 0)

# initial data
x₀ = [2., 0.]

# 2D system 
nd = length(x₀)

# vector field
const A = [-0.1  2.0
           -2.0 -0.1]

rhs(xᵢₙ,p,t) = A*xᵢₙ
# rhs(dx,t,xᵢₙ,p) = rhs(xᵢₙ,p,t)

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# x = Array(data)
# stored as matrix with dims [nd,ntime]

num_samp = 12

# samples in p and q space
samp_range = LinRange(-20, 20, num_samp)

# initialize vector of matrices to store ODE solve output
# s depend on size of nd (total dims), 4 in the case here so we use samp_range x samp_range x samp_range x samp_range
s = collect(Iterators.product(fill(samp_range, nd)...))

# compute vector field from x state values
x = [collect(s[i]) for i in eachindex(s)]

x = hcat(x...)

# compute vector field from x state values at each timestep
# stored as matrix with dims [nd,ntime]
ẋ = zero(x)
for i in axes(ẋ,2)
    ẋ[:,i] .= A*x[:,i]
end

# collect training data
tdata = TrainingData(x, ẋ)

# println("x = ", tdata.x)
# println("ẋ = ", tdata.ẋ)


# ----------------------------------------
# Identify SINDy Vector Field
# ----------------------------------------

# choose SINDy method
method = SINDy(lambda = 0.05, noise_level = 0.05)

# compute vector field
# vectorfield = VectorField(method, basis, tdata)
# vectorfield = VectorField(method, basis, data; solver = OptimSolver())
vectorfield, model = VectorField(method, basis, tdata, solver = NNSolver())

#println(vectorfield.coefficients)

#println("   maximum(vectorfield.coefficients) = ", maximum(vectorfield.coefficients))


# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System...")

tstep = 0.01
tspan = (0.0,25.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

# prob_reference = GeometricIntegrators.ODEProblem((dx, t, x, params) -> rhs(dx,t,x), tspan, tstep, x₀)
# data_reference = GeometricIntegrators.integrate(prob_reference, Gauss(1))

prob = ODEProblem(rhs, x₀, tspan)
data = ODE.solve(prob, abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)

prob_approx = ODEProblem(vectorfield, x₀, tspan)
xid = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange) 


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

encoder(x, model) =  (model[1].W  * x .+ model[1].b) 
decoder(x, model) = (model[2].W * x .+ model[2].b)

en_data = [encoder(x[1], model) for x in eachrow(data.u)]
en_data = hcat(en_data...)
en_xid = [encoder(x[1], model) for x in eachrow(xid.u)]
en_xid = hcat(en_xid...)

p1 = plot()
plot!(p1, data.t, en_data[1,:], label = "Data")
plot!(p1, xid.t, en_xid[1,:], label = "Identified")

p2 = plot()
plot!(p2, data.t, en_data[2,:], label = "Data")
plot!(p2, xid.t, en_xid[2,:], label = "Identified")

display(plot(p1, p2))
savefig("linear2d.png")

p3 = plot(data[1,:], en_data[2,:], label="true")
p3 = scatter!(xid[1,:], en_xid[2,:], label="approx", linestyle =:dash, mc=:red, ms=2, ma=0.5, xlabel ="X1", ylabel="X2")
display(plot(p3, show = true, reuse = false))
savefig("linear2d_fig2.png")








# using Distances
# using ForwardDiff
# using Flux

# function set_model(tdata, Ξ)
#     ld = size(tdata.x)[1]
#     ndim = size(tdata.x)[1]
#     model = ( 
#         (W = rand(ld, ndim), b = zeros(ndim)),
#         (W = rand(ndim, ld), b = zeros(ld)),
#         (W = Ξ, ),
#     )
#     return model
# end

# encoder(x, model) =  (model[1].W  * x .+ model[1].b) 
# decoder(x, model) = (model[2].W * x .+ model[2].b)

# # Total loss
# function loss(flattened_model) 
#     #TODO: change from 2,2 to make it more general
#     local model = reconstruct_model(flattened_model, 2, 2)
    
#     # loss_kernel takes two arguments (column of data.x and row of data.ẋ)
#     # First, define a function to extract columns from a matrix
#     colwise(matrix) = (matrix[:, i] for i in 1:size(matrix, 2))
    
#     return loss = 1 / size(tdata.x)[2] * mapreduce(z -> loss_kernel(z..., basis, model), +, Iterators.product(colwise(tdata.x), eachrow(tdata.ẋ'))) + 0.0095 * L_c
# end

# function flatten_model(model)
#     θ = [model[1].W, model[1].b, model[2].W, model[2].b, model[3].W]
#     # Flatten the model into a single vector
#     return flattened_model = cat([vec(θ[i]) for i in 1:length(θ)]..., dims=1)
# end
# function reconstruct_model(flattened_model,ld,ndim)
#     reconstructed_model = (
#         (W = reshape(flattened_model[1:ld*ndim], ld, ndim), b = reshape(flattened_model[(ld*ndim)+1:(ld*ndim)+ndim], ndim)),
#         (W = reshape(flattened_model[(ld*ndim)+ndim+1:(ld*ndim)+ndim+ld*ndim], ndim, ld), b = reshape(flattened_model[(ld*ndim)+ndim+ld*ndim+1:(ld*ndim)+ndim+ld*ndim+ld], ld)),
#         (W = reshape(flattened_model[(ld*ndim)+ndim+ld*ndim+ld+1:end], 21, 2), ),
#     )
#     return reconstructed_model
# end


# function loss_kernel(xᵢₙ, ẋᵢₙ, basis, model)
#     #Reconstruction loss from encoded-decoded xᵢₙ
#     L_r = sqeuclidean(decoder(encoder(xᵢₙ, model), model), xᵢₙ)

#     Θ = basis(encoder(xᵢₙ, model))
#     L_ż = 0.95 * sqeuclidean(model[1].W * ẋᵢₙ, Θ * model[3].W )
    
#     L_ẋ = 0.095* sqeuclidean(Θ * (model[3].W * model[2].W), ẋᵢₙ)

#     return L_r + L_ż + L_ẋ
# end


# Θ = basis(tdata.x)
# Ξ = rand(size(Θ,2), size(tdata.ẋ', 2))

# model = set_model(tdata, Ξ)
# L_c = mean(abs.(model[3].W))


# Θ = basis(encoder(x[:,1], model))

# L_c = sum(abs.(model[3].W))/length(model[3].W)


# flattened_model = flatten_model(model)

# loss(flattened_model)

# test = reconstruct_model(flattened_model, 2, 2)

# Θ[:,biginds]

# for ind in axes(ẋnoisy,1)
#     biginds = .~(smallinds[:,ind])
#     Ξ[biginds,ind] .= solve(Θ[:,biginds], ẋnoisy[ind,:], solver)
# end



#################################################################################
# Working test code
#################################################################################
# using ForwardDiff
# using Distances
# using Flux
# function sparse_solve_two(basis, data, model, smallinds)
#     ndim = length(model[1].b)
#     ld = length(model[2].b)
#     Ξ = zeros(size(model[3].W))

#     # Initialize flattened_model outside the loop
#     flattened_model = []
#     for ind in axes(data.ẋ,1)
#         biginds = .~(smallinds[:,ind])
#         #coeffs
#         coeffs =  model[3].W[.~(smallinds[:,ind]),ind]
#         # Reshape model for the gradient and optimizer calculations
#         # θ is partly a reference to coeffs[biginds] so coeffs[biginds] will be updated
#         θ = [model[1].W, model[1].b, model[2].W, model[2].b, coeffs,]
#         # Flatten the model into a single vector
#         flattened_model = cat([vec(θ[i]) for i in 1:length(θ)]..., dims=1)
    
#         # Mean of the coefficients summed
#         L_c = sum(abs.(coeffs))/length(coeffs)

#         # Total loss
#         function sparse_loss(flattened_model::AbstractVector)
#             #TODO: generalize these lines below
#             # c = zeros(21,2) #zeros(size(basis(data.x),2), size(data.ẋ, 2))
#             # ld = 2
#             # ndim = 2
#             #(ld*ndim)+ndim+ld*ndim+ld+1
#             # c[biginds] .= flattened_model[13:end]
#             # println()
#             # println("c in sparse_loss: $c")
#             # println()
#             local model = (
#                 (W = reshape(flattened_model[1:ld*ndim], ld, ndim), b = reshape(flattened_model[(ld*ndim)+1:(ld*ndim)+ndim], ndim)),
#                 (W = reshape(flattened_model[(ld*ndim)+ndim+1:(ld*ndim)+ndim+ld*ndim], ndim, ld), b = reshape(flattened_model[(ld*ndim)+ndim+ld*ndim+1:(ld*ndim)+ndim+ld*ndim+ld], ld)),
#                 (W = coeffs, ),
#             )

#             sum = 0
#             for i in 1:size(data.x,2)
#                 #Reconstruction loss from encoded-decoded xᵢₙ
#                 L_r = sqeuclidean(decoder(encoder(data.x[:,i], model), model), data.x[:,i])

#                 Θ = basis(encoder(data.x[:,i], model))
#                 if (biginds != nothing)
#                     Θ = Θ[:,biginds]
#                 end
                
#                 test1 = (Θ * model[3].W)#[:,ind]
#                 L_ż = 0.95 * sqeuclidean((model[1].W * data.ẋ[:,i])[ind,:], test1)
                
#                 use = reshape(model[2].W[ind, :], 1, 2)
#                 L_ẋ = 0.095* sqeuclidean((Θ * (model[3].W * use))[:,ind], (data.ẋ[:,i])[ind,:])

#                 sum += L_r + L_ż + L_ẋ
#             end

#             return sum
#             # colwise(matrix) = (matrix[:, i] for i in 1:size(matrix, 2))
#             # return loss = 1 / length(data.x)[2] * mapreduce(z -> loss_kernel(z..., basis, model; biginds), +, Iterators.product(colwise(data.x), data.ẋ[ind,:])) + 0.0095 * L_c
#             # return loss = 1 / size(data.x, 2) * mapreduce(z -> loss_kernel(z, data.ẋ[ind, :], basis, model; biginds), +, eachcol(data.x)) + 0.0095 * L_c
#         end

#         # Array to store the losses
#         epoch_loss_array = Vector{Float64}()

#         for epoch in 1:500
#             # Compute gradients using ForwardDiff.jl
#             gradients = ForwardDiff.gradient(sparse_loss, flattened_model)

#             # Update the parameters using the optimizer
#             Flux.Optimise.update!(Adam(), flattened_model, gradients)
#             Ξ[biginds,ind] = flattened_model[(ld*ndim)+ndim+ld*ndim+ld+1:end]
            
#             # Store the epoch loss
#             epoch_loss = sparse_loss(flattened_model)
#             push!(epoch_loss_array, epoch_loss)

#             # Print loss after some iterations
#             if epoch % 10 == 0
#                 # Ξ = zeros(size(model[3].W))
#                 # Ξ[biginds,ind] = flattened_model[(ld*ndim)+ndim+ld*ndim+ld+1:end]
#                 println()
#                 println("Epoch $epoch: Loss: $epoch_loss")
#                 println("flat world: $ind:, $(flattened_model[(ld*ndim)+ndim+ld*ndim+ld+1:end]))")
#                 # println("Coefficients: $Ξ")
#                 println()
#             end
#         end

#         # Ξ[biginds,ind] = flattened_model[(ld*ndim)+ndim+ld*ndim+ld+1:end]
#     end

#     reconstructed_model = (
#             (W = reshape(flattened_model[1:ld*ndim], ld, ndim), b = reshape(flattened_model[(ld*ndim)+1:(ld*ndim)+ndim], ndim)),
#             (W = reshape(flattened_model[(ld*ndim)+ndim+1:(ld*ndim)+ndim+ld*ndim], ndim, ld), b = reshape(flattened_model[(ld*ndim)+ndim+ld*ndim+1:(ld*ndim)+ndim+ld*ndim+ld], ld)),
#             (W = Ξ, ),
#         )
#     return reconstructed_model
# end


# # function loss_kernel(xᵢₙ, ẋᵢₙ, basis, model; biginds = nothing)
# #     #Reconstruction loss from encoded-decoded xᵢₙ
# #     L_r = sqeuclidean(decoder(encoder(xᵢₙ, model), model), xᵢₙ)

# #     Θ = basis(encoder(xᵢₙ, model))
# #     if (biginds != nothing)
# #         Θ = Θ[:,biginds]
# #     end

# #     L_ż = 0.95 * sqeuclidean(model[1].W * ẋᵢₙ, Θ * model[3].W )
    
# #     L_ẋ = 0.095* sqeuclidean(Θ * (model[3].W * model[2].W), ẋᵢₙ)

# #     return L_r + L_ż + L_ẋ
# # end

# encoder(x, model) =  (model[1].W  * x .+ model[1].b) 
# decoder(x, model) = (model[2].W * x .+ model[2].b)

# function flatten_model(model)
#     θ = [model[1].W, model[1].b, model[2].W, model[2].b, model[3].W]
#     # Flatten the model into a single vector
#     return flattened_model = cat([vec(θ[i]) for i in 1:length(θ)]..., dims=1)
# end

# function reconstruct_model(flattened_model,ld,ndim)
#     reconstructed_model = (
#         (W = reshape(flattened_model[1:ld*ndim], ld, ndim), b = reshape(flattened_model[(ld*ndim)+1:(ld*ndim)+ndim], ndim)),
#         (W = reshape(flattened_model[(ld*ndim)+ndim+1:(ld*ndim)+ndim+ld*ndim], ndim, ld), b = reshape(flattened_model[(ld*ndim)+ndim+ld*ndim+1:(ld*ndim)+ndim+ld*ndim+ld], ld)),
#         #TODO: generalize line below
#         (W = reshape(flattened_model[(ld*ndim)+ndim+ld*ndim+ld+1:end], 21, 2), ),
#     )
#     return reconstructed_model
# end



# transpose = Matrix(ẋ')
# qdata = TrainingData(x, transpose)
# # Pool Data (evaluate library of candidate basis functions on training data)
# Θ = basis(qdata.x)

# # Ξ is the coefficients of the bases(Θ)
# Ξ = rand(size(Θ,2), size(tdata.ẋ', 2))
# Ξ[1:2,1] .= 0
# Ξ[5:10,2] .= 0
# function set_model(data, Ξ)
#     ld = size(data.x)[1]
#     ndim = size(data.x)[1]
#     model = ( 
#         (W = rand(ld, ndim), b = zeros(ndim)),
#         (W = rand(ndim, ld), b = zeros(ld)),
#         (W = Ξ, ),
#     )
#     return model
# end

# # initialize parameters
# model = set_model(tdata, Ξ)


# smallinds = abs.(model[3].W) .< method.λ
# biginds = .~smallinds
# test = sparse_solve_two(basis, tdata, model, smallinds)



# using Flux
# function set_model_two(data, Ξ)
#     encoder = Chain(
#     Dense(size(data.x)[1] => 128, sigmoid), 
#     Dense(128 => 64, sigmoid), 
#     Dense(64 => 32, sigmoid),
#     Dense(32 => size(data.x)[1])
#     )
#     decoder = Chain(
#     Dense(size(data.x)[1] => 32, sigmoid),  
#     Dense(32 => 64, sigmoid),
#     Dense(64 => 128),
#     Dense(128 => size(data.x)[1])
#     )

#     model = ( 
#         (W = encoder,),
#         (W = decoder,),
#         (W = Ξ, ),
#     )
#     return model
# end

# data = TrainingData(x, Matrix(ẋ'))

# # Pool Data (evaluate library of candidate basis functions on training data)
# Θ = basis(data.x)

# # Ξ is the coefficients of the bases(Θ)
# Ξ = zeros(size(Θ,2), size(data.ẋ, 2))

# test_model = set_model_two(data, Ξ)
# encoder(x, model) =  (model[1].W(x)) 
# decoder(x, model) = (model[2].W(x))

# test = encoder(tdata.x[:,1], test_model)

