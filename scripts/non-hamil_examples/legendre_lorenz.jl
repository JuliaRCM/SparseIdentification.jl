
using DifferentialEquations
using Distributions
using ODE
using Plots
using Random
using LegendrePolynomials
using SparseIdentification

gr()


# --------------------
# Setup
# --------------------

# search space up to third order polynomials
polyorder = 3

# no trigonometric functions
usesine = false

# Lorenz's parameters (chaotic)
sigma = 10.0
beta = 8/3
rho = 28.0

# generate basis
basis = CompoundBasis(polyorder = polyorder)

# initial data
x₀ = [-8.0, 8.0, 27.0]

# 3D system 
nd = length(x₀)


# noise level
eps = 0.05

# lambda parameter
lambda = 0.05

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

p = (sigma, beta, rho)



########################################################################################################################
########################################################################################################################
###################################### CODE TO GENERATE DATA ###########################################################
########################################################################################################################
########################################################################################################################

function library_size(n, poly_order, use_sine=false, include_constant=true)
    l = 0
    for k in 0:poly_order
        l += binomial(n+k-1, k)
    end
    if use_sine
        l += n
    end
    if !include_constant
        l -= 1
    end
    return l
end


function get_lorenz_data(n_ics; noise_strength=0)
    t = 0:0.02:5
    n_steps = length(t)
    input_dim = 128
    
    ic_means = [0, 0, 25]
    ic_widths = 2 .* [36, 48, 41]

    # training data
    # ics = ic_widths .* (rand(n_ics, 3) .- 0.5) .+ ic_means
    ics = [ic_widths .* (rand(3) .- 0.5) .+ ic_means for _ in 1:n_ics]
    ics = Matrix((hcat(ics...))')
    data = generate_lorenz_data(ics, t, input_dim, linear=false, normalization=[1/40, 1/40, 1/40])
    data["x"] = reshape(data["x"], (n_steps * n_ics, input_dim)) .+ noise_strength .* randn(n_steps * n_ics, input_dim)
    data["dx"] = reshape(data["dx"], (n_steps * n_ics, input_dim)) .+ noise_strength .* randn(n_steps * n_ics, input_dim)
    data["ddx"] = reshape(data["ddx"], (n_steps * n_ics, input_dim)) .+ noise_strength .* randn(n_steps * n_ics, input_dim)

    return data
end

function lorenz_coefficients(normalization; poly_order=3, sigma=10.0, beta=8/3, rho=28.0)
    Xi = zeros(Float64, library_size(3, poly_order), 3)
    Xi[2, 1] = -sigma
    Xi[3, 1] = sigma * normalization[1] / normalization[2]
    Xi[2, 2] = rho * normalization[2] / normalization[1]
    Xi[3, 2] = -1
    Xi[7, 2] = -normalization[2] / (normalization[1] * normalization[3])
    Xi[4, 3] = -beta
    Xi[6, 3] = normalization[3] / (normalization[1] * normalization[2])
    return Xi
end

function simulate_lorenz(z0, t; sigma=10.0, beta=8/3, rho=28.0)
    function f(du, u, p, t)
        du[1] = sigma * (u[2] - u[1])
        du[2] = u[1] * (rho - u[3]) - u[2]
        du[3] = u[1] * u[2] - beta * u[3]
    end

    function df(du, dz, z, t)
        du[1] = sigma*(dz[2] - dz[1])
        du[2] = dz[1]*(rho - z[3]) + z[1]*(-dz[3]) - dz[2]
        du[3] = dz[1]*z[2] + z[1]*dz[2] - beta*dz[3]
        return du
    end

    prob = ODEProblem(f, z0, (0.0, maximum(t)))
    sol = DifferentialEquations.solve(prob, saveat=t)
    z = hcat(sol.u...)'

    dz = zeros(size(z))
    ddz = zeros(size(z))
    for i in 1:length(t)
        du = zero(z[1,:])
        dz[i,:] = df(du, dz[i,:], z[i,:], t[i])
        
        ddz[i,:] = df(du, ddz[i,:], dz[i,:], t[i])
    end

    return z, dz, ddz
end

function generate_lorenz_data(ics, t, n_points; linear=true, normalization=nothing,
                              sigma=10.0, beta=8/3, rho=28.0)
    n_ics = size(ics, 1)
    n_steps = length(t)
    dt = t[2] - t[1]

    d = 3
    z = zeros(n_ics, n_steps, d)
    dz = zeros(n_ics, n_steps, d)
    ddz = zeros(n_ics, n_steps, d)
    for i in 1:n_ics
        z[i, :, :], dz[i, :, :], ddz[i, :, :] = simulate_lorenz(ics[i, :], t, sigma=sigma, beta=beta, rho=rho)
    end

    if normalization !== nothing
        z .= z .* reshape(normalization, 1, 1, 3)
        dz .= dz .* reshape(normalization, 1, 1, 3)
        ddz .= ddz .* reshape(normalization, 1, 1, 3)
    end

    n = n_points
    L = 1
    y_spatial = LinRange(-L, L, n)

    modes = zeros(2 * d, n)
    for i in 1:2 * d
        modes[i, :] = Pl.(y_spatial, i)
    end

    x1 = zeros(n_ics, n_steps, n)
    x2 = zeros(n_ics, n_steps, n)
    x3 = zeros(n_ics, n_steps, n)
    x4 = zeros(n_ics, n_steps, n)
    x5 = zeros(n_ics, n_steps, n)
    x6 = zeros(n_ics, n_steps, n)

    
    x = zeros(n_ics, n_steps, n)
    dx = zeros(n_ics, n_steps, n)
    ddx = zeros(n_ics, n_steps, n)
    for i in 1:n_ics
        for j in 1:n_steps
            x1 = modes[1, :] .* z[i, j, 1]
            x2 = modes[2, :] .* z[i, j, 2]
            x3 = modes[3, :] .* z[i, j, 3]
            x4 = modes[4, :] .* z[i, j, 1]^3
            x5 = modes[5, :] .* z[i, j, 2]^3
            x6 = modes[6, :] .* z[i, j, 3]^3

            x[i, j, :] .= x1 + x2 + x3
            if !linear
                x[i, j, :] .+= x4 + x5 + x6
            end

            dx1 = modes[1, :] .* dz[i, j, 1]
            dx2 = modes[2, :] .* dz[i, j, 2]
            dx3 = modes[3, :] .* dz[i, j, 3]
            dx4 = modes[4, :] .* 3 * z[i, j, 1]^2 * dz[i, j, 1]
            dx5 = modes[5, :] .* 3 * z[i, j, 2]^2 * dz[i, j, 2]
            dx6 = modes[6, :] .* 3 * z[i, j, 3]^2 * dz[i, j, 3]

            dx[i, j, :] .= dx1 + dx2 + dx3
            if !linear
                dx[i, j, :] .+= dx4 + dx5 + dx6
            end

            ddx1 = modes[1, :] .* ddz[i, j, 1]
            ddx2 = modes[2, :] .* ddz[i, j, 2]
            ddx3 = modes[3, :] .* ddz[i, j, 3]
            ddx4 = modes[4, :] .* (6 * z[i, j, 1] * dz[i, j, 1]^2 + 3 * z[i, j, 1]^2 * ddz[i, j, 1])
            ddx5 = modes[5, :] .* (6 * z[i, j, 2] * dz[i, j, 2]^2 + 3 * z[i, j, 2]^2 * ddz[i, j, 2])
            ddx6 = modes[6, :] .* (6 * z[i, j, 3] * dz[i, j, 3]^2 + 3 * z[i, j, 3]^2 * ddz[i, j, 3])

            ddx[i, j, :] .= ddx1 + ddx2 + ddx3
            if !linear
                ddx[i, j, :] .+= ddx4 + ddx5 + ddx6
            end
        end
    end


    if normalization === nothing
        sindy_coefficients = lorenz_coefficients([1.0, 1.0, 1.0], sigma=sigma, beta=beta, rho=rho)
    else
        sindy_coefficients = lorenz_coefficients(normalization, sigma=sigma, beta=beta, rho=rho)
    end

    data = Dict(
        "t" => t,
        "y_spatial" => y_spatial,
        "modes" => modes,
        "x" => x,
        "dx" => dx,
        "ddx" => ddx,
        "z" => z,
        "dz" => dz,
        "ddz" => ddz,
        "sindy_coefficients" => Float32.(sindy_coefficients)
    )

    return data
end
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# Get data, arguments: initial conditions=20, noise_strength=1e-6
training_data = get_lorenz_data(20, noise_strength=1e-6)

# Store data
tdata = TrainingData(Float32.(training_data["x"]'), Float32.(training_data["dx"]')) 

# choose SINDy method
method = SINDy(lambda = 0.05, noise_level = 0.0, coeff = 0.52, batch_size = 80)


println("Computing Vector Field...")

# compute vector field using least squares regression (/) solver
# vectorfield = VectorField(method, basis, tdata)
#Using BFGS() solver
# vectorfield = VectorField(method, basis, tdata, solver = OptimSolver())

#Using Neural Network solver
vectorfield, model = VectorField(method, basis, tdata, solver = NNSolver())


# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System...")

# With T_end=20.0
# FIGURE 1: LORENZ for T in[0,20]
# True model:
tstep = 0.001
tspan = (0.001, 20.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

p = (sigma, beta, rho)
prob = ODEProblem(lorenz, x₀, tspan, p)
# stored as dims [states x iters] matrix 
xA = ODE.solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12) 

# Approximate model:
prob_approx = ODEProblem(vectorfield, x₀, tspan)
xB = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

p1 = plot(xA, vars=(1,2,3), xlabel="x", ylabel="y", zlabel="z", label="true")
p2 = plot(xB, vars=(1,2,3), xlabel="x", ylabel="y", zlabel="z", label="approx")
display(plot(p1, p2, layout=(1,2), show = true, reuse = false, size=(1000,1000)))

savefig("Fig1_ex2_lorenz.png")

println("Plotting Figure 2...")

# Figure 2:
p3 = plot(xA,vars=(0,1), linecolor = :black, linewidth = 1.5, label="true_state1")
plot!(p3, xB,vars=(0,1), linecolor = :red, linestyle = :dash, linewidth = 1.5, label="approx_state1")
p4 = plot(xA,vars=(0,2), linecolor = :black, linewidth = 1.5, label="true_state2")
plot!(p4, xB,vars=(0,2), linecolor = :red, linestyle = :dash, linewidth = 1.5, label="approx_state2")
display(plot(p3, p4, layout=(1,2), show = true, reuse = false, size=(1000,1000), xlabel="Time", ylabel="X"))

savefig("Fig2_ex2_lorenz.png")


println("Integrating and Plotting Figure 3...")

# With T_end=250.0
# True model:
tstep = 0.001
tspan = (0.001, 250.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

p = (sigma, beta, rho)
prob = ODEProblem(lorenz, x₀, tspan, p)
xA = ODE.solve(prob, abstol=1e-6, reltol=1e-6) #stored as dims [3 x iters] matrix 

# Approximate model:
prob_approx = ODEProblem(vectorfield, x₀, tspan)
xB = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)

# Figure 3:
dtA = [0; diff(xA.t)]
dtB = [0; diff(xB.t)]
p5 = plot(xA[1,:], xA[2,:], xA[3,:], zcolor=dtA, xlabel="x", ylabel="y", zlabel="z", label="true")
p6 = plot(xB[1,:], xB[2,:], xB[3,:], zcolor=dtB, xlabel="x", ylabel="y", zlabel="z", label="approx")
display(plot(p5, p6, layout=(1,2), show = true, reuse = false, size=(1000,1000)))

savefig("Fig3_ex2_lorenz.png")


#*******************************************************************#
#*******************************************************************#
#*******************************************************************#
# New plotting code copied from linear2d.jl
println("Plotting Extra...")

p1 = plot()
plot!(p1, xA.t, data[1,:], label = "Data")
plot!(p1, xid.t, xid[1,:], label = "Identified")

p2 = plot()
plot!(p2, xA.t, data[2,:], label = "Data")
plot!(p2, xid.t, xid[2,:], label = "Identified")

plot(p1, p2)

p3 = plot(xA[1,:], xA[2,:], label="true", linestyle =:dash)
p3 = scatter!(xB[1,:], xB[2,:], label="approx", linestyle =:dash, mc=:red, ms=2, ma=0.5, xlabel ="X1", ylabel="X2")
display(plot(p3, show = true, reuse = false))
savefig("linear2d_fig2.png")








########################################################################################################################
########################################################################################################################
#-----------------------------------------------------Flux Setup ------------------------------------------------------#
########################################################################################################################
########################################################################################################################

using Flux
using Base.Threads

# Initialize a model with random parameters and Ξ = 0
function set_model(data, basis)
    encoder = Chain(
    Dense(size(data.x)[1] => 64, sigmoid), 
    Dense(64 => 32), 
    )

    decoder = Chain(
    Dense(32 => 64, sigmoid),  
    Dense(64 => size(data.x)[1])
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


# initialize parameters
model = set_model(tdata, basis)


###################################################################################################
###################################################################################################
###################################################################################################
# -------------------------------------Zygote Code-------------------------------------------------
###################################################################################################
###################################################################################################
###################################################################################################
using Zygote

# Needed because Flux.gradient can't handle Flux.jacobian
function batched_jacobian(model_layer, x_batch)
    # output size using first sample
    output_dim = size(model_layer(x_batch[:, 1]))[1]
    # batch_size using number of samples
    batch_size = size(x_batch, 2)

    # The jacobian at each of the samples
    batch_jac = zeros(output_dim, batch_size, size(x_batch, 1))
    
    for i in 1:batch_size
        x_input = x_batch[:, i]
        jac = Flux.jacobian(model_layer, x_input)[1]
        batch_jac[:, i, :] = jac
    end
    
    return batch_jac
end

function update_model_coeffs!(model_W, smallinds, Ξ)
    for ind in 1:size(model_W, 2)
        non_zero_indices = findall(.~smallinds[:, ind])
        @views model_W[:, ind][non_zero_indices] .= Ξ[ind]
    end
end

# Get ż from dz/dx and ẋ
function enc_ż(enc_jac_batch, ẋ_batch)
    # Size is equal to encoded features and number of batches
    ż_ref = zero(enc_jac_batch[:,:, 1])
    for i in 1:size(enc_jac_batch, 2)
        ż_ref[:, i] = (enc_jac_batch[:,i,:] * (ẋ_batch[:,i]))
    end
    return ż_ref
end

# Get ẋ from decoder derivative (dx/dz) and ż
function dec_ẋ(dec_jac_batch, ż)
    # Size is equal to decoded features and number of batches
    dec_mult_ẋ = Zygote.Buffer(dec_jac_batch[:,:, 1])
    for i in 1:size(dec_jac_batch, 2)
        dec_mult_ẋ[:, i] = dec_jac_batch[:,i,:] * ż[:,i]
    end
    return copy(dec_mult_ẋ)
end

# Get ż from SINDy coefficients and basis functions
function set_ż_SINDY(enc_x_batch, Θ, Ξ, smallinds)
    ż_SINDy = Zygote.Buffer(zeros(size(enc_x_batch, 1), size(Θ,1)))
    for ind in axes(enc_x_batch, 1)
        # non-zero coefficients of the ind state
        biginds = .~(smallinds[:, ind])
        ż_SINDy[ind,:] = Θ[:, biginds] * Ξ[ind]
    end
    return copy(ż_SINDy)
end


total_samples = size(tdata.x)[2]
num_batches = ceil(Int, total_samples / method.batch_size)

# Flux gradient has problem working with the structure data directly
x = Float32.(tdata.x)
ẋ = Float32.(tdata.ẋ)

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
opt_state = Flux.setup(Adam(), model)

@time for epoch in 1:300
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
    if epoch % 5 == 0
        println("Epoch $epoch: Average Loss: $epoch_loss")
        # println("Epoch $epoch: Coefficents: $(model[3].W)")
        println()
    end
end

# Plot the initial loss array
display(plot(log.(epoch_loss_array), label = "Initial Optimization Loss"))



# Initialize smallinds before the loop
smallinds = falses(size(model[3].W))

# find coefficients below λ threshold
smallinds .= abs.(model[3].W) .< method.lambda

# set all small coefficients to zero
model[3].W[smallinds] .= 0

Ξ = separate_coeffs(model[3].W, smallinds)

total_samples = size(tdata.x)[2]
num_batches = ceil(Int, total_samples / method.batch_size)

# Flux gradient has problem working with the structure data
x = Float32.(tdata.x)
ẋ = Float32.(tdata.ẋ)

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
opt_state = Flux.setup(Adam(), (model[1].W, model[2].W, Ξ))

for epoch in 1:5
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
    if epoch % 1 == 0
        println("Epoch $epoch: Loss: $epoch_loss")
        # println("Coefficients: $(model[3].W))")
        println()
    end
end

# Convert vector of vectors to a single vector
SINDy_loss_array = vcat(SINDy_loss_array...)

# Plot the SINDy loss array
display(plot(log.(SINDy_loss_array), label = "SINDy Optimization Loss"))







###################################################################################################
###################################################################################################
###################################################################################################
# -------------------------------------Enzyme Code-------------------------------------------------
###################################################################################################
###################################################################################################
###################################################################################################
using Enzyme

# Needed because Flux.gradient can't handle Flux.jacobian
function batched_jacobian(model_layer, x_batch)
    # output size using first sample
    output_dim = size(model_layer(x_batch[:, 1]))[1]

    # batch_size using number of samples
    batch_size = size(x_batch, 2)
    
    # The jacobian at each of the samples
    batch_jac = zeros(output_dim, batch_size, size(x_batch, 1))
    
    for i in 1:batch_size
        x_input = x_batch[:, i]
        jac = Flux.jacobian(model_layer, x_input)[1]
        batch_jac[:, i, :] = jac
    end
    
    return batch_jac
end

function update_model_coeffs!(model_W, smallinds, Ξ)
    for ind in 1:size(model_W, 2)
        non_zero_indices = findall(.~smallinds[:, ind])
        @views model_W[:, ind][non_zero_indices] .= Ξ[ind]
    end
end

# Get ż from dz/dx and ẋ
function enc_ż(enc_jac_batch, ẋ_batch)
    # Size is equal to encoded features and number of batches
    ż_ref = zero(enc_jac_batch[:,:, 1])
    for i in 1:size(enc_jac_batch, 2)
        ż_ref[:, i] = (enc_jac_batch[:,i,:] * (ẋ_batch[:,i]))
    end
    return ż_ref
end

# Get ẋ from decoder derivative (dx/dz) and ż
function dec_ẋ(dec_jac_batch, ż)
    # Size is equal to decoded features and number of batches
    dec_mult_ẋ = zero(dec_jac_batch[:,:, 1])
    for i in 1:size(dec_jac_batch, 2)
        dec_mult_ẋ[:, i] = dec_jac_batch[:,i,:] * ż[:,i]
    end
    return dec_mult_ẋ
end

function Diff_ẋ(dec_jac_batch, grad_fθ, ẋ_batch)
    dec_mult_ẋ = zero(ẋ_batch)
    for i in 1:size(dec_jac_batch, 2)
        dec_mult_ẋ[:, i] = dec_jac_batch[:,i,:] * grad_fθ[:,i]
    end
    return sum(abs2, dec_mult_ẋ - ẋ_batch)
end

# Get ż from SINDy coefficients and basis functions
function set_ż_SINDY(enc_x_batch, Θ, Ξ, smallinds)
    ż_SINDy = Zygote.Buffer(zeros(size(enc_x_batch, 1), size(Θ,1)))
    for ind in axes(enc_x_batch, 1)
        # non-zero coefficients of the ind state
        biginds = .~(smallinds[:, ind])
        ż_SINDy[ind,:] = Θ[:, biginds] * Ξ[ind]
    end
    return copy(ż_SINDy)
end

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

model_gradients = deepcopy(model)

total_samples = size(tdata.x)[2]
num_batches = ceil(Int, total_samples / method.batch_size)

# Flux gradient has problem working with the structure data directly
x = Float32.(tdata.x)
ẋ = Float32.(tdata.ẋ)

# Coefficients for the loss_kernel terms
alphas = round(sum(abs2, x) / sum(abs2, ẋ), sigdigits = 3)

# Array to store the losses
epoch_loss_array = Vector{Float64}()

# Set up the optimizer's state
opt_state = Flux.setup(Adam(), model)


@time for epoch in 1:100
    epoch_loss = 0.0
    # Shuffle the data indices for each epoch
    shuffled_indices = shuffle(1:total_samples)

    for batch in 1:1
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

        # Compute gradients using Enzyme
        Enzyme.autodiff(Reverse, (model, x_batch, ẋ_batch, ż_ref, dec_jac_batch, method, alphas) -> loss(model, x_batch, ẋ_batch, ż_ref, dec_jac_batch, method, alphas), Active, Duplicated(model, model_gradients), Const(x_batch), Const(ẋ_batch), Const(ż_ref), Const(dec_jac_batch), Const(method), Const(alphas))

        Flux.Optimise.update!(opt_state, model, model_gradients)

        # Accumulate the loss for the current batch
        epoch_loss += loss(model, x_batch, ẋ_batch, ż_ref, dec_jac_batch, basis, method, alphas)
    end
    # Compute the average loss for the epoch
    epoch_loss /= num_batches

    # Store the epoch loss
    push!(epoch_loss_array, epoch_loss)

    # Print loss after some iterations
    if epoch % 2 == 0
        println("Epoch $epoch: Average Loss: $epoch_loss")
        # println("Epoch $epoch: Coefficents: $(model[3].W)")
        println()
    end
end