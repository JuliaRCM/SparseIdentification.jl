# %%
using Distributions
using GeometricIntegrators
using Optim
using Random
using Flux
using Enzyme
using Zygote
using Distances
using Symbolics
using Plots
using RuntimeGeneratedFunctions
using LinearAlgebra
# For saving vectors to file
using DelimitedFiles
RuntimeGeneratedFunctions.init(@__MODULE__)

gr()

_prod(a, b, c, arrs...) = a .* _prod(b, c, arrs...)
_prod(a, b) = a .* b
_prod(a) = a

# generates a vector out of symbolic arrays (p,q) with a certain dimension
function get_z_vector(dims)
    @variables q[1:dims]
    @variables p[1:dims]
    z = vcat(q,p)
    return z
end

# make combinations of bases of just the order that is given 
# e.g order = 2 will give just the bases whose powers sum to 2
function poly_combos(z, order, inds...)
    if order == 0
        return Num[1]
    elseif order == length(inds)
        return [_prod([z[i] for i in inds]...)]
    else
        start_ind = length(inds) == 0 ? 1 : inds[end]
        return vcat([poly_combos(z, order, inds..., j) for j in start_ind:length(z)]...)
    end
end

# gives all bases monomials up to a certain order
function primal_monomial_basis(z, order::Int)
    return Vector{Symbolics.Num}(vcat([poly_combos(z, i) for i in 1:order]...))
end

# calculates coefficient bases up to a certain order
# mostly for use with trigonometric functions example sin(k*z),
# where k is the coefficient
function primal_coeff_basis(z, max_coeff::Int)
    return Vector{Symbolics.Num}(vcat([k .* z for k in 1:max_coeff]...))
end

# calculates +,-,*,/ between states as a new basis
# the return output is a set to avoid duplicates
function primal_operator_basis(z, operator)
    return Vector{Symbolics.Num}([operator(z[i], z[j]) for i in 1:length(z)-1 for j in i+1:length(z)] ∪ [operator(z[j], z[i]) for i in 1:length(z)-1 for j in i+1:length(z)])
end

function primal_power_basis(z, max_power::Int)
    if max_power > 0
        return Vector{Symbolics.Num}(vcat([z.^i for i in 1:max_power]...))
    elseif max_power < 0
        return Vector{Symbolics.Num}(vcat([z.^-i for i in 1:abs(max_power)]...))
    end
end

function polynomial_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = primal_monomial_basis(z, polyorder)
    primes = vcat(primes, primal_coeff_basis(z, max_coeff))
    if operator !== nothing
        primes = vcat(primes, primal_operator_basis(z, operator))
    end
    return primes
end

function trigonometric_basis(z::Vector{Symbolics.Num} = get_z_vector(2), coeffs = nothing, polyorder::Int = 0, operator = nothing, max_coeff::Int = 0)
    primes = polynomial_basis(z, polyorder = polyorder, operator = operator, max_coeff = max_coeff)
    if coeffs != nothing
    return vcat(sin.(collect(coeffs[1:Int(end/2)] .* primes)), cos.(collect(coeffs[Int(end/2)+1:end] .* primes)))
    else 
        return vcat(sin.(primes), cos.(primes))
    end
end

function exponential_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = polynomial_basis(z, polyorder = polyorder, operator = operator, max_coeff = max_coeff)
    return exp.(primes)
end

function logarithmic_basis(z::Vector{Symbolics.Num} = get_z_vector(2); polyorder::Int = 0, operator=nothing, max_coeff::Int = 0)
    primes = polynomial_basis(z, polyorder = polyorder, operator = operator, max_coeff = max_coeff)
    return log.(abs.(primes))
end

function mixed_states_basis(basis::Vector{Symbolics.Num}...)
    mixed_states = Tuple(basis)
    
    ham = Vector{Symbolics.Num}()
    for i in eachindex(mixed_states)
        for j in i+1:lastindex(mixed_states)
            ham = vcat(ham, [mixed_states[i][k] * mixed_states[j][l] for k in 1:length(mixed_states[i]) for l in 1:length(mixed_states[j])])
        end
    end
    
    return Vector{Symbolics.Num}(ham)
end

# returns the number of required coefficients for the basis
function get_numCoeffs(basis::Vector{Symbolics.Num})
    return length(basis)
end

#TODO: maybe basis shouldn't be variable number of arguments
# gets a vector of combinations of hamiltonian basis
function get_basis_set(basis::Vector{Symbolics.Num}...)
    # gets a vector of combinations of basis
    basis = vcat(basis...)
    
    # removes duplicates
    basis = Vector{Symbolics.Num}(collect(unique(basis)))

    return basis
end

mutable struct HamiltonianSINDy{T, GHT}
    basis::Vector{Symbolics.Num} # the augmented basis for sparsification
    analytical_fθ::GHT
    z::Vector{Symbolics.Num} 
    λ::T # Sparsification Parameter
    noise_level::T # Noise amplitude added to the data
    noiseGen_timeStep::T # Time step for the integrator to get noisy data 
    nloops::Int # Sparsification Loops
    batch_size::Int # Batch size for training
    basis_coeff::Float64 # Coefficient for the coefficients of the basis
    
    function HamiltonianSINDy(basis::Vector{Symbolics.Num},
        analytical_fθ::GHT = missing,
        z::Vector{Symbolics.Num} = get_z_vector(2);
        λ::T = 0.05,
        noise_level::T = 0.00,
        noiseGen_timeStep::T = 0.05,
        nloops::Int = 10,
        batch_size::Int = 1,
        basis_coeff::Float64) where {T, GHT <: Union{Base.Callable,Missing}}

        new{T, GHT}(basis, analytical_fθ, z, λ, noise_level, noiseGen_timeStep, nloops, batch_size::Int, basis_coeff::Float64)
    end
end

function gen_noisy_ref_data(method::HamiltonianSINDy, x)
    # initialize timestep data for analytical solution
    tstep = method.noiseGen_timeStep
    tspan = (zero(tstep), tstep)

    function next_timestep(x)
        prob_ref = ODEProblem((dx, t, x, params) -> method.analytical_fθ(dx, x, params, t), tspan, tstep, x)
        sol = integrate(prob_ref, Gauss(2))
        sol.q[end]
    end

    data_ref = [next_timestep(_x) for _x in x]

    # add noise
    data_ref_noisy = [_x .+ method.noise_level .* randn(size(_x)) for _x in data_ref]

    return data_ref_noisy

end

struct TrainingData{AT<:AbstractArray}
    x::AT # initial condition
    ẋ::AT # initial condition
    y::AT # noisy data at next time step

    TrainingData(x::AT, ẋ::AT, y::AT) where {AT} = new{AT}(x, ẋ, y)
    TrainingData(x::AT, ẋ::AT) where {AT} = new{AT}(x, ẋ)
end

# %%
function trig_B6_basis(a)
    z = get_z_vector(2)
    out = []
    out = vcat(out, sum(collect(a[1:2] .* z[1:2])))
    out = vcat(out, sum(collect(a[3:4] .* z[1:2])))
    out = vcat(out, sum(collect(a[1:2] .* z[3:4])))
    out = vcat(out, sum(collect(a[3:4] .* z[3:4])))
    out = vcat(out, sum(collect(a[1:2] .* z[1:2])).^2)
    out = vcat(out, sum(collect(a[3:4] .* z[1:2])).^2)
    out = vcat(out, sum(collect(a[1:2] .* z[3:4])).^2)
    out = vcat(out, sum(collect(a[3:4] .* z[3:4])).^2)
    out = vcat(out, sin.(sum(collect(a[1:2] .* (z[1:2])))))
    out = vcat(out, sin.(sum(collect(a[3:4] .* (z[1:2])))))
    out = vcat(out, cos.(sum(collect(a[1:2] .* (z[1:2])))))
    out = vcat(out, cos.(sum(collect(a[3:4] .* (z[1:2])))))
    out = vcat(out, sin.(sum(collect(a[1:2] .* (z[3:4])))))
    out = vcat(out, sin.(sum(collect(a[3:4] .* (z[3:4])))))
    out = vcat(out, cos.(sum(collect(a[1:2] .* (z[3:4])))))
    out = vcat(out, cos.(sum(collect(a[3:4] .* (z[3:4])))))
    Vector{Symbolics.Num}(out)
end

# %%
function poly_basis_maker(z, nd, polyorder)
    
    polynomial = polynomial_basis(z, polyorder=polyorder)
    poly_basis = get_basis_set(polynomial)
    return poly_basis
end

function B3(z, d, trig_polyorder, basis)
    total_trig_args = 2*(binomial(2d + trig_polyorder, trig_polyorder) - 1) # multiply by 2 for sin and cos terms
    @variables a[1:length(basis)+2*total_trig_args] # multiply by 2 to also account for amplitude coefficients
    trig_basis = trigonometric_basis(z, a[1:total_trig_args], trig_polyorder)
    poly_trig_basis = polynomial_basis(trig_basis, polyorder = 1)
    basis = get_basis_set(basis, poly_trig_basis)
    return basis, a, total_trig_args
end

function B4(z, d, trig_polyorder, basis)
    num_trig_arg = 2*(binomial(2d + trig_polyorder, trig_polyorder) - 1) # one coeffs for the freq and one the amplitude of each basis

    # code for half trig argument bases
    halfz = [(z[1]+z[2])/2, (z[3]+z[4])/2]
    num_half_trig_args = 2*length(halfz) # one coeffs for the freq and one the amplitude of each basis

    total_trig_args = num_trig_arg + num_half_trig_args

    # collects and sums combinations of basis and coefficients  
    @variables a[1:2*total_trig_args+length(basis)] # use multiply by 2 to account for amplitude coefficients also

    # code for trig argument bases
    trig_basis = trigonometric_basis(z, a[1:num_trig_arg], trig_polyorder)
    poly_trig_basis = polynomial_basis(trig_basis, polyorder = 1)

    half_trig_basis = trigonometric_basis(halfz, a[num_trig_arg+1:total_trig_args], trig_polyorder)
    poly_halfTrig_basis = polynomial_basis(half_trig_basis, polyorder = 1)

    # basis = get_basis_set(basis, poly_trig_basis)
    basis = get_basis_set(basis, poly_trig_basis, poly_halfTrig_basis)
    return basis, a, total_trig_args
end

function B5(z, d, trig_polyorder, basis)
    num_trig_arg = 2*(binomial(2d + trig_polyorder, trig_polyorder) - 1) # one coeffs for the freq and one the amplitude of each basis
    num_half_trig_args  = 12 # one coeffs for the freq and one the amplitude of each basis
    total_trig_args = num_trig_arg + num_half_trig_args

    # collects and sums combinations of basis and coefficients  
    # @variables a[1:length(basis)+2*num_trig_arg] # add a variable for each coefficient of the trig basis and also for each trig argument
    @variables a[1:2*total_trig_args+length(basis)] # use multiply by 2 to account for amplitude coefficients also

    # code for trig argument bases
    trig_basis = trigonometric_basis(z, a[1:num_trig_arg], trig_polyorder)
    poly_trig_basis = polynomial_basis(trig_basis, polyorder = 1)

    # code for half trig argument bases
    halfz = [(z[1]+z[2])/2, (z[1]+z[3])/2, (z[1]+z[4])/2, (z[2]+z[3])/2, (z[2]+z[4])/2, (z[3]+z[4])/2]
    half_trig_basis = trigonometric_basis(halfz, a[num_trig_arg+1:total_trig_args], trig_polyorder)
    poly_halfTrig_basis = polynomial_basis(half_trig_basis, polyorder = 1)

    # basis = get_basis_set(basis, poly_trig_basis)
    basis = get_basis_set(basis, poly_trig_basis, poly_halfTrig_basis)
    return basis, a, total_trig_args
end

function B6(z, d, trig_polyorder, basis)
    total_trig_args = 2d 
    @variables a[1:length(basis)+16+total_trig_args]
    trig_basis = trig_B6_basis(a)
    basis = get_basis_set(basis, trig_basis)
    return basis, a, total_trig_args
end
    
function basis_func_maker(trig_basis_case, z, d, trig_polyorder, poly_basis)
    if trig_basis_case == 1 || trig_basis_case == 2 || trig_basis_case == 3
        return B3(z, d, trig_polyorder, poly_basis)
    elseif trig_basis_case == 4
        return B4(z, d, trig_polyorder, poly_basis)
    elseif trig_basis_case == 5
        return B5(z, d, trig_polyorder, poly_basis)
    elseif trig_basis_case == 6
        return B6(z, d, trig_polyorder, poly_basis)
    else
        error("case can be integer from 1 to 6 only")
    end
end


# %%
# returns a function that can build the gradient of the hamiltonian
function ΔH_func_builder(d::Int, z::Vector{Symbolics.Num}, poly_basis::Vector{Symbolics.Num}, trig_polyorder::Int, trig_basis_case::Int) 
    # nd is the total number of dimensions of all the states, e.g. if q,p each of 3 dims, that is 6 dims in total
    nd = 2d
    Dz = Differential.(z)

    basis, a, total_trig_args =  basis_func_maker(trig_basis_case, z, d, trig_polyorder, poly_basis)
    
    # collect and sum combinations of basis and coefficients
    ham = sum(collect(a[total_trig_args+1:end] .* basis)) #start count of (a) coefficients from after the number of trig arguments
    
    # gives derivative of the hamiltonian, but not the skew-symmetric true one
    f = [expand_derivatives(dz(ham)) for dz in Dz]

    #simplify the expression potentially to make it faster
    f = simplify(f)
    
    # line below makes the vector into a hamiltonian vector field by multiplying with the skew-symmetric matrix
    ∇H = vcat(f[d+1:2d], -f[1:d])
    
    # builds a function that calculates Hamiltonian gradient and converts the function to a native Julia function
    ∇H_eval = @RuntimeGeneratedFunction(Symbolics.inject_registered_module_functions(build_function(∇H, z, a)[2]))
    
    return ∇H_eval
end

# %%
# --------------------
# Setup
# --------------------

println("Setting up...")

# 2D system with 4 variables [q₁, q₂, p₁, p₂]
const nd = 4

# initialize analytical function, keep λ smaller than ϵ so system is identifiable
ϵ = 0.5
m = 1

# two-dim simple harmonic oscillator (not used anywhere only in case some testing needed)
# H_ana(x, p, t) = ϵ * x[1]^2 + ϵ * x[2]^2 + 1/(2*m) * x[3]^2 + 1/(2*m) * x[4]^2
# H_ana(x, p, t) = cos(x[1]) + cos(x[2]) + 1/(2*m) * x[3]^2 + 1/(2*m) * x[4]^2

# Gradient function of the 2D hamiltonian
# grad_H_ana(x) = [x[3]; x[4]; -2ϵ * x[1]; -2ϵ * x[2]]
grad_H_ana(x) = [x[3]; x[4]; sin(x[1]); sin(x[2])]
function grad_H_ana!(dx, x, p, t)
    dx .= grad_H_ana(x)
end

function generate_training_data(num_samp = 10, input_range = 20, nd = 4)
    # samples in p and q space
    samp_range = LinRange(-input_range, input_range, num_samp)

    # s depend on size of nd (total dims), 4 in the case here so we use samp_range x samp_range x samp_range x samp_range
    s = collect(Iterators.product(fill(samp_range, nd)...))

    # compute vector field from x state values
    x = [collect(s[i]) for i in eachindex(s)]

    dx = zeros(nd)
    p = 0
    t = 0
    ẋ = [grad_H_ana!(copy(dx), _x, p, t) for _x in x]

    # Change to matrices for faster computations
    x = hcat(x...)
    ẋ = hcat(ẋ...)
    tdata = TrainingData(Float32.(x), Float32.(ẋ))
    return tdata
end

# dimension of each variable in the system
d = nd ÷ 2

# Symbolic variables for the states
z = get_z_vector(nd/2)

# %%
# poly_basis = poly_basis_maker(z, 2d, 1)
# returns function that builds hamiltonian gradient through symbolics
# fθ = ΔH_func_builder(d, z, poly_basis, trig_polyorder, trig_basis_case)

# %%
# bases_collect = []
# coeffs_collect = []
# augmented_bases_funcs = []


# %%
# push!(augmented_bases_funcs, fθ)
# push!(bases_collect, method.basis)
# push!(coeffs_collect, a)

# %% [markdown]
# #### Define all training functions

# %%
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

# Get ż from dz/dx and ẋ
function enc_ż(enc_jac_batch, ẋ_batch)
    ż_ref = zero(ẋ_batch)
    for i in 1:size(enc_jac_batch, 2)
        ż_ref[:, i] = enc_jac_batch[:,i,:] * (ẋ_batch[:,i])
    end
    return ż_ref
end

function evaluate_fθ(fθ, enc_x_batch, coeffs)
    f = zero(enc_x_batch[:,1])
    out = zero(enc_x_batch)
    for i in 1:size(enc_x_batch, 2)
        fθ(f, enc_x_batch[:,i], coeffs)
        out[:,i] = f
    end
    return out
end

# Get ẋ from decoder derivative (dx/dz) and ż
function dec_ẋ(dec_jac_batch, ż)
    dec_mult_ẋ = zero(ż)
    for i in 1:size(dec_jac_batch, 2)
        dec_mult_ẋ[:, i] = dec_jac_batch[:,i,:] * ż[:,i]
    end
    return dec_mult_ẋ
end

function Diff_ż(grad_fθ, ż_ref)
    return sum(abs2, (grad_fθ - ż_ref))
end

function Diff_ẋ(dec_jac_batch, grad_fθ, ẋ_ref)
    ẋ_SINDy = zero(ẋ_ref)
    for i in 1:size(dec_jac_batch, 2)
        ẋ_SINDy[:, i] = dec_jac_batch[:,i,:] * grad_fθ[:,i]
    end
    return sum(abs2, (ẋ_SINDy - ẋ_ref))
end

function set_coeffs(model_Coeffs, Ξ, biginds)
    coeffs = zero(model_Coeffs)
    coeffs[biginds] = Ξ
    return coeffs
end

function loss(model, x_batch, ẋ_batch, ż_ref_batch, dec_jac_batch, alphas, method)
    enc_x_batch = model[1].W(x_batch)

    coeffs = model[3].W

    # Compute the reconstruction loss for the entire batch
    L_r = sum(abs2, model[2].W(enc_x_batch) - x_batch)

    # Note: grad_fθ, dec_mult_ẋ, and L_c in loss function so model acts on terms in loss function
    # and gradient can see that and use that for its update calculations

    # encoded gradient from SINDy
    grad_fθ = evaluate_fθ(fθ, enc_x_batch, coeffs)

    # Difference b/w encoded gradients from SINDy and reference
    #TODO: could also try alphas/10 instead of alphas/100
    L_ż = alphas / 100 * Diff_ż(grad_fθ, ż_ref_batch)

    # Difference b/w decoded-encoded gradients from SINDy against reference
    ẋ_diff = Diff_ẋ(dec_jac_batch, grad_fθ, ẋ_batch)
    L_ẋ = alphas * ẋ_diff

    # Compute the total loss for the entire batch
    batchLoss = L_r + L_ż + L_ẋ

    # Mean of the coefficients averaged
    L_c = sum(abs, coeffs) / length(coeffs)

    batch_loss_average = batchLoss / size(x_batch, 2) + method.basis_coeff * L_c
    
    return batch_loss_average
end

# %% [markdown]
# #### Define the model

# %%
function set_model(data, num_coeff, initial_coeffs::String)
    ld = size(data.x)[1]
    ndim = size(data.x)[1]

    encoder = Chain(
    Dense(ndim => ld)
    )

    decoder = Chain(
        Dense(ld => ndim)
    )

    if initial_coeffs == "zeros"
        model = ( 
            (W = encoder,),
            (W = decoder,),
            (W = zeros(Float32, num_coeff),), 
        )
    end

    if initial_coeffs == "ones"
        model = ( 
            (W = encoder,),
            (W = decoder,),
            (W = ones(Float32, num_coeff),),
        )
    end

    # set initial encoder/decoder weights to identity
    # TODO: try not setting these to identity and see how well it works
    model[1].W.layers[1].weight .= Matrix(LinearAlgebra.I, ndim, ld)
    model[2].W.layers[1].weight .= Matrix(LinearAlgebra.I, ld, ndim)
    
    return model
end

# %%
function setup_test(data, method, model, alphas)
    # Derivatives of the encoder and decoder
    enc_jac = batched_jacobian(model[1].W, data.x)
    dec_jac = batched_jacobian(model[2].W, model[1].W(data.x))

    # encoded gradient ż = dz/dx*ẋ
    ż_ref = enc_ż(enc_jac, data.ẋ)

    # Array to store the losses
    Initial_loss_array = Vector{Float32}()

    # Store the epoch loss
    push!(Initial_loss_array, loss(model, data.x, data.ẋ, ż_ref, dec_jac, alphas, method))
    return Initial_loss_array
end

# %%
function initial_loop(model, method, data, total_samples, num_batches, alphas, opt, model_type, Initial_loss_array)
    opt_state = Flux.setup(opt, model)
    for epoch in 1:1000
        model_gradients = deepcopy(model)
        epoch_loss = 0.0
        # Shuffle the data indices for each epoch
        shuffled_indices = shuffle(1:total_samples)
        for batch in 1:num_batches
            # Get the indices for the current batch
            batch_start = (batch - 1) * method.batch_size + 1
            batch_end = min(batch * method.batch_size, total_samples)
            batch_indices = shuffled_indices[batch_start:batch_end]

            # Extract the data for the current batch
            x_batch = data.x[:, batch_indices]
            ẋ_batch = data.ẋ[:, batch_indices]

            # Derivatives of the encoder and decoder
            enc_jac_batch = batched_jacobian(model[1].W, x_batch)
            dec_jac_batch = batched_jacobian(model[2].W, model[1].W(x_batch))
            
            # Get the encoded derivative: ż
            ż_ref_batch = enc_ż(enc_jac_batch, ẋ_batch)
            
            # Compute gradients using Enzyme
            Enzyme.autodiff(Reverse, (model, x_batch, ẋ_batch, ż_ref_batch, dec_jac_batch, alphas, method) -> loss(model, x_batch, ẋ_batch, ż_ref_batch, dec_jac_batch, alphas, method), Active, Duplicated(model, model_gradients), Const(x_batch), Const(ẋ_batch), Const(ż_ref_batch), Const(dec_jac_batch), Const(alphas), Const(method))

            # Update the parameters
            # Flux.Optimise.update!(opt_state, model, model_gradients)

            # Update the parameters
            if model_type == "fixed"
                Flux.Optimise.update!(opt_state[3], model[3], model_gradients[3])
            else
                Flux.Optimise.update!(opt_state, model, model_gradients)
            end

            # Accumulate the loss for the current batch
            epoch_loss += loss(model, x_batch, ẋ_batch, ż_ref_batch, dec_jac_batch, alphas, method)
        end
        # Compute the average loss for the epoch
        epoch_loss /= num_batches

        # Store the epoch loss
        push!(Initial_loss_array, epoch_loss)
    end
    return model, Initial_loss_array
end

# %%
function sparse_loss(enc_paras, dec_paras, Ξ, model_Coeffs, x_batch, ẋ_batch, ż_ref, dec_jac_batch, alphas, method, biginds)
    coeffs = set_coeffs(model_Coeffs, Ξ, biginds)

    enc_x_batch = enc_paras(x_batch)

    # Compute the reconstruction loss for the entire batch
    L_r = sum(abs2, dec_paras(enc_x_batch) - x_batch)

    # Note: grad_fθ, dec_mult_ẋ, and L_c in loss function so model acts on terms in loss function
    # and gradient can see that and use that for its update calculations

    # encoded gradient from SINDy
    grad_fθ = evaluate_fθ(fθ, enc_x_batch, coeffs)

    # Difference b/w encoded gradients from SINDy and reference
    L_ż = alphas / 10 * Diff_ż(grad_fθ, ż_ref)

    # Difference b/w decoded-encoded gradients from SINDy against reference
    ẋ_diff = Diff_ẋ(dec_jac_batch, grad_fθ, ẋ_batch)
    L_ẋ = alphas * ẋ_diff

    # Compute the total loss for the entire batch
    batchLoss = L_r + L_ż + L_ẋ

    # Mean of the coefficients averaged
    L_c = sum(abs, model_Coeffs) / length(model_Coeffs)

    batch_loss_average = batchLoss / size(x_batch, 2) + method.basis_coeff * L_c
    
    return batch_loss_average
end

# %%
function sindy_loop(model, method, data, total_samples, num_batches, alphas, opt, model_type)
    # Array to store the losses of each SINDy loop
    SINDy_loss_array = Vector{Vector{Float32}}()  # Store vectors of losses

    # Initialize smallinds before the loop
    smallinds = falses(size(model[3].W))

    for n in 1:method.nloops
        # find coefficients below λ threshold
        smallinds .= abs.(model[3].W) .< method.λ

        biginds = .~smallinds

        # check if there are any small coefficients != 0 left
        all(model[3].W[smallinds] .== 0) && break

        # set all small coefficients to zero
        model[3].W[smallinds] .= 0

        # Set up the optimizer's state
        Ξ = model[3].W[biginds]

        # Array to store the losses of each epoch
        epoch_loss_array = Vector{Float64}()

        # Set up the optimizer's state
        opt_state = Flux.setup(opt, (model[1].W, model[2].W, Ξ))
        # if model_type == "fixed"
        #     Flux.Optimisers.freeze!(opt_state[1])
        #     Flux.Optimisers.freeze!(opt_state[2])
        # end

        for epoch in 1:500
            # Gradients of the encoder, decoder and model coefficients
            grad_W1 = deepcopy(model[1].W)
            grad_W2 = deepcopy(model[2].W)
            grad_W3 = deepcopy(Ξ)
            
            epoch_loss = 0.0
            # Shuffle the data indices for each epoch
            shuffled_indices = shuffle(1:total_samples)

            for batch in 1:num_batches
                # Get the indices for the current batch
                batch_start = (batch - 1) * method.batch_size + 1
                batch_end = min(batch * method.batch_size, total_samples)
                batch_indices = shuffled_indices[batch_start:batch_end]

                # Extract the data for the current batch
                x_batch = data.x[:, batch_indices]
                ẋ_batch = data.ẋ[:, batch_indices]

                # Derivatives of the encoder and decoder
                enc_jac_batch = batched_jacobian(model[1].W, x_batch)
                dec_jac_batch = batched_jacobian(model[2].W, model[1].W(x_batch))
                
                # Get the encoded derivative: ż
                ż_ref_batch = enc_ż(enc_jac_batch, ẋ_batch)

                # Compute gradients using Enzyme
                Enzyme.autodiff(Reverse, (enc_paras, dec_paras, Ξ, model_Coeffs, x_batch, ẋ_batch, ż_ref_batch, dec_jac_batch, alphas, method, biginds) -> sparse_loss(enc_paras, dec_paras, Ξ, model_Coeffs, x_batch, ẋ_batch, ż_ref_batch, dec_jac_batch, alphas, method, biginds), Active, Duplicated(model[1].W, grad_W1), Duplicated(model[2].W, grad_W2), Duplicated(Ξ, grad_W3), Const(model[3].W), Const(x_batch), Const(ẋ_batch), Const(ż_ref_batch), Const(dec_jac_batch), Const(alphas), Const(method), Const(biginds))

                # Update the parameters
                if model_type == "fixed"
                    Flux.Optimise.update!(opt_state[3], Ξ, grad_W3)
                else
                    Flux.Optimise.update!(opt_state, (model[1].W, model[2].W, Ξ), (grad_W1, grad_W2, grad_W3))
                end

                # Accumulate the loss for the current batch
                epoch_loss += sparse_loss(model[1].W, model[2].W, Ξ, model[3].W, x_batch, ẋ_batch, ż_ref_batch, dec_jac_batch, alphas, method, biginds)
            end

            model[3].W[biginds] .= Ξ

            # Compute the average loss for the epoch
            epoch_loss /= num_batches
            
            # Store the epoch loss
            push!(epoch_loss_array, epoch_loss)
        end

        # Store the SINDy loop loss
        push!(SINDy_loss_array, epoch_loss_array)
    end

    # Convert vector of vectors to a single vector
    SINDy_loss_array = vcat(SINDy_loss_array...)

    return model, SINDy_loss_array
end

# %%
function final_loop(model, method, data, total_samples, num_batches, alphas, opt, model_type)
    # Array to store the losses of each epoch
    final_loss_array = Vector{Float64}()

    # find coefficients below λ threshold
    smallinds = abs.(model[3].W) .<= 0

    biginds = .~smallinds

    # Set up the optimizer's state
    Ξ = model[3].W[biginds]

    # Set up the optimizer's state
    opt_state = Flux.setup(opt, (model[1].W, model[2].W, Ξ))
    # if model_type == "fixed"
    #     Flux.Optimisers.freeze!(opt_state[1])
    #     Flux.Optimisers.freeze!(opt_state[2])
    # end

    for epoch in 1:1000
        # Gradients of the encoder, decoder and model coefficients
        grad_W1 = deepcopy(model[1].W)
        grad_W2 = deepcopy(model[2].W)
        grad_W3 = deepcopy(Ξ)
        
        epoch_loss = 0.0
        # Shuffle the data indices for each epoch
        shuffled_indices = shuffle(1:total_samples)

        for batch in 1:num_batches
            # Get the indices for the current batch
            batch_start = (batch - 1) * method.batch_size + 1
            batch_end = min(batch * method.batch_size, total_samples)
            batch_indices = shuffled_indices[batch_start:batch_end]

            # Extract the data for the current batch
            x_batch = data.x[:, batch_indices]
            ẋ_batch = data.ẋ[:, batch_indices]

            # Derivatives of the encoder and decoder
            enc_jac_batch = batched_jacobian(model[1].W, x_batch)
            dec_jac_batch = batched_jacobian(model[2].W, model[1].W(x_batch))
            
            # Get the encoded derivative: ż
            ż_ref_batch = enc_ż(enc_jac_batch, ẋ_batch)

            # Compute gradients using Enzyme
            Enzyme.autodiff(Reverse, (enc_paras, dec_paras, Ξ, model_Coeffs, x_batch, ẋ_batch, ż_ref_batch, dec_jac_batch, alphas, method, biginds) -> sparse_loss(enc_paras, dec_paras, Ξ, model_Coeffs, x_batch, ẋ_batch, ż_ref_batch, dec_jac_batch, alphas, method, biginds), Active, Duplicated(model[1].W, grad_W1), Duplicated(model[2].W, grad_W2), Duplicated(Ξ, grad_W3), Const(model[3].W), Const(x_batch), Const(ẋ_batch), Const(ż_ref_batch), Const(dec_jac_batch), Const(alphas), Const(method), Const(biginds))

            # Update the parameters
            if model_type == "fixed"
                Flux.Optimise.update!(opt_state[3], Ξ, grad_W3)
            else
                Flux.Optimise.update!(opt_state, (model[1].W, model[2].W, Ξ), (grad_W1, grad_W2, grad_W3))
            end

            # Accumulate the loss for the current batch
            epoch_loss += sparse_loss(model[1].W, model[2].W, Ξ, model[3].W, x_batch, ẋ_batch, ż_ref_batch, dec_jac_batch, alphas, method, biginds)
        end

        model[3].W[biginds] .= Ξ

        # Compute the average loss for the epoch
        epoch_loss /= num_batches
        
        # Store the epoch loss
        push!(final_loss_array, epoch_loss)
    end
    return model, final_loss_array
end

# %%
function save_plot(basis_directory, name, model_type, input_range, batch_size, loss_array, log_plot)
    # Construct the file name based on loop variables
    loss_plot = joinpath(basis_directory, "$(name)_model_$(model_type)_$(input_range)_$batch_size.png")
    # Save the plot to the file
    if log_plot == true
        plot(log.(loss_array), label = "$(name) Optimization Loss", xlabel = "Iterations", ylabel = "Log Loss")
    else
        plot(log.(loss_array), label = "$(name) Optimization Loss", xlabel = "Iterations", ylabel = "Loss")
    end
    savefig(loss_plot)
end

# %%
function running_loop(data, method, model, model_type, input_range, basis_directory)   
    total_samples = size(data.x)[2]
    num_batches = ceil(Int, total_samples / method.batch_size)

    # Coefficients for the loss_kernel terms
    alphas = round(sum(abs2, data.x) / sum(abs2, data.ẋ), sigdigits = 3)    
    
    Initial_loss_array = setup_test(data, method, model, alphas)

    # Set up the optimizer's state
    # TODO: could try different learning rates
    opt = Adam(0.001, (0.9, 0.8))
    # opt_state_initial = Flux.setup(opt, model)
    # if model_type == "fixed"
    #     Flux.Optimisers.freeze!(opt_state_initial[1])
    #     Flux.Optimisers.freeze!(opt_state_initial[2])
    # end

    log_plot = true
    println("initial loss calculation...")
    model, Initial_loss_array = initial_loop(model, method, data, total_samples, num_batches, alphas, opt, model_type, Initial_loss_array)
    println("plotting initial loss calculation...")
    save_plot(basis_directory, "initial", model_type, input_range, method.batch_size, Initial_loss_array, log_plot)

    println("sindy loss calculation...")
    model, SINDy_loss_array = sindy_loop(model, method, data, total_samples, num_batches, alphas, opt, model_type)
    println("plotting sindy loss calculation...")
    save_plot(basis_directory, "sindy", model_type, input_range, method.batch_size, SINDy_loss_array, log_plot)

    # reduce learning rate before final loss calculation
    opt = Adam(0.0001, (0.9, 0.8))
    
    log_plot = false
    println("final loss calculation...")
    model, final_loss_array = final_loop(model, method, data, total_samples, num_batches, alphas, opt, model_type)
    println("plotting final loss calculation...")
    save_plot(basis_directory, "final", model_type, input_range, method.batch_size, final_loss_array, log_plot)
    return model
end

# %% [markdown]
# #  Running Loop

# %%
basis_cases = [
    # (trig_basis_case, poly_basis_power, trig_basis_power)
    (1, 2, 1),
    (2, 3, 1),
    (3, 3, 2),
    (4, 2, 1),
    (5, 2, 1),
    (6, 2, -1), # -1 here because it is unused in this case
]
model_types = ["general", "fixed"]
sample_ranges = [5, 10, 20]
batch_sizes = [256, 512, 1024]

# %%
# Initialize the runtime function
fθ = ΔH_func_builder(d, z, poly_basis_maker(z, nd, 2), 1, 1)

# Initialize the models' arrays
model_encoder_array = []
model_coeffs_array = []
model_decoder_array = []

# Specify the file paths
model_encoder_file = "modelEncoders.csv"
model_coeffs_file = "modelCoeffs.csv"
model_decoder_file = "modelDecoders.csv"

# Create a directory to save the plots
output_directory = "plots"
if !isdir(output_directory)
    mkdir(output_directory)
end


# %% [markdown]
# # Start the Testing Loop

# %%
method = HamiltonianSINDy(z, grad_H_ana!, z, λ = 0.05, noise_level = 0.0, noiseGen_timeStep = 0.0, batch_size = 1, basis_coeff = 0.62)

# *********************************************
# *********************************************
# Run the testing loop
# *********************************************
# *********************************************

@time begin
    for (trig_basis_case, poly_basis_power, trig_basis_power) in basis_cases
        # Create a directory for the current basis_case
        basis_directory = joinpath(output_directory, "basis_$trig_basis_case")
        if !isdir(basis_directory)
            mkdir(basis_directory)
        end

        poly_basis = poly_basis_maker(z, nd, poly_basis_power)
        # returns function that builds hamiltonian gradient through symbolics
        global fθ = ΔH_func_builder(d, z, poly_basis, trig_basis_power, trig_basis_case)
        basis, a, total_trig_args = basis_func_maker(trig_basis_case, z, d, trig_basis_power, poly_basis)
        method.basis = basis
        for input_range in sample_ranges
            #TODO: could also try bigger or smaller num_samp
            num_samp = 12 # number of samples are then actually 12*12*12*12 = 20736 for 4 variables
            tdata = generate_training_data(num_samp, input_range, nd)
            #TODO: could also set basis_coeff to different values and see what happens
            # method = HamiltonianSINDy(basis, grad_H_ana!, z, λ = 0.05, noise_level = 0.0, noiseGen_timeStep = 0.0, batch_size = batch_size, basis_coeff = 0.62)
            for batch_size in batch_sizes 
                method.batch_size = batch_size
                for model_type in model_types
                    println()
                    #TODO: could also set initialization to "zeros" and see what happens
                    ###### Define here because each combination should have a new model
                    model = set_model(tdata, length(a), "ones")
                    println("basis_$trig_basis_case, model_type:$model_type, sample_domain: (-$input_range, $input_range), batch_size: $batch_size")

                    elapsed_time = @elapsed begin
                        model = running_loop(tdata, method, model, model_type, input_range, basis_directory)  
                    end
                    println("Elapsed time: $elapsed_time seconds")

                    push!(model_encoder_array, model[1].W.layers[1].weight)
                    push!(model_encoder_array, model[1].W.layers[1].bias)
                    # Save the vector to the encoder file
                    writedlm(model_encoder_file, model_encoder_array, ',')
                    
                    push!(model_coeffs_array, model[3].W)
                    # Save the vector to the coeffs file
                    writedlm(model_coeffs_file, model_coeffs_array, ',')

                    push!(model_decoder_array, model[2].W.layers[1].weight)
                    push!(model_decoder_array, model[2].W.layers[1].bias)
                    # Save the vector to the decoder file
                    writedlm(model_decoder_file, model_decoder_array, ',')
                end
            end
        end
    end
end