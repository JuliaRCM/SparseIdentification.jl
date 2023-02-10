#This file uses uniform sampling to find the SINDY solution of a hamiltonian system in 1D and 2D

using DifferentialEquations
using Distributions
using ODE
using Plots
using Random
using SparseIdentification
using Zygote
using ForwardDiff
using Optim

gr()


# --------------------
# Setup
# --------------------

println("Setting up...")

# define the number of variables, q,p in this case gives 2 variables
const d = 2

# 2D system with 4 variables [q₁, q₂, p₁, p₂] where q₂ = 0 and p₂ = 0
nd = 4

# search space up to polyorder polynomials (highest polynomial order)
const polyorder = 3 

# trigonometric functions
usesine = true

# maximum wave number of trig basis for function library to explore
# trig_wave_num can be adjusted if higher frequency arguments expected
const trig_wave_num = 3

# binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
nparam = calculate_nparams(d, polyorder, usesine, trig_wave_num)

# (a) initialized to a vector of zeros b/c easier to optimze zeros for our case
a = zeros(nparam)

# 2 dims each of p and q gives 2*d = 4 variables
out = zeros(nd)

# noise level
eps = 0.05

# lambda parameter (must be close to noise value so that only coeffs with value around the noise are sparsified away)
lambda = 0.05

# initialize analytical function, keep ϵ bigger than lambda so system is identifiable
ϵ = 0.5
m = 1

# returns function that builds hamiltonian gradient through symbolics
" the function hamilGradient_general!() needs this "
∇H_sparse = hamilGrad_func_builder(d, usesine, polyorder, nparam, trig_wave_num)

" wrapper function for generalized SINDY hamiltonian gradient.
Needs the output of ∇H_sparse to work!
It is in a syntax that is suitable to be evaluated by a loss function
for optimization "

function hamilGradient_general!(out, z, a::AbstractVector{T}, t) where T
    ∇H_sparse(out, z, a)
    return out
end

# one-dim simple harmonic oscillator (not used anywhere only in case some testing needed)
# H_ana(x, p, t) = ϵ * x[1]^2 + 1/(2*m) * x[3]^2 + 0 * x[2] + 0 * x[4]

# Gradient function of the 1D hamiltonian
#grad_H_ana(x, p, t) = [x[3]; 0.0; -2ϵ * x[1]; 0.0]
grad_H_ana(x, p, t) = [x[3]; 0.0; sin(x[1]); 0.0]


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# number of samples
num_samp = 100

# samples in p and q space
samp_range = LinRange(-20, 20, num_samp)

# initialize vector of matrices to store ODE solve output
x = zeros(4*num_samp, num_samp)

for i in 1:nd:nd*num_samp

    # generate q₁ and p₁ matrix in given range 
    x[i,:] = samp_range
    x[i+2,:] .= samp_range[Int((i+3)/4)]
end

# compute vector field from x state values
# stored as matrix with dims [nd,ntime]
ẋ = zero(x)

for i in axes(ẋ,2)
    for j in 1:size(ẋ,1)-3
        ẋ[j:j+3,i] = grad_H_ana(x[j:j+3,i], 0, 0)
    end
end

# add noise
ẋ .+= eps .* randn(size(ẋ))


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

println("Sparsify Dynamics...")

function loss(a::AbstractVector)
    result = zeros(eltype(a), size(ẋ))

    out = zeros(eltype(a), nd)
    
    for i in 1:size(x,1)-3 
        for j in axes(x, 2)
            result[i:i+3,j] = hamilGradient_general!(out, x[i:i+3,j], a, 0)
        end
    end

    return mapreduce( y -> y^2, +, (ẋ) .- result)
end

a .= sparsify_hamiltonian_dynamics(a, loss, lambda)

println(a)


# ----------------------------------------
# Compute approximate gradient
# ----------------------------------------

println("Compute approximate gradient...")

xid = zeros(size(ẋ))
out = zeros(nd)

for i in 1:size(x,1)-3 
    for j in axes(x, 2)
        xid[i:i+3,j] = hamilGradient_general!(out, x[i:i+3,j], a, 0)
    end
end


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

# calculate difference between answers
diff = (xid.- ẋ) ./ ẋ

display(plot(heatmap(diff), title="Cos(q₁): Relative difference b/w analytical and calculated gradient in a 1D system", titlefontsize=8))












########################################################################################################################
########################################################################################################################
########################################################################################################################
################################################# 2D system ############################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

using DifferentialEquations
using Distributions
using ODE
using Plots
using Random
using SparseIdentification
using Zygote
using ForwardDiff
using Optim

gr()


# --------------------
# Setup
# --------------------

println("Setting up...")

# define the number of variables, q,p in this case gives 2 variables
const d = 2

# 2D system with 4 variables [q₁, q₂, p₁, p₂] where q₂ = 0 and p₂ = 0
nd = 4

# search space up to polyorder polynomials (highest polynomial order)
const polyorder = 3 

# trigonometric functions
usesine = true

# maximum wave number of trig basis for function library to explore
# trig_wave_num can be adjusted if higher frequency arguments expected
const trig_wave_num = 3

# binomial used to get the combination of variables till the highest order without repeat, nparam = 34 for 3rd order, with z = q,p each of 2 dims
nparam = calculate_nparams(d, polyorder, usesine, trig_wave_num)

# (a) initialized to a vector of zeros b/c easier to optimze zeros for our case
a = zeros(nparam)

# 2 dims each of p and q gives 2*d = 4 variables
out = zeros(nd)

# noise level
eps = 0.05

# lambda parameter (must be close to noise value so that only coeffs with value around the noise are sparsified away)
lambda = 0.05

# initialize analytical function, keep ϵ bigger than lambda so system is identifiable
ϵ = 0.5
m = 1

# returns function that builds hamiltonian gradient through symbolics
" the function hamilGradient_general!() needs this "
∇H_sparse = hamilGrad_func_builder(d, usesine, polyorder, nparam, trig_wave_num)

" wrapper function for generalized SINDY hamiltonian gradient.
Needs the output of ∇H_sparse to work!
It is in a syntax that is suitable to be evaluated by a loss function
for optimization "

function hamilGradient_general!(out, z, a::AbstractVector{T}, t) where T
    ∇H_sparse(out, z, a)
    return out
end

# two-dim simple harmonic oscillator (not used anywhere only in case some testing needed)
# H_ana(x, p, t) = 1/(2*m) * x[3]^2 + ϵ * x[1]^2 + 1/(2*m) * x[4]^2 + ϵ * x[2]^2

# Gradient function of the 2D hamiltonian
#grad_H_ana(x, p, t) = [x[3]; x[4]; -2ϵ * x[1]; -2ϵ * x[2]]
 
grad_H_ana(x, p, t) = [x[3]; x[4]; sin(x[1]); sin(x[2])]

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# number of samples
num_samp = 100

# samples in p and q space
samp_range = LinRange(-20, 20, num_samp)

# initialize vector of matrices to store ODE solve output
x = zeros(4*num_samp, num_samp)

for i in 1:nd:nd*num_samp-3

    # generate q₁ and p₁ matrix in given range 
    x[i,:] = samp_range
    x[i+1,:] = rand(-20:.1:20, num_samp)
    x[i+2,:] .= samp_range[Int((i+3)/4)]
    x[i+3,:] = rand(-20:.1:20, num_samp)
end

# compute vector field from x state values
# stored as matrix with dims [nd,ntime]
ẋ = zero(x)

for i in axes(ẋ,2)
    for j in 1:size(ẋ,1)-3
        ẋ[j:j+3,i] = grad_H_ana(x[j:j+3,i], 0, 0)
    end
end

# add noise
ẋ .+= eps .* randn(size(ẋ))


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

println("Sparsify Dynamics...")

function loss(a::AbstractVector)
    result = zeros(eltype(a), size(ẋ))

    out = zeros(eltype(a), nd)
    
    for i in 1:size(x,1)-3 
        for j in axes(x, 2)
            result[i:i+3,j] = hamilGradient_general!(out, x[i:i+3,j], a, 0)
        end
    end

    return mapreduce( y -> y^2, +, (ẋ) .- result)
end

a .= sparsify_hamiltonian_dynamics(a, loss, lambda)

println(a)


# ----------------------------------------
# Compute approximate gradient
# ----------------------------------------

println("Compute approximate gradient...")

xid = zeros(size(ẋ))
out = zeros(nd)

for i in 1:size(x,1)-3 
    for j in axes(x, 2)
        xid[i:i+3,j] = hamilGradient_general!(out, x[i:i+3,j], a, 0)
    end
end


# ----------------------------------------
# Plot Relative Difference HeatMap
# ----------------------------------------

println("Plotting Relative Difference HeatMap...")

# calculate difference between answers
diff = (xid.- ẋ) ./ ẋ

display(plot(heatmap(diff), title="Cos(q₁, q₂): Relative Difference b/w analytical and calculated gradient in a 2D system", titlefontsize=8))


# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System...")

tstep = 0.01
tspan = (0.0,25.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

# initialize vector of matrices to store ODE solve output
testXid = Vector{Matrix{Float64}}(undef, 0)

# initialize time as a vector of vectors to store solution at specific times of ODE solve output
approx_time = Vector{Vector{Float64}}(undef, 0)

for i in 1:size(x,1)-3 
    for j in 1:2
        prob_approx = ODEProblem(hamilGradient_general!, x[i:i+3,j], tspan, a)
        output = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange) 
    
        # xid stores the different trajectories, as a vector of matrices
        push!(testXid, Array(output))
        push!(approx_time, Array(output.t))
    end
end


# initialize vector of matrices to store ODE solve output
testX = Vector{Matrix{Float64}}(undef, 0)

# initialize time vector of vectors to store solution at specific times of ODE solveoutput
time = Vector{Vector{Float64}}(undef, 0)


for i in 1:size(x,1)-3 
    for j in 1:2

        prob = ODEProblem(grad_H_ana, x[i:i+3,j], tspan)
        data = ODE.solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)

        push!(testX, Array(data))

        push!(time, Array(data.t))
    end

end
testX = reduce(vcat, testX)


# ----------------------------------------
# Plot Dynamics
# ----------------------------------------

println("Plotting System Dynamics...")

for i in 1:5

    p1 = plot()
    plot!(p1, time[i], testX[i + 3*(i-1), :], markershape=:circle, label = "Data q₁")
    plot!(p1, approx_time[i], testXid[i][1,:], label = "Identified q₁")

    xlabel!("Time")
    ylabel!("q₁")

    p3 = plot()
    plot!(p3, time[i], testX[i + 3*(i-1) + 2, :], markershape=:circle, label = "Data p₁")
    plot!(p3, approx_time[i], testXid[i][3,:], label = "Identified p₁")
    
    xlabel!("Time")
    ylabel!("p₁")
    display(plot(p1, p3, title="Analytical vs Calculated gradient in a 2D system"))
    
end

for i in 1:5

    p2 = plot()
    plot!(p2, time[i], testX[i + 3*(i-1) + 1, :], markershape=:circle, label = "Data q₂")
    plot!(p2, approx_time[i], testXid[i][2,:], label = "Identified q₂")

    xlabel!("Time")
    ylabel!("q₂")

    p4 = plot()
    plot!(p4, time[i], testX[i + 3*(i-1) + 3, :], markershape=:circle, label = "Data p₂")
    plot!(p4, approx_time[i], testXid[i][4,:], label = "Identified p₂")
    
    xlabel!("Time")
    ylabel!("p₄")
    display(plot(p2, p4, title="Analytical vs Calculated gradient in a 2D system"))
    
end
