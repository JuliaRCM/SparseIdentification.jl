# This file uses multiple trajectories to find the SINDY solution of a hamiltonian system in 1D

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
const d = 1

#############################################################
#############################################################
# IMP SETUP NOTE: 2D system with 4 variables [q₁, q₂, p₁, p₂]
const nd = 4d
#############################################################
#############################################################

# search space up to polyorder polynomials (highest polynomial order)
const polyorder = 3 

######################################################################
######################################################################
# maximum wave number of trig basis for function library to explore
# trig_wave_num can be adjusted if higher frequency arguments expected
const trig_wave_num = 0
######################################################################
######################################################################

# 1 dim each of [q₁, q₂, p₁, p₂] gives 4*d = 4 variables
out = zeros(nd)

# initialize analytical function, keep ϵ bigger than λ so system is identifiable
ϵ = 0.5
m = 1

tstep = 0.01
tspan = (0.0,25.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

# noise level
eps = 0.05

# lambda parameter (must be close to noise value so that only coeffs with value around the noise are sparsified away)
lambda = 0.05


# one-dim simple harmonic oscillator (not used anywhere only in case some testing needed)
# H_ana(x, p, t) = 1/(2*m) * x[3]^2 + ϵ * x[1]^2 + 0 * x[2] + 0 * x[4]

# Gradient function of the 1D hamiltonian
grad_H_ana(x, p, t) = [x[3]; 0.0; -2ϵ * x[1]; 0.0]

#grad_H_ana(x, p, t) = [x[3]; 0.0; sin(x[1]); 0.0]

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

# generate num_trajec number of sample trajectories for training data
num_trajec = 5

# initialize vector of matrices to store ODE solve output
x = Vector{Matrix{Float64}}(undef, 0)

# initialize time vector of vectors to store solution at specific times of ODE solveoutput
time = Vector{Vector{Float64}}(undef, 0)

# initial function x₀ = [q₁, q₂, p₁, p₂]
x₀ = zeros(nd, num_trajec)

for i in 1:num_trajec

    # generate random q₁ and p₁ array in range 1 to 10 with one decimal place precision
    x₀[1,i] = rand(1:.1:10)
    x₀[3,i] = rand(1:.1:10)

    prob = ODEProblem(grad_H_ana, x₀[:,i], tspan)
    data = ODE.solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)

    push!(x, Array(data))

    push!(time, Array(data.t))

end
x = reduce(vcat, x)

# compute vector field from x state values at each timestep
# stored as matrix with dims [nd,ntime]
ẋ = zero(x)

for i in axes(ẋ,2)
    for j in 1:size(ẋ,1)-3
        ẋ[j:j+3,i] = grad_H_ana(x[j:j+3,i], 0, 0)
    end
end

# collect training data
tdata = TrainingData(x, ẋ)



#########################################################################
# HAM = hamiltonianFunction(x₀, a)
# # calcuate Hessian of Hamiltonian w.r.t (a) coeffecients
# ∂H²∂a²(x,a) = ForwardDiff.hessian(a -> hamiltonianFunction(x, a), a)
# hess_a = ∂H²∂a²(x₀,a)
#########################################################################


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

# choose SINDy method
# (λ parameter must be close to noise value so that only coeffs with value around the noise are sparsified away)
method = HamiltonianSINDy(grad_H_ana, λ = lambda, noise_level = eps, polyorder = polyorder, trigonometric = trig_wave_num)

# compute vector field
vectorfield = VectorField(method, tdata)

println(vectorfield.coefficients)

# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System...")

# initialize vector of matrices to store ODE solve output
xid = Vector{Matrix{Float64}}(undef, 0)

# initialize time as a vector of vectors to store solution at specific times of ODE solve output
approx_time = Vector{Vector{Float64}}(undef, 0)

for i in 1:num_trajec
    prob_approx = ODEProblem(vectorfield, x₀[:,i], tspan, a)
    output = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange) 

    # xid stores the different trajectories, as a vector of matrices
    push!(xid, Array(output))
    push!(approx_time, Array(output.t))
end

# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

for i in 1:num_trajec

    p1 = plot()
    plot!(p1, time[i], x[i + 3*(i-1), :], markershape=:circle, label = "Data q₁")
    plot!(p1, approx_time[i], xid[i][1,:], label = "Identified q₁")

    xlabel!("Time")
    ylabel!("q₁")

    p3 = plot()
    plot!(p3, time[i], x[i + 3*(i-1) + 2, :], markershape=:circle, label = "Data p₁")
    plot!(p3, approx_time[i], xid[i][3,:], label = "Identified p₁")
    
    xlabel!("Time")
    ylabel!("p₁")
    display(plot(p1, p3, title="Analytical vs Calculated gradient in a 1D system"))
    
end