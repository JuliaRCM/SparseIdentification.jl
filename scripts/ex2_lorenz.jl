
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

# search space up to fifth order polynomials
polyorder = 5

# no trigonometric functions
usesine = false

#Lorenz's parameters (chaotic)
sigma = 10.0
beta = 8/3
rho = 28.0

# generate basis
basis = CompoundBasis()

# initial data
x₀ = [-8.0, 8.0, 27.0]

# 3D system 
nd = length(x₀)

tstep = 0.001
tspan = (0.001, 100.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

# noise level
eps = 1

# lambda parameter
lambda = 0.025

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

p = (sigma, beta, rho)
prob = ODEProblem(lorenz, x₀, tspan, p)

# stored as dims [states x iters] matrix
data = ODE.solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12, saveat = trange, tstops = trange) 
x = Array(data)

# compute Derivative

# ẋ is of dims [states x iter]
ẋ = Array{Float64}(undef, size(x,1), size(x,2)) 
for i in axes(x,2)
    ẋ[:, i] = lorenz(x[:, i], p, 0)
end

# add noise
ẋ = ẋ + eps*rand(Normal(), size(ẋ))

# ------------------------------------------------------------
# Pool Data (evaluate library of candidate basis functions on training data)
# ------------------------------------------------------------

println("Pool Data...")

Θ = evaluate(x, basis)


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

println("Sparsify Dynamics...")

 Ξ = sparsify_dynamics(Θ, ẋ, lambda)
#Ξ = sparsify_dynamics(Θ, ẋ, lambda; solver = OptimSolver())

#println(Ξ)

#println("   maximum(Ξ) = ", maximum(Ξ))

# To look at data output of pool
poolDataLIST(["x", "y", "z"], Ξ, nd, polyorder, usesine)

# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System...")

# With T_end=20.0
# FIGURE 1: LORENZ for T in[0,20]
# True model:
tspan = (0.0, 20.0)
p = (sigma, beta, rho)
prob = ODEProblem(lorenz, x₀, tspan, p)
# stored as dims [states x iters] matrix 
xA = ODE.solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12) 

# Approximate model:
p = (Ξ = Ξ, basis = basis)
prob_approx = ODEProblem(sparse_galerkin!, x₀, tspan, p)
xB = ODE.solve(prob_approx, Tsit5(), abstol=1e-12, reltol=1e-12)

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
tspan = (0.0, 250.0)
p = (sigma, beta, rho)
prob = ODEProblem(lorenz, x₀, tspan, p)
xA = ODE.solve(prob, abstol=1e-6, reltol=1e-6) #stored as dims [3 x iters] matrix 

# Approximate model:
p = (Ξ = Ξ, basis = basis)
prob_approx = ODEProblem(sparse_galerkin!, x₀, tspan, p)
xB = ODE.solve(prob_approx, Tsit5(), abstol=1e-6, reltol=1e-6)

# Figure 3:
dtA = [0; diff(xA.t)]
dtB = [0; diff(xB.t)]
p5 = plot(xA[1,:], xA[2,:], xA[3,:], zcolor=dtA, xlabel="x", ylabel="y", zlabel="z", label="true")
p6 = plot(xB[1,:], xB[2,:], xB[3,:], zcolor=dtB, xlabel="x", ylabel="y", zlabel="z", label="approx")
display(plot(p5, p6, layout=(1,2), show = true, reuse = false, size=(1000,1000)))

savefig("Fig3_ex2_lorenz.png")