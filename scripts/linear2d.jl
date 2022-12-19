
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

# generate basis
basis = CompoundBasis()

# initial datra
x₀ = [2., 0.]

# 2D system 
nd = length(x₀)

tstep = 0.01
# tspan = (0.0,0.05)
tspan = (0.0,25.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

# noise level
eps = 0.05 

# lambda parameter
lambda = 0.05

# vector field
const A = [-0.1  2.0
           -2.0 -0.1]

rhs(x,p,t) = A*x


# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

prob = ODEProblem(rhs, x₀, tspan)
data = ODE.solve(prob, abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)
x = Array(data)
# stored as matrix with dims [nd,ntime]

# compute vector field from x state values at each timestep
# stored as matrix with dims [nd,ntime]
ẋ = zero(x)
for i in axes(ẋ,2)
    ẋ[:,i] .= A*x[:,i]
end

# add noise
# dx .+= eps .* randn(size(x))

# println("x = ", x)
# println("ẋ = ", ẋ)


# ------------------------------------------------------------
# Pool Data (evaluate library of candidate basis functions on training data)
# ------------------------------------------------------------

println("Pool Data...")

Θ = evaluate(x, basis)


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

print("Sparsify Dynamics...")

# Ξ = sparsify_dynamics(Θ, ẋ, lambda, 0)
Ξ = sparsify_dynamics(Θ, ẋ, lambda)
# Ξ = sparsify_dynamics(Θ, ẋ, lambda; solver = OptimSolver())

println(Ξ)

println("   maximum(Ξ) = ", maximum(Ξ))


# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System...")

p = (Ξ = Ξ, basis = basis)

prob_approx = ODEProblem(sparse_galerkin, x₀, tspan, p)
xid = ODE.solve(prob_approx, ode4(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange) 


# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

p1 = plot()
plot!(p1, data.t, data[1,:], label = "Data")
plot!(p1, xid.t, xid[1,:], label = "Identified")

p2 = plot()
plot!(p2, data.t, data[2,:], label = "Data")
plot!(p2, xid.t, xid[2,:], label = "Identified")

plot(p1, p2)
savefig("linear2d.png")
