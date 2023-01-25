
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

# search space up to third order polynomials
polyorder = 3

# initial function x₀ = [q₁, q₂, p₁, p₂]
x₀ = [2., 1., 1., 0.5]

# let a vector be ones initially of length 34 (b/c 34 is number of poly combinations for 2 variables, with 2 dims of highest order 3)
a = ones(34)

# 2 dims each of p and q gives 4 variables
out = zeros(4)

# no trigonometric functions
usesine = false

# 2D system 
nd = length(x₀)

tstep = 0.01
tspan = (0.0,25.0)
trange = range(tspan[begin], step = tstep, stop = tspan[end])

# noise level
eps = 0.05

# lambda parameter
lambda = 0.05

# initialize analytical function, keep ϵ bigger than lambda so system is identifiable
ϵ = 0.5
m = 1

# one-dim simple harmonic oscillator (not used anywhere only in case some testing needed)
# H_ana(x, p, t) = 1/(2*m) * x[3]^2 + ϵ * x[1]^2 + 0 * x[2] + 0 * x[4]

#grad_H_ana(x, p, t) = [x[3]; x[4]; -2ϵ * x[1]; -2ϵ * x[2]]
grad_H_ana(x, p, t) = [x[3]; 0.0; -2ϵ * x[1]; 0.0]
 

# ------------------------------------------------------------
# Training Data
# ------------------------------------------------------------

println("Generate Training Data...")

prob = ODEProblem(grad_H_ana, x₀, tspan)
data = ODE.solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange)
x = Array(data)

# compute vector field from x state values at each timestep
# stored as matrix with dims [nd,ntime]
ẋ = zero(x)

for i in axes(ẋ,2)
    ẋ[:,i] = grad_H_ana(x[:,i], 0, 0)
end

# add noise
ẋ .+= eps .* randn(size(x))

# ------------------------------------------------------------
# Pool Data (evaluate library of candidate basis functions on training data)
# ------------------------------------------------------------

#println("Pool Data...")

#θ = hamil_basis_maker(x, polyorder)



#########################################################################
# HAM = hamiltonianFunction(x₀, a)
# # calcuate Hessian of Hamiltonian w.r.t (a) coeffecients
# ∂H²∂a²(x,a) = ForwardDiff.hessian(a -> hamiltonianFunction(x, a), a)
# hess_a = ∂H²∂a²(x₀,a)
#########################################################################


# ----------------------------------------
# Compute Sparse Regression
# ----------------------------------------

print("Sparsify Dynamics...")

function loss(a::AbstractVector)
    result = ones(eltype(a), size(x))
    out = zeros(eltype(a), size(x, 1))
    for i in 1:size(x, 2)
        result[:,i] = hamilGradient!(out, x[:,i], a, 0)
    end

    return mapreduce( y -> y^2, +, (ẋ) .- result)
end

a .= sparsify_hamiltonian_dynamics(a, loss, lambda)

println(a)



# ----------------------------------------
# Integrate Identified System
# ----------------------------------------

println("Integrate Identified System...")

prob_approx = ODEProblem(hamilGradient!, x₀, tspan, a)
xid = ODE.solve(prob_approx, Tsit5(), abstol=1e-10, reltol=1e-10, saveat = trange, tstops = trange) 



# ----------------------------------------
# Plot Results
# ----------------------------------------

println("Plotting...")

p1 = plot()
plot!(p1, data.t, data[1,:], label = "Data")
plot!(p1, xid.t, xid[1,:], label = "Identified")

p3 = plot()
plot!(p3, data.t, data[3,:], label = "Data")
plot!(p3, xid.t, xid[3,:], label = "Identified")

display(plot(p1, p3))
savefig("Oscillator_Hamiltonian_Dynamics.png")