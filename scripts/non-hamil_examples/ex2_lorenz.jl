
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

# p = (sigma, beta, rho)
# prob = ODEProblem(lorenz, x₀, tspan, p)

# # stored as dims [states x iters] matrix
# data = ODE.solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12, saveat = trange, tstops = trange) 
# x = Array(data)


num_samp = 15

# samples in p and q space
samp_range = LinRange(-20, 20, num_samp)

# initialize vector of matrices to store ODE solve output
# s depend on size of nd (total dims), 4 in the case here so we use samp_range x samp_range x samp_range x samp_range
s = collect(Iterators.product(fill(samp_range, nd)...))

# compute vector field from x state values
x = [collect(s[i]) for i in eachindex(s)]

x = hcat(x...)

# compute Derivative

# ẋ is of dims [states x iter]
ẋ = Array{Float64}(undef, size(x,1), size(x,2)) 
for i in axes(x,2)
    ẋ[:, i] = lorenz(x[:, i], p, 0)
end

# add noise
ẋ = ẋ + eps*rand(Normal(), size(ẋ))

# collect training data
tdata = TrainingData(Float32.(x), Float32.(ẋ))

# choose SINDy method
method = SINDy(lambda = 0.05, noise_level = 0.0)


println("Computing Vector Field...")

# compute vector field using least squares regression (/) solver
vectorfield = VectorField(method, basis, tdata)

#Using BFGS() solver
# vectorfield = VectorField(method, basis, tdata, solver = OptimSolver())


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
